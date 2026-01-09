#include "seam_straightening.h"
#include "mesh.h"
#include "logging.h"
#include "rdp.h"
#include "harmonic_map_utils.h"
#include <vcg/complex/algorithms/update/topology.h>
#include <vcg/complex/algorithms/clean.h>
#include <vcg/complex/algorithms/outline_support.h>
#include <vcg/math/disjoint_set.h>
#include <map>
#include <set>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <iomanip>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cmath>
#include <array>
#include <Eigen/StdVector>

extern "C" {
#include "../vcglib/wrap/triangle/triangle.h"
}

namespace UVDefrag {

struct Point2D { double x, y; };

struct SeamChain {
    std::vector<Mesh::VertexPointer> vertices;
    std::vector<std::pair<Mesh::FacePointer, int>> edges; 
    ChartHandle chart;
    bool isSeam = false;
    bool isTrueBoundary = false;
    int twinChainIdx = -1;
    bool twinIsReversed = false;
};

struct BoundaryVertexInfo {
    bool isImmutable = false;
    bool isIsolated = false; // Only pin isolated triangles, not ears
    bool isEar = false; // Incident on a face with 2 boundary/seam edges
    std::vector<std::pair<Mesh::FacePointer, int>> incidentBoundaryEdges;
    std::vector<Mesh::VertexPointer> geometricTwins; // Vertices at same 3D position
};

// Global Snapshot for Thread-Safe RDP
using GlobalUVSnapshot = std::unordered_map<Mesh::FacePointer, std::array<Point2d, 3>>;

Point2d GetSnapshotUV(const GlobalUVSnapshot& snapshot, const SeamChain& chain, int vertexIdx) {
    if (vertexIdx < (int)chain.edges.size()) {
        auto& edge = chain.edges[vertexIdx];
        return snapshot.at(edge.first)[edge.second];
    } else {
        auto& edge = chain.edges.back();
        return snapshot.at(edge.first)[(edge.second + 1) % 3];
    }
}

void IntegrateSeamStraightening(GraphHandle graph, const SeamStraighteningParameters& params) {
    if (params.initialTolerance <= 0.0) {
        LOG_INFO << "Seam straightening skipped (tolerance is zero or negative)";
        return;
    }

    Mesh& m = graph->mesh;
    tri::UpdateTopology<Mesh>::FaceFace(m);
    tri::UpdateTopology<Mesh>::VertexFace(m);

    LOG_INFO << "Phase 1: Boundary Classification and Geometric Twin Finding...";
    LOG_INFO << "  Geometric Tolerance: " << params.geometricTolerance;

    // Diagnostic Counters
    struct DiagStats {
        int v_total = 0;
        int v_isolated = 0;
        int v_hole = 0;
        int v_nonmanifold = 0;
        int v_junction = 0;
        int v_mutable = 0;
        int chains_total = 0;
        int chains_matched = 0;
        double len_avg = 0;
    } diag;

    int numChartsAttempted = 0;
    int numChartsSucceeded = 0;
    int numInversions = 0;
    int numWarpFailures = 0;
    int totalSegmentsBefore = 0;
    int totalSegmentsAfter = 0;
    int chartsNoWork = 0;
    int chartsWithWork = 0;
    int chartsFailedAllAttempts = 0;
    int chartsReachedPinnedStage = 0;
    int chartsSucceededRegular = 0;
    int chartsSucceededPinned = 0;
    long long attemptsTotal = 0;      // counts attempts (not charts)
    long long attemptsToSuccess = 0;  // counts attempts-to-first-success (successful charts only)

    struct TierStats {
        int count = 0;
        int success = 0;
        int successPinnedEars = 0;
        int noWork = 0;
        int withWork = 0;
        int failedAllAttempts = 0;
        int reachedPinnedStage = 0;
        int succeededRegular = 0;
        int succeededPinned = 0;
        int before = 0;
        int after = 0;
        int inversions = 0;    // attempt-level
        int warpFailures = 0;  // attempt-level
        long long attemptsTotal = 0;     // attempt-level
        long long attemptsToSuccess = 0; // chart-level sum of attempts-to-first-success
        std::vector<int> successPerAttempt;
    };
    std::vector<TierStats> tiers(10);
    std::vector<double> binThresholds(9);
    for (auto& t : tiers) t.successPerAttempt.assign(std::max(2, params.maxWarpAttempts * 2), 0);

    std::map<Mesh::VertexPointer, BoundaryVertexInfo> vertexInfo;
    
    auto isBoundary = [&](Mesh::FacePointer f, int e) {
        Mesh::FacePointer ff = f->FFp(e);
        if (ff == f) return true; // Hole or Cut
        if (ff->id != f->id) return true; // Seam
        return false;
    };

    // 1. Initial Pass: Collect Boundary Vertices
    std::vector<Mesh::VertexPointer> boundaryVertices;
    for (auto& entry : graph->charts) {
        ChartHandle chart = entry.second;
        for (auto f : chart->fpVec) {
            int boundaryEdgesInFace = 0;
            for (int i = 0; i < 3; ++i) {
                if (isBoundary(f, i)) {
                    boundaryEdgesInFace++;
                    Mesh::VertexPointer v0 = f->V(i);
                    Mesh::VertexPointer v1 = f->V((i + 1) % 3);
                    
                    if (vertexInfo.find(v0) == vertexInfo.end()) boundaryVertices.push_back(v0);
                    if (vertexInfo.find(v1) == vertexInfo.end()) boundaryVertices.push_back(v1);

                    vertexInfo[v0].incidentBoundaryEdges.push_back({f, i});
                    vertexInfo[v1].incidentBoundaryEdges.push_back({f, i});
                }
            }
            
            // Standard ears (2 boundary edges) are left mutable to allow straightening initially,
            // but tracked for fallback pinning.
            if (boundaryEdgesInFace == 3) {
                for (int i = 0; i < 3; ++i) vertexInfo[f->V(i)].isIsolated = true;
            } else if (boundaryEdgesInFace == 2) {
                for (int i = 0; i < 3; ++i) vertexInfo[f->V(i)].isEar = true;
            }
        }
    }

    // 2. Geometric Twin Finding (Sort-and-Sweep with Transitive Welding)
    std::vector<Mesh::VertexPointer> sortedVertices = boundaryVertices;
    std::sort(sortedVertices.begin(), sortedVertices.end(), [](Mesh::VertexPointer a, Mesh::VertexPointer b) {
        if (a->P().X() != b->P().X()) return a->P().X() < b->P().X();
        if (a->P().Y() != b->P().Y()) return a->P().Y() < b->P().Y();
        return a->P().Z() < b->P().Z();
    });

    vcg::DisjointSet<MeshVertex> ds;
    for (auto v : sortedVertices) ds.MakeSet(v);

    for (size_t i = 0; i < sortedVertices.size(); ++i) {
        for (size_t j = i + 1; j < sortedVertices.size(); ++j) {
            // Early out based on X-axis distance
            if (sortedVertices[j]->P().X() - sortedVertices[i]->P().X() > params.geometricTolerance) break;
            
            // Use <= to allow exact welding when tolerance is 0.0
            if (Distance(sortedVertices[i]->P(), sortedVertices[j]->P()) <= params.geometricTolerance) {
                ds.Union(sortedVertices[i], sortedVertices[j]);
            }
        }
    }

    // Map each vertex to a unique Geometric ID
    std::map<Mesh::VertexPointer, int> vertexToUniqueId;
    std::map<Mesh::VertexPointer, int> dsToUniqueId;
    int nextUniqueId = 0;

    for (size_t i = 0; i < sortedVertices.size(); ++i) {
        Mesh::VertexPointer root = ds.FindSet(sortedVertices[i]);
        if (dsToUniqueId.find(root) == dsToUniqueId.end()) {
            dsToUniqueId[root] = nextUniqueId++;
        }
        vertexToUniqueId[sortedVertices[i]] = dsToUniqueId[root];
    }
    
    LOG_INFO << "  Welding Results: " << sortedVertices.size() << " boundary wedges -> " << nextUniqueId << " unique geometric vertices.";

    // Populate geometricTwins
    std::map<int, std::vector<Mesh::VertexPointer>> idToVertices;
    for (auto const& pair : vertexToUniqueId) idToVertices[pair.second].push_back(pair.first);

    for (auto const& pair : idToVertices) {
        const std::vector<Mesh::VertexPointer>& group = pair.second;
        for (auto v : group) {
            for (auto other : group) {
                if (v != other) vertexInfo[v].geometricTwins.push_back(other);
            }
        }
    }

    // 3. Build Geometric Boundary Graph to classify Holes vs Seams
    std::map<std::pair<int, int>, int> geometricEdgeCounts;

    for (auto& pair : vertexInfo) {
        Mesh::VertexPointer v = pair.first;
        int idStart = vertexToUniqueId[v];

        for (auto& be : pair.second.incidentBoundaryEdges) {
            Mesh::VertexPointer v0 = be.first->V(be.second);
            Mesh::VertexPointer v1 = be.first->V((be.second + 1) % 3);
            
            if (v0 == v) {
                int idEnd = vertexToUniqueId[v1];
                if (idStart == idEnd) continue; 
                std::pair<int, int> key = std::minmax(idStart, idEnd);
                geometricEdgeCounts[key]++;
            }
        }
    }

    // 4. Determine Immutability & Run Diagnostics
    std::vector<bool> idIsImmutable(nextUniqueId, false);
    
    for (auto const& groupPair : idToVertices) {
        int id = groupPair.first;
        diag.v_total++;
        
        // Check Isolated status
        bool groupIsIsolated = false;
        for (auto v : groupPair.second) {
            if (vertexInfo[v].isIsolated) { groupIsIsolated = true; break; }
        }
        
        if (groupIsIsolated) {
            idIsImmutable[id] = true;
            diag.v_isolated++;
            continue;
        }

        std::set<int> neighbors;
        bool hasHoleEdge = false;
        bool hasNonManifoldEdge = false; 

        // Gather neighbors from all twins
        for (auto v : groupPair.second) {
             for (auto& be : vertexInfo[v].incidentBoundaryEdges) {
                Mesh::VertexPointer v0 = be.first->V(be.second);
                Mesh::VertexPointer v1 = be.first->V((be.second + 1) % 3);
                
                Mesh::VertexPointer neighborV = (v0 == v) ? v1 : v0;
                int neighborId = vertexToUniqueId[neighborV];
                if (id == neighborId) continue;

                neighbors.insert(neighborId);
                
                std::pair<int, int> key = std::minmax(id, neighborId);
                int count = geometricEdgeCounts[key];
                
                if (count == 1) hasHoleEdge = true;
                if (count > 2) hasNonManifoldEdge = true;
             }
        }

        // Immutability Logic
        if (hasHoleEdge) {
            idIsImmutable[id] = true;
            diag.v_hole++;
        } else if (hasNonManifoldEdge) {
            idIsImmutable[id] = true;
            diag.v_nonmanifold++;
        } else if (neighbors.size() != 2) {
            idIsImmutable[id] = true; 
            diag.v_junction++;
        } else {
            idIsImmutable[id] = false; 
            diag.v_mutable++;
        }
    }

    for (auto& pair : vertexInfo) {
        pair.second.isImmutable = idIsImmutable[vertexToUniqueId[pair.first]];
    }

    // 5. Extract Chains (Maximal)
    std::vector<SeamChain> chains;
    std::set<std::pair<Mesh::FacePointer, int>> visitedEdges;

    for (auto& entry : graph->charts) {
        ChartHandle chart = entry.second;
        for (auto f : chart->fpVec) {
            for (int i = 0; i < 3; ++i) {
                if (isBoundary(f, i) && visitedEdges.find({f, i}) == visitedEdges.end()) {
                    
                    std::pair<Mesh::FacePointer, int> currEdge = {f, i};
                    
                    // Backtrack to find start
                    while (true) {
                        Mesh::VertexPointer vStart = currEdge.first->V(currEdge.second);
                        if (vertexInfo[vStart].isImmutable) break; 

                        std::pair<Mesh::FacePointer, int> prevEdge = {nullptr, -1};
                        for (auto& inc : vertexInfo[vStart].incidentBoundaryEdges) {
                            if (inc.first->id != chart->id) continue;
                            Mesh::VertexPointer eEnd = inc.first->V((inc.second + 1) % 3);
                            if (eEnd == vStart) {
                                if (inc == currEdge) continue; 
                                prevEdge = inc;
                                break;
                            }
                        }

                        if (prevEdge.first == nullptr) break; 
                        
                        if (visitedEdges.find(prevEdge) != visitedEdges.end()) {
                             bool inCurrentLoop = false;
                             if(prevEdge == std::make_pair(f,i)) inCurrentLoop = true; 
                             if(inCurrentLoop) break; 
                             break; 
                        }
                        
                        currEdge = prevEdge;
                        if (currEdge == std::make_pair(f,i)) break; 
                    }

                    // Trace forward
                    if (visitedEdges.find(currEdge) != visitedEdges.end()) continue;

                    SeamChain chain;
                    chain.chart = chart;
                    std::vector<std::pair<Mesh::FacePointer, int>> path;
                    
                    std::pair<Mesh::FacePointer, int> trace = currEdge;
                    path.push_back(trace);
                    visitedEdges.insert(trace);

                    while (true) {
                        Mesh::VertexPointer vEnd = trace.first->V((trace.second + 1) % 3);
                        
                        if (vertexInfo[vEnd].isImmutable) break;
                        if (vEnd == path.front().first->V(path.front().second)) break; 

                        std::pair<Mesh::FacePointer, int> next = {nullptr, -1};
                        for (auto& inc : vertexInfo[vEnd].incidentBoundaryEdges) {
                            if (inc.first->id != chart->id) continue;
                            if (visitedEdges.find(inc) != visitedEdges.end()) continue;
                            if (inc.first->V(inc.second) == vEnd) {
                                next = inc;
                                break;
                            }
                        }

                        if (next.first == nullptr) break;
                        path.push_back(next);
                        visitedEdges.insert(next);
                        trace = next;
                    }

                    for (auto& e : path) {
                        chain.vertices.push_back(e.first->V(e.second));
                        chain.edges.push_back(e);
                    }
                    chain.vertices.push_back(path.back().first->V((path.back().second + 1) % 3));
                    
                    chain.isSeam = true; 
                    chains.push_back(chain);
                }
            }
        }
    }

    // 6. Chain Matching
    struct ChainKey {
        std::vector<int> pointIds;
        bool operator<(const ChainKey& o) const {
            if (pointIds.size() != o.pointIds.size()) return pointIds.size() < o.pointIds.size();
            return pointIds < o.pointIds;
        }
    };
    
    std::map<ChainKey, int> idToChain;
    int numTwinPairs = 0;

    for (int i = 0; i < (int)chains.size(); ++i) {
        if (!chains[i].isSeam) continue;
        if (chains[i].twinChainIdx != -1) continue;

        ChainKey key;
        for (auto v : chains[i].vertices) key.pointIds.push_back(vertexToUniqueId[v]);
        
        if (idToChain.count(key)) {
            int j = idToChain[key];
            chains[i].twinChainIdx = j;
            chains[j].twinChainIdx = i;
            chains[i].twinIsReversed = false;
            chains[j].twinIsReversed = false;
            numTwinPairs++;
        } else {
            std::reverse(key.pointIds.begin(), key.pointIds.end());
            if (idToChain.count(key)) {
                int j = idToChain[key];
                chains[i].twinChainIdx = j;
                chains[j].twinChainIdx = i;
                chains[i].twinIsReversed = true;
                chains[j].twinIsReversed = true;
                numTwinPairs++;
            } else {
                std::reverse(key.pointIds.begin(), key.pointIds.end()); 
                idToChain[key] = i;
            }
        }
    }
    
    // Diagnostic Output
    diag.chains_total = chains.size();
    diag.chains_matched = numTwinPairs * 2;
    long long total_verts = 0;
    for(auto& c : chains) total_verts += c.vertices.size();
    diag.len_avg = chains.empty() ? 0 : (double)total_verts / chains.size();

    LOG_INFO << "=== SEAM DIAGNOSTICS ===";
    LOG_INFO << "Geometric Vertices: " << diag.v_total;
    LOG_INFO << "  Immutable (Isolated): " << diag.v_isolated << " (" << std::fixed << std::setprecision(1) << (100.0*diag.v_isolated/diag.v_total) << "%)";
    LOG_INFO << "  Immutable (Hole): " << diag.v_hole << " (" << (100.0*diag.v_hole/diag.v_total) << "%)";
    LOG_INFO << "  Immutable (Non-Manifold): " << diag.v_nonmanifold << " (" << (100.0*diag.v_nonmanifold/diag.v_total) << "%)";
    LOG_INFO << "  Immutable (Junction/Terminus): " << diag.v_junction << " (" << (100.0*diag.v_junction/diag.v_total) << "%)";
    LOG_INFO << "  Mutable: " << diag.v_mutable << " (" << (100.0*diag.v_mutable/diag.v_total) << "%)";
    LOG_INFO << "Chains: " << diag.chains_total << " (Matched: " << diag.chains_matched << ")";
    LOG_INFO << "Avg Chain Length (verts): " << diag.len_avg;
    LOG_INFO << "========================";

    // 7. Bin Setup for Tiered Logging
    std::vector<double> chartAreas3D;
    for (auto& entry : graph->charts) chartAreas3D.push_back(entry.second->Area3D());
    std::sort(chartAreas3D.begin(), chartAreas3D.end());
    if (!chartAreas3D.empty()) {
        for (int i = 0; i < 9; ++i) {
            int idx = (int)((i + 1) * chartAreas3D.size() / 10);
            binThresholds[i] = chartAreas3D[std::min(idx, (int)chartAreas3D.size() - 1)];
        }
    }
    auto getBin = [&](double area) {
        for (int i = 0; i < 9; ++i) if (area < binThresholds[i]) return i;
        return 9;
    };

    // 8. Processing Loop (Parallel)
    std::map<ChartHandle, std::vector<int>> chartToChains;
    for(int i = 0; i < (int)chains.size(); ++i) chartToChains[chains[i].chart].push_back(i);

    std::vector<ChartHandle> chartVec;
    chartVec.reserve(graph->charts.size());
    for(auto& entry : graph->charts) chartVec.push_back(entry.second);

    GlobalUVSnapshot globalSnapshot;
    std::vector<double> chartMedianAreas;
    for (auto& entry : graph->charts) {
        std::vector<double> faceAreas;
        for (auto f : entry.second->fpVec) {
            Point2d p0 = f->WT(0).P();
            Point2d p1 = f->WT(1).P();
            Point2d p2 = f->WT(2).P();
            globalSnapshot[f] = { p0, p1, p2 };
            
            if (params.adaptiveTolerance) {
                double area = std::abs((p1.X() - p0.X()) * (p2.Y() - p0.Y()) - (p1.Y() - p0.Y()) * (p2.X() - p0.X())) * 0.5;
                if (area > 1e-12) faceAreas.push_back(area);
            }
        }
        if (params.adaptiveTolerance && !faceAreas.empty()) {
            size_t mid = faceAreas.size() / 2;
            std::nth_element(faceAreas.begin(), faceAreas.begin() + mid, faceAreas.end());
            chartMedianAreas.push_back(faceAreas[mid]);
        }
    }

    double globalReferenceArea = 1.0;
    if (params.adaptiveTolerance && !chartMedianAreas.empty()) {
        size_t mid = chartMedianAreas.size() / 2;
        std::nth_element(chartMedianAreas.begin(), chartMedianAreas.begin() + mid, chartMedianAreas.end());
        globalReferenceArea = chartMedianAreas[mid];
        LOG_INFO << "Adaptive tolerance enabled. Global baseline (median of chart-median areas): " << globalReferenceArea;
    }

    // Pre-calculate base chart tolerances
    std::map<ChartHandle, double> chartToTol;
    for (auto chart : chartVec) {
        double tol = params.initialTolerance;
        if (params.adaptiveTolerance) {
            std::vector<double> localAreas;
            for (auto f : chart->fpVec) {
                const auto& uvs = globalSnapshot.at(f);
                double area = std::abs((uvs[1].X() - uvs[0].X()) * (uvs[2].Y() - uvs[0].Y()) - (uvs[1].Y() - uvs[0].Y()) * (uvs[2].X() - uvs[0].X())) * 0.5;
                if (area > 1e-12) localAreas.push_back(area);
            }
            if (!localAreas.empty()) {
                size_t mid = localAreas.size() / 2;
                std::nth_element(localAreas.begin(), localAreas.begin() + mid, localAreas.end());
                tol *= std::sqrt(localAreas[mid] / globalReferenceArea);
            }
        }
        // Apply safety bounds
        tol = std::max(params.minTolerancePixels, std::min(params.maxTolerancePixels, tol));
        chartToTol[chart] = tol;
    }

    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#endif
    LOG_INFO << "Phase 2 & 3: Straightening & Warping (Parallel " << num_threads << " threads)...";

    // 9. Pre-Simplify Chains for Consistency across Charts
    // Lod index k corresponds to attempt k (halving tolerance each time)
    std::vector<std::vector<std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>>> chainLODs(chains.size(), 
        std::vector<std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>>(params.maxWarpAttempts));

    for (int ci = 0; ci < (int)chains.size(); ++ci) {
        const auto& chain = chains[ci];
        int twinIdx = chain.twinChainIdx;
        
        // Ensure we only process each pair once for consistency, but populate both slots
        if (twinIdx != -1 && twinIdx < ci) continue; 

        double baseTol = chartToTol[chain.chart];
        if (twinIdx != -1) {
            baseTol = std::min(baseTol, chartToTol[chains[twinIdx].chart]);
        }

        // Prepare input polyline
        std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> poly4D;
        if (twinIdx != -1) {
            const auto& chainB = chains[twinIdx];
            bool reversed = chain.twinIsReversed;
            for (int k = 0; k < (int)chain.vertices.size(); ++k) {
                int kB = reversed ? (chainB.vertices.size() - 1 - k) : k;
                Point2d pA = GetSnapshotUV(globalSnapshot, chain, k);
                Point2d pB = GetSnapshotUV(globalSnapshot, chainB, kB); 
                poly4D.push_back(Eigen::Vector4d(pA.X(), pA.Y(), pB.X(), pB.Y()));
            }
        } else {
             for (int k = 0; k < (int)chain.vertices.size(); ++k) {
                Point2d pA = GetSnapshotUV(globalSnapshot, chain, k);
                poly4D.push_back(Eigen::Vector4d(pA.X(), pA.Y(), 0, 0));
            }
        }

        for (int attempt = 0; attempt < params.maxWarpAttempts; ++attempt) {
            double tol = baseTol * std::pow(0.5, attempt);
            std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> simplified = GeometryUtils::simplifyRDP(poly4D, tol);
            chainLODs[ci][attempt] = simplified;
            
            if (twinIdx != -1) {
                // For twin, the 4D points need to be swapped (pB is now pA) and potentially reversed
                std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> twinSimplified;
                bool reversed = chain.twinIsReversed;
                if (reversed) {
                    for (int k = (int)simplified.size() - 1; k >= 0; --k) {
                        const auto& p4 = simplified[k];
                        twinSimplified.push_back(Eigen::Vector4d(p4.z(), p4.w(), p4.x(), p4.y()));
                    }
                } else {
                    for (size_t k = 0; k < simplified.size(); ++k) {
                        const auto& p4 = simplified[k];
                        twinSimplified.push_back(Eigen::Vector4d(p4.z(), p4.w(), p4.x(), p4.y()));
                    }
                }
                chainLODs[twinIdx][attempt] = twinSimplified;
            }
        }
    }

    std::vector<int> retryStats(params.maxWarpAttempts, 0);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < (int)chartVec.size(); ++i) {
        ChartHandle chart = chartVec[i];
        int binIdx = getBin(chart->Area3D());
        const int maxTotalAttempts = params.maxWarpAttempts * 2;
        
#ifdef _OPENMP
        #pragma omp atomic
#endif
        numChartsAttempted++;
#ifdef _OPENMP
        #pragma omp atomic
#endif
        tiers[binIdx].count++;

        bool success = false;
        int successAttempt = -1;
        bool successWasPinned = false;
        int attemptsRun = 0;
        
        auto chainIt = chartToChains.find(chart);
        if (chainIt == chartToChains.end()) {
             // No boundary/seam chains in this chart => nothing to warp/straighten.
             success = true;
#ifdef _OPENMP
            #pragma omp atomic
#endif
            chartsNoWork++;
#ifdef _OPENMP
            #pragma omp atomic
#endif
            tiers[binIdx].noWork++;
        } else {
            const std::vector<int>& myChains = chainIt->second;
#ifdef _OPENMP
            #pragma omp atomic
#endif
            chartsWithWork++;
#ifdef _OPENMP
            #pragma omp atomic
#endif
            tiers[binIdx].withWork++;

            for (int attempt = 0; attempt < maxTotalAttempts; ++attempt) {
                attemptsRun++;
                bool pinEars = (attempt >= params.maxWarpAttempts);
                int subAttempt = attempt % params.maxWarpAttempts;
                
                std::map<Mesh::VertexPointer, Point2D> boundaryTargets;
                int currentBefore = 0;
                int currentAfter = 0;

                for (int ci : myChains) {
                    const auto& chain = chains[ci];
                    const auto& simplified = chainLODs[ci][subAttempt];
                    
                    currentBefore += (int)chain.vertices.size() - 1;
                    currentAfter += (int)simplified.size() - 1;

                    auto computeArcLengths = [](const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& poly) {
                        std::vector<double> lengths(poly.size(), 0.0);
                        for (size_t k = 1; k < poly.size(); ++k) lengths[k] = lengths[k - 1] + (poly[k] - poly[k - 1]).norm();
                        return lengths;
                    };
                    
                    // Input 4D points for this chain (at full resolution)
                    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> poly4D_full;
                    if (chain.twinChainIdx != -1) {
                        const auto& chainB = chains[chain.twinChainIdx];
                        bool reversed = chain.twinIsReversed;
                        for (int k = 0; k < (int)chain.vertices.size(); ++k) {
                            int kB = reversed ? (chainB.vertices.size() - 1 - k) : k;
                            Point2d pA = GetSnapshotUV(globalSnapshot, chain, k);
                            Point2d pB = GetSnapshotUV(globalSnapshot, chainB, kB); 
                            poly4D_full.push_back(Eigen::Vector4d(pA.X(), pA.Y(), pB.X(), pB.Y()));
                        }
                    } else {
                        for (int k = 0; k < (int)chain.vertices.size(); ++k) {
                            Point2d pA = GetSnapshotUV(globalSnapshot, chain, k);
                            poly4D_full.push_back(Eigen::Vector4d(pA.X(), pA.Y(), 0, 0));
                        }
                    }

                    std::vector<double> origLengths = computeArcLengths(poly4D_full);
                    std::vector<double> simplifiedLengths = computeArcLengths(simplified);
                    double totalLen = origLengths.back();
                    double simplifiedTotalLen = simplifiedLengths.back();

                    for (int k = 0; k < (int)chain.vertices.size(); ++k) {
                        Mesh::VertexPointer v = chain.vertices[k];
                        Point2d startUV = GetSnapshotUV(globalSnapshot, chain, k);
                        
                        if (pinEars && vertexInfo[v].isEar) {
                            boundaryTargets[v] = { startUV.X(), startUV.Y() };
                            continue;
                        }

                        double l = origLengths[k];
                        Eigen::Vector4d p4;
                        if (simplifiedTotalLen < 1e-12) p4 = simplified[0];
                        else {
                            double targetL = (l / totalLen) * simplifiedTotalLen;
                            auto it = std::lower_bound(simplifiedLengths.begin(), simplifiedLengths.end(), targetL);
                            int idx = std::distance(simplifiedLengths.begin(), it);
                            if (idx == 0) p4 = simplified.front();
                            else if (idx == (int)simplified.size()) p4 = simplified.back();
                            else {
                                double t = (targetL - simplifiedLengths[idx - 1]) / (simplifiedLengths[idx] - simplifiedLengths[idx - 1]);
                                p4 = (1.0 - t) * simplified[idx - 1] + t * simplified[idx];
                            }
                        }
                        boundaryTargets[v] = { p4.x(), p4.y() };
                    }
                }

                vcg::Box2d bbox = chart->UVBox();
                double dimX = std::max(bbox.DimX(), 1e-6);
                double dimY = std::max(bbox.DimY(), 1e-6);
                bbox.min -= vcg::Point2d(dimX * 0.2, dimY * 0.2);
                bbox.max += vcg::Point2d(dimX * 0.2, dimY * 0.2);

                struct triangulateio in, out;
                memset(&in, 0, sizeof(in));
                memset(&out, 0, sizeof(out));

                struct RawPt { Mesh::VertexPointer v; double x, y; };
                std::vector<RawPt> rawPoints;
                rawPoints.reserve(chart->fpVec.size() * 3);

                for (auto f : chart->fpVec) {
                    const auto& uvs = globalSnapshot.at(f);
                    for(int k=0; k<3; ++k) rawPoints.push_back({f->V(k), uvs[k].X(), uvs[k].Y()});
                }

                std::sort(rawPoints.begin(), rawPoints.end(), [](const RawPt& a, const RawPt& b) {
                    if (a.x != b.x) return a.x < b.x;
                    return a.y < b.y;
                });

                std::map<Mesh::VertexPointer, int> vToTriIdx;
                std::vector<Point2D> uniquePoints;
                
                if (!rawPoints.empty()) {
                    uniquePoints.push_back({rawPoints[0].x, rawPoints[0].y});
                    vToTriIdx[rawPoints[0].v] = 0;
                    const double weldThresholdSq = 1e-8; // Robust threshold (~0.0001 pixels squared)
                    for (size_t k = 1; k < rawPoints.size(); ++k) {
                        const auto& prev = uniquePoints.back();
                        double dx = rawPoints[k].x - prev.x;
                        double dy = rawPoints[k].y - prev.y;
                        if (dx*dx + dy*dy < weldThresholdSq) {
                            vToTriIdx[rawPoints[k].v] = (int)uniquePoints.size() - 1;
                        } else {
                            vToTriIdx[rawPoints[k].v] = (int)uniquePoints.size();
                            uniquePoints.push_back({rawPoints[k].x, rawPoints[k].y});
                        }
                    }
                }

                in.numberofpoints = (int)uniquePoints.size() + 4;
                in.pointlist = (TRI_REAL *) malloc(in.numberofpoints * 2 * sizeof(TRI_REAL));
                for (int k = 0; k < (int)uniquePoints.size(); ++k) {
                    in.pointlist[k * 2] = uniquePoints[k].x;
                    in.pointlist[k * 2 + 1] = uniquePoints[k].y;
                }

                int nextIdx = (int)uniquePoints.size();
                in.pointlist[nextIdx * 2] = bbox.min.X(); in.pointlist[nextIdx * 2 + 1] = bbox.min.Y(); int bb0 = nextIdx++;
                in.pointlist[nextIdx * 2] = bbox.max.X(); in.pointlist[nextIdx * 2 + 1] = bbox.min.Y(); int bb1 = nextIdx++;
                in.pointlist[nextIdx * 2] = bbox.max.X(); in.pointlist[nextIdx * 2 + 1] = bbox.max.Y(); int bb2 = nextIdx++;
                in.pointlist[nextIdx * 2] = bbox.min.X(); in.pointlist[nextIdx * 2 + 1] = bbox.max.Y(); int bb3 = nextIdx++;

                in.numberofsegments = 4;
                in.segmentlist = (int *) malloc(in.numberofsegments * 2 * sizeof(int));
                in.segmentlist[0] = bb0; in.segmentlist[1] = bb1;
                in.segmentlist[2] = bb1; in.segmentlist[3] = bb2;
                in.segmentlist[4] = bb2; in.segmentlist[5] = bb3;
                in.segmentlist[6] = bb3; in.segmentlist[7] = bb0;

#ifdef _OPENMP
                #pragma omp critical(triangulate_lib)
#endif
                {
                    triangulate((char*)"zpQ", &in, &out, nullptr);
                }

                std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> ambientVerts(out.numberofpoints);
                for (int k = 0; k < out.numberofpoints; ++k) ambientVerts[k] = Eigen::Vector2d(out.pointlist[k * 2], out.pointlist[k * 2 + 1]);
                std::vector<Triangle> ambientFaces(out.numberoftriangles);
                for (int k = 0; k < out.numberoftriangles; ++k) {
                    ambientFaces[k].v[0] = out.trianglelist[k * 3];
                    ambientFaces[k].v[1] = out.trianglelist[k * 3 + 1];
                    ambientFaces[k].v[2] = out.trianglelist[k * 3 + 2];
                }

                std::vector<int> fixedIndices;
                std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> targetPositions;
                std::vector<bool> isFixed(ambientVerts.size(), false);

                fixedIndices.push_back(bb0); targetPositions.push_back(ambientVerts[bb0]); isFixed[bb0] = true;
                fixedIndices.push_back(bb1); targetPositions.push_back(ambientVerts[bb1]); isFixed[bb1] = true;
                fixedIndices.push_back(bb2); targetPositions.push_back(ambientVerts[bb2]); isFixed[bb2] = true;
                fixedIndices.push_back(bb3); targetPositions.push_back(ambientVerts[bb3]); isFixed[bb3] = true;

                for (auto const& pair : boundaryTargets) {
                    Mesh::VertexPointer v = pair.first;
                    const Point2D& pos = pair.second;
                    int idx = vToTriIdx[v];
                    if (!isFixed[idx]) {
                        fixedIndices.push_back(idx);
                        targetPositions.push_back(Eigen::Vector2d(pos.x, pos.y));
                        isFixed[idx] = true;
                    }
                }

                if (ApplyCompositeWarp(ambientVerts, ambientFaces, fixedIndices, targetPositions)) {
                    std::map<Mesh::VertexPointer, Point2d> computedUVs;
                    for (auto f : chart->fpVec) {
                        for(int k=0; k<3; ++k) {
                            int idx = vToTriIdx[f->V(k)];
                            computedUVs[f->V(k)] = Point2d(ambientVerts[idx].x(), ambientVerts[idx].y());
                        }
                    }

                    bool inverted = false;
                    for (auto f : chart->fpVec) {
                        Point2d p0 = computedUVs[f->V(0)];
                        Point2d p1 = computedUVs[f->V(1)];
                        Point2d p2 = computedUVs[f->V(2)];
                        double area = ((p1.X()-p0.X())*(p2.Y()-p0.Y()) - (p1.Y()-p0.Y())*(p2.X()-p0.X())) / 2.0;
                        if (area < -1e-12) { inverted = true; break; }
                    }
                    
                    if (!inverted) {
                        success = true;
                        successAttempt = attempt;
                        successWasPinned = pinEars;
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        numChartsSucceeded++;
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        tiers[binIdx].success++;
                        
                        if (pinEars) {
#ifdef _OPENMP
                            #pragma omp atomic
#endif
                            tiers[binIdx].successPinnedEars++;
                        }

#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        tiers[binIdx].successPerAttempt[attempt]++;

#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        retryStats[subAttempt]++; // Keep retryStats based on subAttempt for existing logs

#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        totalSegmentsBefore += currentBefore;
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        totalSegmentsAfter += currentAfter;

#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        tiers[binIdx].before += currentBefore;
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        tiers[binIdx].after += currentAfter;

                        chart->isSeamStraightened = true;
                        if (params.colorize) {
                            for (auto f : chart->fpVec) f->C() = vcg::Color4b(255, 165, 0, 255);
                        }
                        for (auto f : chart->fpVec) {
                            for (int k = 0; k < 3; ++k) f->WT(k).P() = computedUVs[f->V(k)];
                        }
                    } else {
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        numInversions++;
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        tiers[binIdx].inversions++;
                    }
                } else {
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    numWarpFailures++;
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    tiers[binIdx].warpFailures++;
                }

                free(in.pointlist); free(in.segmentlist);
                trifree(out.pointlist); trifree(out.trianglelist); trifree(out.segmentlist);
                
                if (success) break;
            }

            // Chart-level accounting (attempt counts etc.)
            if (attemptsRun > params.maxWarpAttempts) {
#ifdef _OPENMP
                #pragma omp atomic
#endif
                chartsReachedPinnedStage++;
#ifdef _OPENMP
                #pragma omp atomic
#endif
                tiers[binIdx].reachedPinnedStage++;
            }

            if (!success) {
#ifdef _OPENMP
                #pragma omp atomic
#endif
                chartsFailedAllAttempts++;
#ifdef _OPENMP
                #pragma omp atomic
#endif
                tiers[binIdx].failedAllAttempts++;
            } else {
                // attempts-to-first-success
                long long atts = (long long)successAttempt + 1;
#ifdef _OPENMP
                #pragma omp atomic
#endif
                attemptsToSuccess += atts;
#ifdef _OPENMP
                #pragma omp atomic
#endif
                tiers[binIdx].attemptsToSuccess += atts;

                if (successWasPinned) {
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    chartsSucceededPinned++;
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    tiers[binIdx].succeededPinned++;
                } else {
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    chartsSucceededRegular++;
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    tiers[binIdx].succeededRegular++;
                }
            }

#ifdef _OPENMP
            #pragma omp atomic
#endif
            attemptsTotal += (long long)attemptsRun;
#ifdef _OPENMP
            #pragma omp atomic
#endif
            tiers[binIdx].attemptsTotal += (long long)attemptsRun;
        }
    }

    LOG_INFO << "Seam Straightening completed:";
    LOG_INFO << "  - Charts: " << numChartsSucceeded << " / " << numChartsAttempted << " straightened successfully";
    LOG_INFO << "  - Charts (diagnostics): no-work=" << chartsNoWork
             << ", with-work=" << chartsWithWork
             << ", failed-all-attempts=" << chartsFailedAllAttempts
             << ", reached-pinned-stage=" << chartsReachedPinnedStage
             << ", succeeded-regular=" << chartsSucceededRegular
             << ", succeeded-pinned=" << chartsSucceededPinned;
    if (numInversions > 0 || numWarpFailures > 0) {
        LOG_INFO << "  - Failures (attempt-level): " << numInversions << " inversions, " << numWarpFailures << " warp solver failures";
    }
    if (chartsWithWork > 0) {
        double avgAttemptsAll = (double)attemptsTotal / (double)chartsWithWork;
        double avgAttemptsSuccess = (numChartsSucceeded > 0) ? ((double)attemptsToSuccess / (double)numChartsSucceeded) : 0.0;
        LOG_INFO << "  - Attempts: avg-per-chart(with-work)=" << std::fixed << std::setprecision(2) << avgAttemptsAll
                 << ", avg-to-success(successful charts)=" << avgAttemptsSuccess
                 << ", total-attempts(with-work)=" << attemptsTotal;
    }
    for (int i = 0; i < params.maxWarpAttempts; ++i) {
        if (retryStats[i] > 0) {
            LOG_INFO << "    * " << retryStats[i] << " succeeded at attempt " << (i + 1);
        }
    }
    LOG_INFO << "  - Seam Chains: " << chains.size() << " extracted (" << numTwinPairs << " twin pairs)";
    if (totalSegmentsBefore > 0) {
        double reduction = 100.0 * (1.0 - (double)totalSegmentsAfter / totalSegmentsBefore);
        LOG_INFO << "  - Boundary Complexity: " << totalSegmentsBefore << " -> " << totalSegmentsAfter 
                 << " segments (" << reduction << "% reduction)";
    }

    LOG_INFO << "=== SIZE-ADAPTIVE PERFORMANCE (10 Tiers) ===";
    LOG_INFO << std::setw(12) << "Bin (Area3D)" << " | "
             << std::setw(8) << "Charts" << " | "
             << std::setw(8) << "Work" << " | "
             << std::setw(7) << "FailAll" << " | "
             << std::setw(7) << "PinTry" << " | "
             << std::setw(7) << "PinOK" << " | "
             << std::setw(8) << "AttAll" << " | "
             << std::setw(8) << "AttOK" << " | "
             << std::setw(10) << "Reduction" << " | "
             << std::setw(5) << "Inv" << "/"
             << std::setw(5) << "Warp" << " | "
             << "Success@Attempt (Regular / Pinned)";
    for (int i = 0; i < 10; ++i) {
        const auto& s = tiers[i];
        double low = (i == 0) ? 0 : binThresholds[i-1];
        double high = (i == 9) ? (chartAreas3D.empty() ? 0 : chartAreas3D.back()) : binThresholds[i];
        double red = s.before > 0 ? 100.0 * (1.0 - (double)s.after/s.before) : 0;
        double attAll = (s.withWork > 0) ? ((double)s.attemptsTotal / (double)s.withWork) : 0.0;
        double attOk = (s.success > 0) ? ((double)s.attemptsToSuccess / (double)s.success) : 0.0;
        
        std::stringstream ss;
        ss << std::scientific << std::setprecision(1) << low << "-" << high;
        
        std::stringstream attempts;
        for (int a = 0; a < params.maxWarpAttempts; ++a) {
            if (a > 0) attempts << ",";
            attempts << s.successPerAttempt[a];
        }
        attempts << " / ";
        for (int a = 0; a < params.maxWarpAttempts; ++a) {
            if (a > 0) attempts << ",";
            attempts << s.successPerAttempt[a + params.maxWarpAttempts];
        }

        LOG_INFO << std::setw(12) << ss.str() << " | " 
                 << std::setw(3) << s.success << "/" << std::setw(3) << s.count << " | "
                 << std::setw(3) << s.withWork << "/" << std::setw(3) << s.count << " | "
                 << std::setw(7) << s.failedAllAttempts << " | "
                 << std::setw(7) << s.reachedPinnedStage << " | "
                 << std::setw(7) << s.succeededPinned << " | "
                 << std::fixed << std::setprecision(2) << std::setw(8) << attAll << " | "
                 << std::fixed << std::setprecision(2) << std::setw(8) << attOk << " | "
                 << std::fixed << std::setprecision(1) << std::setw(8) << red << "% | "
                 << std::setw(5) << s.inversions << "/"
                 << std::setw(5) << s.warpFailures << " | "
                 << attempts.str();
    }
    LOG_INFO << "============================================";
}

} // namespace UVDefrag