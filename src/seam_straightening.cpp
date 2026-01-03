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
    bool isEar = false;
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
    // Ensure topology reflects texture connectivity (this makes seams look like borders in FF topology)
    tri::UpdateTopology<Mesh>::FaceFace(m);
    tri::UpdateTopology<Mesh>::VertexFace(m);

    LOG_INFO << "Phase 1: Boundary Classification and Geometric Twin Finding...";

    int numChartsAttempted = 0;
    int numChartsSucceeded = 0;
    int totalSegmentsBefore = 0;
    int totalSegmentsAfter = 0;

    std::map<Mesh::VertexPointer, BoundaryVertexInfo> vertexInfo;
    
    auto isBoundary = [&](Mesh::FacePointer f, int e) {
        Mesh::FacePointer ff = f->FFp(e);
        if (ff == f) return true; // Hole or Cut
        if (ff->id != f->id) return true; // Seam (if charts have different RegionIDs)
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
            if (boundaryEdgesInFace >= 2) {
                // "Ear" faces (peninsulas) must be pinned to prevent degeneracy
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

    // Use Disjoint Set for transitive welding (A~B and B~C => A~C)
    vcg::DisjointSet<MeshVertex> ds;
    for (auto v : sortedVertices) ds.MakeSet(v);

    for (size_t i = 0; i < sortedVertices.size(); ++i) {
        for (size_t j = i + 1; j < sortedVertices.size(); ++j) {
            // Early out based on X-axis distance
            if (sortedVertices[j]->P().X() - sortedVertices[i]->P().X() > params.geometricTolerance) break;
            
            if (Distance(sortedVertices[i]->P(), sortedVertices[j]->P()) < params.geometricTolerance) {
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

    // Populate geometricTwins for referencing
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
    // We count how many times each unique 3D edge appears on the boundary.
    // Count == 1: True Hole (unpaired).
    // Count >= 2: Seam (paired).
    std::map<std::pair<int, int>, int> geometricEdgeCounts;

    for (auto& pair : vertexInfo) {
        Mesh::VertexPointer v = pair.first;
        int idStart = vertexToUniqueId[v];

        // incidentBoundaryEdges contains edges incident to v. 
        // We need to identify the other endpoint to form the edge.
        for (auto& be : pair.second.incidentBoundaryEdges) {
            // be is (Face, index). Edge is from V(index) -> V(index+1).
            Mesh::VertexPointer v0 = be.first->V(be.second);
            Mesh::VertexPointer v1 = be.first->V((be.second + 1) % 3);
            
            // We only care about edges leaving v (to avoid double counting when iterating all verts)
            if (v0 == v) {
                int idEnd = vertexToUniqueId[v1];
                if (idStart == idEnd) continue; // Degenerate edge, ignore

                // Undirected edge key
                std::pair<int, int> key = std::minmax(idStart, idEnd);
                geometricEdgeCounts[key]++;
            }
        }
    }

    // 4. Determine Immutability
    // A vertex is mutable if and only if:
    // 1. It is not part of a hole (no incident unpaired edges).
    // 2. It has exactly 2 geometric neighbors (Valence 2, i.e., it's a line, not a junction).
    // 3. It is not an Ear.
    
    // We compute this per Geometric ID.
    std::vector<bool> idIsImmutable(nextUniqueId, false);
    
    // Iterate via idToVertices to check adjacency efficiently
    for (auto const& groupPair : idToVertices) {
        int id = groupPair.first;
        
        // Check "Ear" status (if any twin is an Ear, the geometric vertex is locked)
        bool groupIsEar = false;
        for (auto v : groupPair.second) {
            if (vertexInfo[v].isEar) { groupIsEar = true; break; }
        }
        if (groupIsEar) {
            idIsImmutable[id] = true;
            continue;
        }

        std::set<int> neighbors;
        bool hasHoleEdge = false;
        bool hasNonManifoldEdge = false; // Edges shared by >2 surfaces (rare but possible)

        // Gather neighbors from all twins
        for (auto v : groupPair.second) {
             for (auto& be : vertexInfo[v].incidentBoundaryEdges) {
                Mesh::VertexPointer v0 = be.first->V(be.second);
                Mesh::VertexPointer v1 = be.first->V((be.second + 1) % 3);
                
                // Identify neighbor
                Mesh::VertexPointer neighborV = (v0 == v) ? v1 : v0;
                int neighborId = vertexToUniqueId[neighborV];
                if (id == neighborId) continue;

                neighbors.insert(neighborId);
                
                // Check edge property
                std::pair<int, int> key = std::minmax(id, neighborId);
                int count = geometricEdgeCounts[key];
                
                if (count == 1) hasHoleEdge = true;
                if (count > 2) hasNonManifoldEdge = true;
             }
        }

        // Immutability Logic
        if (hasHoleEdge) {
            idIsImmutable[id] = true; // Pin holes to preserve silhouette
        } else if (hasNonManifoldEdge) {
            idIsImmutable[id] = true; // Pin complex geometry
        } else if (neighbors.size() != 2) {
            idIsImmutable[id] = true; // Pin endpoints and junctions (Valence != 2)
        } else {
            idIsImmutable[id] = false; // Simple Seam Interior -> Mutable
        }
    }

    // Apply to individual vertices
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
                    
                    // Found a seed edge. 
                    // To extract maximal chains, we must find the *beginning* of this chain segment.
                    // Backtrack while the start vertex is Mutable and we haven't looped.
                    
                    std::pair<Mesh::FacePointer, int> currEdge = {f, i};
                    std::vector<std::pair<Mesh::FacePointer, int>> backtrackingPath; 

                    while (true) {
                        Mesh::VertexPointer vStart = currEdge.first->V(currEdge.second);
                        if (vertexInfo[vStart].isImmutable) break; // Start is pinned, so this is the start.

                        // Find the previous edge in the chain (incident to vStart, ending at vStart)
                        std::pair<Mesh::FacePointer, int> prevEdge = {nullptr, -1};
                        for (auto& inc : vertexInfo[vStart].incidentBoundaryEdges) {
                            if (inc.first->id != chart->id) continue;
                            
                            // Check if this edge *ends* at vStart
                            Mesh::VertexPointer eEnd = inc.first->V((inc.second + 1) % 3);
                            if (eEnd == vStart) {
                                // Must be unvisited to backtrack safely (or the one we just came from if looping)
                                if (inc == currEdge) continue; 
                                prevEdge = inc;
                                break;
                            }
                        }

                        if (prevEdge.first == nullptr) break; 
                        
                        // Loop detection
                        if (visitedEdges.find(prevEdge) != visitedEdges.end()) {
                             bool inCurrentLoop = false;
                             if(prevEdge == std::make_pair(f,i)) inCurrentLoop = true; 
                             if(inCurrentLoop) break; 
                             break; 
                        }
                        
                        currEdge = prevEdge;
                        if (currEdge == std::make_pair(f,i)) break; // Full loop
                    }

                    // Now trace forward from currEdge
                    SeamChain chain;
                    chain.chart = chart;
                    std::vector<std::pair<Mesh::FacePointer, int>> path;
                    
                    if (visitedEdges.find(currEdge) != visitedEdges.end()) continue;

                    std::pair<Mesh::FacePointer, int> trace = currEdge;
                    path.push_back(trace);
                    visitedEdges.insert(trace);

                    while (true) {
                        Mesh::VertexPointer vEnd = trace.first->V((trace.second + 1) % 3);
                        
                        // Stop if we hit a pinned vertex
                        if (vertexInfo[vEnd].isImmutable) break;
                        
                        // Stop if we loop back to start of this specific path
                        if (vEnd == path.front().first->V(path.front().second)) break; 

                        // Find next
                        std::pair<Mesh::FacePointer, int> next = {nullptr, -1};
                        for (auto& inc : vertexInfo[vEnd].incidentBoundaryEdges) {
                            if (inc.first->id != chart->id) continue;
                            if (visitedEdges.find(inc) != visitedEdges.end()) continue;
                            
                            // Must start at vEnd
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
                    // Add final vertex
                    chain.vertices.push_back(path.back().first->V((path.back().second + 1) % 3));
                    
                    chain.isSeam = true; 
                    chains.push_back(chain);
                }
            }
        }
    }

    LOG_INFO << "Extracted " << chains.size() << " chains.";

    // 6. Chain Matching (Twin Identification)
    // Key: Sequence of Geometric IDs
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
                std::reverse(key.pointIds.begin(), key.pointIds.end()); // restore
                idToChain[key] = i;
            }
        }
    }
    LOG_INFO << "Matched " << numTwinPairs << " twin pairs of seam chains.";

    // 7. Processing Loop (Parallel)
    std::map<ChartHandle, std::vector<int>> chartToChains;
    for(int i = 0; i < (int)chains.size(); ++i) chartToChains[chains[i].chart].push_back(i);

    std::vector<ChartHandle> chartVec;
    chartVec.reserve(graph->charts.size());
    for(auto& entry : graph->charts) chartVec.push_back(entry.second);

    // Create Immutable Snapshot
    GlobalUVSnapshot globalSnapshot;
    for (auto& entry : graph->charts) {
        for (auto f : entry.second->fpVec) {
            globalSnapshot[f] = { f->WT(0).P(), f->WT(1).P(), f->WT(2).P() };
        }
    }

    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#endif
    LOG_INFO << "Phase 2 & 3: Straightening & Warping (Parallel " << num_threads << " threads)...";

    std::vector<int> retryStats(params.maxWarpAttempts, 0);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < (int)chartVec.size(); ++i) {
        ChartHandle chart = chartVec[i];
        
#ifdef _OPENMP
        #pragma omp atomic
#endif
        numChartsAttempted++;

        double chartTol = params.initialTolerance;
        bool success = false;
        
        auto chainIt = chartToChains.find(chart);
        if (chainIt == chartToChains.end()) continue;
        const std::vector<int>& myChains = chainIt->second;

        for (int attempt = 0; attempt < params.maxWarpAttempts; ++attempt) {
            std::map<Mesh::VertexPointer, Point2D> boundaryTargets;

            // RDP Calculation
            for (int ci : myChains) {
                const auto& chain = chains[ci];
                int twinIdx = chain.twinChainIdx;

                if (twinIdx != -1) {
                    // 4D Seam Straightening
                    const auto& chainB = chains[twinIdx];
                    
                    // Use matched orientation
                    bool reversed = chain.twinIsReversed;
                    
                    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> poly4D;
                    for (int k = 0; k < (int)chain.vertices.size(); ++k) {
                        int kB = reversed ? (chainB.vertices.size() - 1 - k) : k;
                        Point2d pA = GetSnapshotUV(globalSnapshot, chain, k);
                        Point2d pB = GetSnapshotUV(globalSnapshot, chainB, kB); 
                        poly4D.push_back(Eigen::Vector4d(pA.X(), pA.Y(), pB.X(), pB.Y()));
                    }
                    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> simplified = GeometryUtils::simplifyRDP(poly4D, chartTol);
                    
                    if (attempt == 0) {
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        totalSegmentsBefore += (int)poly4D.size() - 1;
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        totalSegmentsAfter += (int)simplified.size() - 1;
                    }
                    
                    auto computeArcLengths = [](const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& poly) {
                        std::vector<double> lengths(poly.size(), 0.0);
                        for (size_t k = 1; k < poly.size(); ++k) lengths[k] = lengths[k - 1] + (poly[k] - poly[k - 1]).norm();
                        return lengths;
                    };
                    std::vector<double> origLengths = computeArcLengths(poly4D);
                    std::vector<double> simplifiedLengths = computeArcLengths(simplified);
                    double totalLen = origLengths.back();
                    double simplifiedTotalLen = simplifiedLengths.back();

                    for (int k = 0; k < (int)chain.vertices.size(); ++k) {
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
                        boundaryTargets[chain.vertices[k]] = { p4.x(), p4.y() };
                    }
                } else {
                    // 2D Fallback: Pin to original
                    for (int k = 0; k < (int)chain.vertices.size(); ++k) {
                        Point2d pA = GetSnapshotUV(globalSnapshot, chain, k);
                        boundaryTargets[chain.vertices[k]] = { pA.X(), pA.Y() };
                    }
                    if (attempt == 0) {
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        totalSegmentsBefore += (int)chain.vertices.size() - 1;
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        totalSegmentsAfter += (int)chain.vertices.size() - 1;
                    }
                }
            }

            // Ambient Mesh Setup
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
                for (size_t k = 1; k < rawPoints.size(); ++k) {
                    const auto& prev = uniquePoints.back();
                    double dx = rawPoints[k].x - prev.x;
                    double dy = rawPoints[k].y - prev.y;
                    if (dx*dx + dy*dy < 1e-20) {
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
                    if (area <= 1e-12) { inverted = true; break; }
                }
                
                if (!inverted) {
                    success = true;
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    numChartsSucceeded++;
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    retryStats[attempt]++;

                    chart->isSeamStraightened = true;
                    if (params.colorize) {
                        for (auto f : chart->fpVec) f->C() = vcg::Color4b(255, 165, 0, 255);
                    }
                    for (auto f : chart->fpVec) {
                        for (int k = 0; k < 3; ++k) f->WT(k).P() = computedUVs[f->V(k)];
                    }
                }
            }

            free(in.pointlist); free(in.segmentlist);
            trifree(out.pointlist); trifree(out.trianglelist); trifree(out.segmentlist);
            
            if (success) break;
            chartTol *= 0.5;
        }
    }

    LOG_INFO << "Seam Straightening completed:";
    LOG_INFO << "  - Charts: " << numChartsSucceeded << " / " << numChartsAttempted << " straightened successfully";
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
}

} // namespace UVDefrag