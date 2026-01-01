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
#include <omp.h>
#include <cmath>
#include <array>

extern "C" {
#include "../vcglib/wrap/triangle/triangle.h"
}

namespace UVDefrag {

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
    bool isTJunction = false;
    bool isTrueBoundary = false; // Truly a mesh hole
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
    Mesh& m = graph->mesh;
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
            if (boundaryEdgesInFace >= 2) {
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
    vcg::DisjointSet<int> ds((int)sortedVertices.size());
    for (size_t i = 0; i < sortedVertices.size(); ++i) {
        for (size_t j = i + 1; j < sortedVertices.size(); ++j) {
            // Early out based on X-axis distance
            if (sortedVertices[j]->P().X() - sortedVertices[i]->P().X() > params.geometricTolerance) break;
            
            if (Distance(sortedVertices[i]->P(), sortedVertices[j]->P()) < params.geometricTolerance) {
                ds.Union((int)i, (int)j);
            }
        }
    }

    // Map each vertex to a unique ID based on its representative in the Disjoint Set
    std::map<Mesh::VertexPointer, int> vertexToUniqueId;
    std::map<int, int> dsToUniqueId;
    int nextUniqueId = 0;

    for (size_t i = 0; i < sortedVertices.size(); ++i) {
        int root = ds.Find((int)i);
        if (dsToUniqueId.find(root) == dsToUniqueId.end()) {
            dsToUniqueId[root] = nextUniqueId++;
        }
        vertexToUniqueId[sortedVertices[i]] = dsToUniqueId[root];
    }

    // Fill geometricTwins based on shared IDs
    std::map<int, std::vector<Mesh::VertexPointer>> idToVertices;
    for (auto const& [v, id] : vertexToUniqueId) idToVertices[id].push_back(v);

    for (auto const& [id, group] : idToVertices) {
        if (group.size() > 1) {
            // Seam or cut (multiple vertices at same 3D location)
            for (auto v : group) {
                for (auto other : group) {
                    if (v != other) vertexInfo[v].geometricTwins.push_back(other);
                }
            }
        } else {
            // True mesh border (hole)
            vertexInfo[group[0]].isTrueBoundary = true;
        }
    }

    // 3. Mark Immutable Vertices
    for (auto& [v, info] : vertexInfo) {
        if (info.incidentBoundaryEdges.size() != 2) info.isImmutable = true;
        if (info.isEar) info.isImmutable = true;
        
        // Pin mesh holes to preserve silhouette
        if (info.isTrueBoundary) info.isImmutable = true; 
        
        // Multi-chart Hubs (more than 1 twin implies >2 charts meeting)
        if (info.geometricTwins.size() > 1) info.isImmutable = true;
    }

    // 4. Propagate Protection
    bool changes = true;
    while (changes) {
        changes = false;
        for (auto& [v, info] : vertexInfo) {
            if (info.isImmutable) {
                for (auto twin : info.geometricTwins) {
                    auto& twinInfo = vertexInfo[twin];
                    if (!twinInfo.isImmutable) {
                        twinInfo.isImmutable = true;
                        changes = true;
                    }
                }
            }
        }
    }

    // 5. Extract Chains
    std::vector<SeamChain> chains;
    std::set<std::pair<Mesh::FacePointer, int>> visitedEdges;

    for (auto& entry : graph->charts) {
        ChartHandle chart = entry.second;
        for (auto f : chart->fpVec) {
            for (int i = 0; i < 3; ++i) {
                if (isBoundary(f, i) && visitedEdges.find({f, i}) == visitedEdges.end()) {
                    SeamChain chain;
                    chain.chart = chart;
                    
                    std::vector<std::pair<Mesh::FacePointer, int>> path;
                    path.push_back({f, i});
                    visitedEdges.insert({f, i});

                    auto curr = path.back();
                    while (true) {
                        Mesh::VertexPointer vEnd = curr.first->V((curr.second + 1) % 3);
                        if (vertexInfo[vEnd].isImmutable) break;
                        
                        std::pair<Mesh::FacePointer, int> next = {nullptr, -1};
                        for (auto& edge : vertexInfo[vEnd].incidentBoundaryEdges) {
                            if (edge.first->id == chart->id && visitedEdges.find(edge) == visitedEdges.end()) {
                                if (edge.first->V(edge.second) == vEnd) {
                                    next = edge;
                                    break;
                                }
                            }
                        }
                        if (next.first == nullptr) break;
                        path.push_back(next);
                        visitedEdges.insert(next);
                        curr = next;
                    }

                    for (auto& e : path) {
                        chain.vertices.push_back(e.first->V(e.second));
                        chain.edges.push_back(e);
                    }
                    chain.vertices.push_back(path.back().first->V((path.back().second + 1) % 3));
                    
                    chain.isSeam = true; // Mark potential seam (matched via geometric twins later)
                    chains.push_back(chain);
                }
            }
        }
    }

    LOG_INFO << "Extracted " << chains.size() << " chains.";

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

    LOG_INFO << "Phase 2 & 3: Straightening & Warping (Parallel " << omp_get_max_threads() << " threads)...";

    std::vector<int> retryStats(params.maxWarpAttempts, 0);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)chartVec.size(); ++i) {
        ChartHandle chart = chartVec[i];
        
        #pragma omp atomic
        numChartsAttempted++;

        double chartTol = params.initialTolerance;
        bool success = false;
        const std::vector<int>& myChains = chartToChains[chart];

        for (int attempt = 0; attempt < params.maxWarpAttempts; ++attempt) {
            std::map<Mesh::VertexPointer, Eigen::Vector2d> boundaryTargets;

            // RDP Calculation
            for (int ci : myChains) {
                const auto& chain = chains[ci];
                int twinIdx = chain.twinChainIdx;

                if (twinIdx != -1) {
                    // 4D Seam Straightening
                    const auto& chainB = chains[twinIdx];
                    
                    // Use matched orientation
                    bool reversed = chain.twinIsReversed;
                    
                    std::vector<Eigen::Vector4d> poly4D;
                    for (int k = 0; k < (int)chain.vertices.size(); ++k) {
                        int kB = reversed ? (chainB.vertices.size() - 1 - k) : k;
                        Point2d pA = GetSnapshotUV(globalSnapshot, chain, k);
                        Point2d pB = GetSnapshotUV(globalSnapshot, chainB, kB); 
                        poly4D.push_back(Eigen::Vector4d(pA.X(), pA.Y(), pB.X(), pB.Y()));
                    }
                    std::vector<Eigen::Vector4d> simplified = GeometryUtils::simplifyRDP(poly4D, chartTol);
                    
                    if (attempt == 0) {
                        #pragma omp atomic
                        totalSegmentsBefore += (int)poly4D.size() - 1;
                        #pragma omp atomic
                        totalSegmentsAfter += (int)simplified.size() - 1;
                    }
                    
                    auto computeArcLengths = [](const std::vector<Eigen::Vector4d>& poly) {
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
                        boundaryTargets[chain.vertices[k]] = Eigen::Vector2d(p4.x(), p4.y());
                    }
                } else {
                    // 2D Fallback: Pin to original to prevent drift of unpaired boundaries
                    for (int k = 0; k < (int)chain.vertices.size(); ++k) {
                        Point2d pA = GetSnapshotUV(globalSnapshot, chain, k);
                        boundaryTargets[chain.vertices[k]] = Eigen::Vector2d(pA.X(), pA.Y());
                    }
                    if (attempt == 0) {
                        #pragma omp atomic
                        totalSegmentsBefore += (int)chain.vertices.size() - 1;
                        #pragma omp atomic
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

            struct RawPt { Mesh::VertexPointer v; Eigen::Vector2d p; };
            std::vector<RawPt> rawPoints;
            rawPoints.reserve(chart->fpVec.size() * 3);

            for (auto f : chart->fpVec) {
                const auto& uvs = globalSnapshot.at(f);
                for(int k=0; k<3; ++k) rawPoints.push_back({f->V(k), Eigen::Vector2d(uvs[k].X(), uvs[k].Y())});
            }

            std::sort(rawPoints.begin(), rawPoints.end(), [](const RawPt& a, const RawPt& b) {
                if (a.p.x() != b.p.x()) return a.p.x() < b.p.x();
                return a.p.y() < b.p.y();
            });

            std::map<Mesh::VertexPointer, int> vToTriIdx;
            std::vector<Eigen::Vector2d> uniquePoints;
            
            if (!rawPoints.empty()) {
                uniquePoints.push_back(rawPoints[0].p);
                vToTriIdx[rawPoints[0].v] = 0;
                for (size_t k = 1; k < rawPoints.size(); ++k) {
                    const auto& prev = uniquePoints.back();
                    const auto& curr = rawPoints[k].p;
                    if ((curr - prev).squaredNorm() < 1e-20) {
                        vToTriIdx[rawPoints[k].v] = (int)uniquePoints.size() - 1;
                    } else {
                        vToTriIdx[rawPoints[k].v] = (int)uniquePoints.size();
                        uniquePoints.push_back(curr);
                    }
                }
            }

            in.numberofpoints = (int)uniquePoints.size() + 4;
            in.pointlist = (TRI_REAL *) malloc(in.numberofpoints * 2 * sizeof(TRI_REAL));
            for (int k = 0; k < (int)uniquePoints.size(); ++k) {
                in.pointlist[k * 2] = uniquePoints[k].x();
                in.pointlist[k * 2 + 1] = uniquePoints[k].y();
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

            #pragma omp critical(triangulate_lib)
            {
                triangulate((char*)"zpQ", &in, &out, nullptr);
            }

            std::vector<Eigen::Vector2d> ambientVerts(out.numberofpoints);
            for (int k = 0; k < out.numberofpoints; ++k) ambientVerts[k] = Eigen::Vector2d(out.pointlist[k * 2], out.pointlist[k * 2 + 1]);
            std::vector<Triangle> ambientFaces(out.numberoftriangles);
            for (int k = 0; k < out.numberoftriangles; ++k) {
                ambientFaces[k].v[0] = out.trianglelist[k * 3];
                ambientFaces[k].v[1] = out.trianglelist[k * 3 + 1];
                ambientFaces[k].v[2] = out.trianglelist[k * 3 + 2];
            }

            std::vector<int> fixedIndices;
            std::vector<Eigen::Vector2d> targetPositions;
            std::vector<bool> isFixed(ambientVerts.size(), false);

            fixedIndices.push_back(bb0); targetPositions.push_back(ambientVerts[bb0]); isFixed[bb0] = true;
            fixedIndices.push_back(bb1); targetPositions.push_back(ambientVerts[bb1]); isFixed[bb1] = true;
            fixedIndices.push_back(bb2); targetPositions.push_back(ambientVerts[bb2]); isFixed[bb2] = true;
            fixedIndices.push_back(bb3); targetPositions.push_back(ambientVerts[bb3]); isFixed[bb3] = true;

            for (auto const& [v, pos] : boundaryTargets) {
                int idx = vToTriIdx[v];
                if (!isFixed[idx]) {
                    fixedIndices.push_back(idx);
                    targetPositions.push_back(pos);
                    isFixed[idx] = true;
                }
            }

            if (ApplyCompositeWarp(ambientVerts, ambientFaces, fixedIndices, targetPositions)) {
                // Buffer computed UVs to verify inversion before committing
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
                    #pragma omp atomic
                    numChartsSucceeded++;
                    #pragma omp atomic
                    retryStats[attempt]++;

                    chart->isSeamStraightened = true;
                    if (params.colorize) {
                        for (auto f : chart->fpVec) f->C() = vcg::Color4b(255, 165, 0, 255);
                    }
                    // Commit to thread-safe Wedge attributes
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