#include "seam_straightening.h"
#include "mesh.h"
#include "logging.h"
#include "rdp.h"
#include "harmonic_map_utils.h"
#include <vcg/complex/algorithms/update/topology.h>
#include <vcg/complex/algorithms/clean.h>
#include <vcg/complex/algorithms/outline_support.h>
#include <map>
#include <set>
#include <unordered_map>
#include <queue>
#include <algorithm>

extern "C" {
#include "../vcglib/wrap/triangle/triangle.h"
}

namespace UVDefrag {

struct SeamChain {
    std::vector<Mesh::VertexPointer> vertices;
    std::vector<std::pair<Mesh::FacePointer, int>> edges; // Face and edge index
    ChartHandle chart;
    bool isSeam = false;
    bool isTrueBoundary = false;
    int twinChainIdx = -1;
};

struct BoundaryVertexInfo {
    bool isImmutable = false;
    bool isTJunction = false;
    bool isTrueBoundary = false;
    bool isEar = false;
    std::vector<std::pair<Mesh::FacePointer, int>> incidentBoundaryEdges;
};

static Eigen::Vector3d ComputeBarycentric(const Eigen::Vector2d& p, const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Eigen::Vector2d& c) {
    Eigen::Vector2d v0 = b - a, v1 = c - a, v2 = p - a;
    double den = v0.x() * v1.y() - v1.x() * v0.y();
    if (std::abs(den) < 1e-12) return Eigen::Vector3d(1, 0, 0);
    double v = (v2.x() * v1.y() - v1.x() * v2.y()) / den;
    double w = (v0.x() * v2.y() - v2.x() * v0.y()) / den;
    double u = 1.0 - v - w;
    return Eigen::Vector3d(u, v, w);
}

void IntegrateSeamStraightening(GraphHandle graph, const SeamStraighteningParameters& params) {
    Mesh& m = graph->mesh;
    tri::UpdateTopology<Mesh>::FaceFace(m);
    tri::UpdateTopology<Mesh>::VertexFace(m);

    LOG_INFO << "Phase 1: Boundary Classification and Pairing...";

    int numChartsAttempted = 0;
    int numChartsSucceeded = 0;
    int totalSegmentsBefore = 0;
    int totalSegmentsAfter = 0;
    int numTwinPairs = 0;

    std::map<Mesh::VertexPointer, BoundaryVertexInfo> vertexInfo;
    
    // Helper to identify if an edge is a boundary (mesh hole or seam)
    auto isBoundary = [&](Mesh::FacePointer f, int e) {
        Mesh::FacePointer ff = f->FFp(e);
        if (ff == f) return true; // Mesh hole
        if (ff->id != f->id) return true; // Seam
        return false;
    };

    // 1. Collect boundary edges and count incident boundary edges per vertex
    for (auto& entry : graph->charts) {
        ChartHandle chart = entry.second;
        for (auto f : chart->fpVec) {
            int boundaryEdgesInFace = 0;
            for (int i = 0; i < 3; ++i) {
                if (isBoundary(f, i)) {
                    boundaryEdgesInFace++;
                    Mesh::VertexPointer v0 = f->V(i);
                    Mesh::VertexPointer v1 = f->V((i + 1) % 3);
                    vertexInfo[v0].incidentBoundaryEdges.push_back({f, i});
                    vertexInfo[v1].incidentBoundaryEdges.push_back({f, i});
                    
                    if (f->FFp(i) == f) {
                        vertexInfo[v0].isTrueBoundary = true;
                        vertexInfo[v1].isTrueBoundary = true;
                    }
                }
            }
            if (boundaryEdgesInFace >= 2) {
                for (int i = 0; i < 3; ++i) {
                    vertexInfo[f->V(i)].isEar = true;
                }
            }
        }
    }

    // 2. Mark Immutable vertices
    for (auto& [v, info] : vertexInfo) {
        if (info.incidentBoundaryEdges.size() > 2) {
            info.isTJunction = true;
            info.isImmutable = true;
        }
        if (info.isTrueBoundary) info.isImmutable = true;
        if (info.isEar) info.isImmutable = true;
    }

    // 3. Extract Seam Chains
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

                    auto nextEdge = [&](std::pair<Mesh::FacePointer, int> curr) -> std::pair<Mesh::FacePointer, int> {
                        Mesh::VertexPointer vEnd = curr.first->V((curr.second + 1) % 3);
                        if (vertexInfo[vEnd].isImmutable) return {nullptr, -1};
                        
                        for (auto& edge : vertexInfo[vEnd].incidentBoundaryEdges) {
                            if (edge.first->id == chart->id && visitedEdges.find(edge) == visitedEdges.end()) {
                                if (edge.first->V(edge.second) == vEnd) return edge;
                            }
                        }
                        return {nullptr, -1};
                    };

                    auto prevEdge = [&](std::pair<Mesh::FacePointer, int> curr) -> std::pair<Mesh::FacePointer, int> {
                        Mesh::VertexPointer vStart = curr.first->V(curr.second);
                        if (vertexInfo[vStart].isImmutable) return {nullptr, -1};

                        for (auto& edge : vertexInfo[vStart].incidentBoundaryEdges) {
                            if (edge.first->id == chart->id && visitedEdges.find(edge) == visitedEdges.end()) {
                                if (edge.first->V((edge.second + 1) % 3) == vStart) return edge;
                            }
                        }
                        return {nullptr, -1};
                    };

                    auto curr = path.back();
                    while (true) {
                        auto next = nextEdge(curr);
                        if (next.first == nullptr) break;
                        path.push_back(next);
                        visitedEdges.insert(next);
                        curr = next;
                    }

                    curr = path.front();
                    while (true) {
                        auto prev = prevEdge(curr);
                        if (prev.first == nullptr) break;
                        path.insert(path.begin(), prev);
                        visitedEdges.insert(prev);
                        curr = prev;
                    }

                    for (auto& e : path) {
                        chain.vertices.push_back(e.first->V(e.second));
                        chain.edges.push_back(e);
                    }
                    chain.vertices.push_back(path.back().first->V((path.back().second + 1) % 3));
                    
                    chain.isTrueBoundary = false;
                    for (auto& e : path) if (e.first->FFp(e.second) == e.first) chain.isTrueBoundary = true;
                    chain.isSeam = !chain.isTrueBoundary;

                    chains.push_back(chain);
                }
            }
        }
    }

    LOG_INFO << "Extracted " << chains.size() << " boundary chains.";

    // Link twins (Sides A and B of a cut)
    // Note: This relies on shared vertices or exact float equality for duplicated vertices.
    std::map<std::vector<vcg::Point3d>, int> coordToChain;
    for (int i = 0; i < (int)chains.size(); ++i) {
        if (!chains[i].isSeam) continue;
        std::vector<vcg::Point3d> coords;
        for (auto v : chains[i].vertices) coords.push_back(v->P());
        
        if (coordToChain.count(coords)) {
            int j = coordToChain[coords];
            chains[i].twinChainIdx = j;
            chains[j].twinChainIdx = i;
            numTwinPairs++;
        } else {
            std::reverse(coords.begin(), coords.end());
            if (coordToChain.count(coords)) {
                int j = coordToChain[coords];
                chains[i].twinChainIdx = j;
                chains[j].twinChainIdx = i;
                numTwinPairs++;
            } else {
                std::reverse(coords.begin(), coords.end());
                coordToChain[coords] = i;
            }
        }
    }

    LOG_INFO << "Phase 2, 3 & 4: Processing each chart...";
    std::vector<int> retryStats(params.maxWarpAttempts, 0);

    for (auto& entry : graph->charts) {
        ChartHandle chart = entry.second;
        numChartsAttempted++;
        
        double chartTol = params.initialTolerance;
        bool success = false;

        // Backup original positions for this chart
        std::map<Mesh::VertexPointer, Point2d> originalPositions;
        for (auto f : chart->fpVec) for (int k = 0; k < 3; ++k) originalPositions[f->V(k)] = f->V(k)->T().P();

        for (int attempt = 0; attempt < params.maxWarpAttempts; ++attempt) {
            std::map<Mesh::VertexPointer, Eigen::Vector2d> boundaryTargets;
            std::map<Mesh::VertexPointer, Eigen::Vector2d> boundaryStarts;
            std::vector<std::pair<Mesh::VertexPointer, Mesh::VertexPointer>> constraints;

            for (int ci = 0; ci < (int)chains.size(); ++ci) {
                if (chains[ci].chart != chart) continue;
                auto& chain = chains[ci];
                int twinIdx = chain.twinChainIdx;

                if (twinIdx != -1 && chains[twinIdx].vertices.size() == chain.vertices.size()) {
                    auto& chainB = chains[twinIdx];
                    bool reversed = (chain.vertices.front()->P() != chainB.vertices.front()->P());
                    std::vector<Eigen::Vector4d> poly4D;
                    for (int k = 0; k < (int)chain.vertices.size(); ++k) {
                        int kB = reversed ? (chainB.vertices.size() - 1 - k) : k;
                        poly4D.push_back(Eigen::Vector4d(
                            originalPositions[chain.vertices[k]].X(), originalPositions[chain.vertices[k]].Y(),
                            originalPositions[chainB.vertices[kB]].X(), originalPositions[chainB.vertices[kB]].Y()
                        ));
                    }
                    std::vector<Eigen::Vector4d> simplified = GeometryUtils::simplifyRDP(poly4D, chartTol);
                    
                    if (attempt == 0) {
                        totalSegmentsBefore += (int)poly4D.size() - 1;
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
                        boundaryStarts[chain.vertices[k]] = Eigen::Vector2d(originalPositions[chain.vertices[k]].X(), originalPositions[chain.vertices[k]].Y());
                    }
                } else {
                    for (auto v : chain.vertices) {
                        boundaryTargets[v] = Eigen::Vector2d(originalPositions[v].X(), originalPositions[v].Y());
                        boundaryStarts[v] = Eigen::Vector2d(originalPositions[v].X(), originalPositions[v].Y());
                    }
                }
                for (size_t k = 0; k < chain.vertices.size() - 1; ++k) {
                    constraints.push_back({chain.vertices[k], chain.vertices[k+1]});
                }
            }

            vcg::Box2d bbox = chart->UVBox();
            bbox.Offset(vcg::Point2d(bbox.DimX() * 0.1, bbox.DimY() * 0.1));

            struct triangulateio in, out;
            memset(&in, 0, sizeof(in));
            memset(&out, 0, sizeof(out));

            in.numberofpoints = (int)boundaryStarts.size() + 4;
            in.pointlist = (TRI_REAL *) malloc(in.numberofpoints * 2 * sizeof(TRI_REAL));
            
            std::map<Mesh::VertexPointer, int> vToTriIdx;
            int nextIdx = 0;
            for (auto& [v, pos] : boundaryStarts) {
                in.pointlist[nextIdx * 2] = pos.x();
                in.pointlist[nextIdx * 2 + 1] = pos.y();
                vToTriIdx[v] = nextIdx++;
            }
            in.pointlist[nextIdx * 2] = bbox.min.X(); in.pointlist[nextIdx * 2 + 1] = bbox.min.Y(); int bb0 = nextIdx++;
            in.pointlist[nextIdx * 2] = bbox.max.X(); in.pointlist[nextIdx * 2 + 1] = bbox.min.Y(); int bb1 = nextIdx++;
            in.pointlist[nextIdx * 2] = bbox.max.X(); in.pointlist[nextIdx * 2 + 1] = bbox.max.Y(); int bb2 = nextIdx++;
            in.pointlist[nextIdx * 2] = bbox.min.X(); in.pointlist[nextIdx * 2 + 1] = bbox.max.Y(); int bb3 = nextIdx++;

            in.numberofsegments = (int)constraints.size() + 4;
            in.segmentlist = (int *) malloc(in.numberofsegments * 2 * sizeof(int));
            for (int k = 0; k < (int)constraints.size(); ++k) {
                in.segmentlist[k * 2] = vToTriIdx[constraints[k].first];
                in.segmentlist[k * 2 + 1] = vToTriIdx[constraints[k].second];
            }
            int segOff = (int)constraints.size();
            in.segmentlist[segOff*2] = bb0; in.segmentlist[segOff*2+1] = bb1;
            in.segmentlist[(segOff+1)*2] = bb1; in.segmentlist[(segOff+1)*2+1] = bb2;
            in.segmentlist[(segOff+2)*2] = bb2; in.segmentlist[(segOff+2)*2+1] = bb3;
            in.segmentlist[(segOff+3)*2] = bb3; in.segmentlist[(segOff+3)*2+1] = bb0;

            triangulate((char*)"zpaQ", &in, &out, nullptr);

            std::vector<Eigen::Vector2d> ambientVerts(out.numberofpoints);
            for (int k = 0; k < out.numberofpoints; ++k) ambientVerts[k] = Eigen::Vector2d(out.pointlist[k * 2], out.pointlist[k * 2 + 1]);
            std::vector<Triangle> ambientFaces(out.numberoftriangles);
            for (int k = 0; k < out.numberoftriangles; ++k) {
                ambientFaces[k].v[0] = out.trianglelist[k * 3];
                ambientFaces[k].v[1] = out.trianglelist[k * 3 + 1];
                ambientFaces[k].v[2] = out.trianglelist[k * 3 + 2];
            }

            struct Binding { int triIdx; Eigen::Vector3d bary; };
            std::map<Mesh::VertexPointer, Binding> bindings;
            std::set<Mesh::VertexPointer> allChartVerts;
            for (auto f : chart->fpVec) for (int k = 0; k < 3; ++k) allChartVerts.insert(f->V(k));

            for (auto v : allChartVerts) {
                Eigen::Vector2d p(originalPositions[v].X(), originalPositions[v].Y());
                bool found = false;
                for (int k = 0; k < (int)ambientFaces.size(); ++k) {
                    auto& tri = ambientFaces[k];
                    Eigen::Vector3d bary = ComputeBarycentric(p, ambientVerts[tri.v[0]], ambientVerts[tri.v[1]], ambientVerts[tri.v[2]]);
                    if (bary.x() >= -1e-6 && bary.y() >= -1e-6 && bary.z() >= -1e-6) {
                        bindings[v] = {k, bary};
                        found = true;
                        break;
                    }
                }
                if (!found) { /* Use closest or something? Margin should be enough. */ }
            }

            std::vector<int> fixedAmbientIndices;
            std::vector<Eigen::Vector2d> ambientTargets;
            for (auto& [v, idx] : vToTriIdx) {
                fixedAmbientIndices.push_back(idx);
                ambientTargets.push_back(boundaryTargets[v]);
            }
            fixedAmbientIndices.push_back(bb0); ambientTargets.push_back(ambientVerts[bb0]);
            fixedAmbientIndices.push_back(bb1); ambientTargets.push_back(ambientVerts[bb1]);
            fixedAmbientIndices.push_back(bb2); ambientTargets.push_back(ambientVerts[bb2]);
            fixedAmbientIndices.push_back(bb3); ambientTargets.push_back(ambientVerts[bb3]);

            if (ApplyCompositeWarp(ambientVerts, ambientFaces, fixedAmbientIndices, ambientTargets)) {
                for (auto v : allChartVerts) {
                    auto it = bindings.find(v);
                    if (it == bindings.end()) continue;
                    auto& b = it->second;
                    auto& tri = ambientFaces[b.triIdx];
                    Eigen::Vector2d p = b.bary.x() * ambientVerts[tri.v[0]] + b.bary.y() * ambientVerts[tri.v[1]] + b.bary.z() * ambientVerts[tri.v[2]];
                    v->T().P() = Point2d(p.x(), p.y());
                }
                bool inverted = false;
                for (auto f : chart->fpVec) if (AreaUV(*f) <= 1e-12) { inverted = true; break; }
                if (!inverted) {
                    success = true;
                    numChartsSucceeded++;
                    retryStats[attempt]++;
                    chart->isSeamStraightened = true;
                    if (params.colorize) {
                        for (auto f : chart->fpVec) f->C() = vcg::Color4b(255, 165, 0, 255);
                    }
                    for (auto f : chart->fpVec) for (int k = 0; k < 3; ++k) f->WT(k).P() = f->V(k)->T().P();
                } else {
                    for (auto v : allChartVerts) v->T().P() = originalPositions[v];
                }
            }

            free(in.pointlist); free(in.segmentlist);
            trifree((int*)out.pointlist); trifree(out.trianglelist); trifree(out.segmentlist);
            
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