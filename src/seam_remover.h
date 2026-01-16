/*******************************************************************************
    Copyright (c) 2021, Andrea Maggiordomo, Paolo Cignoni and Marco Tarini

    This file is part of TextureDefrag, a reference implementation for
    the paper ``Texture Defragmentation for Photo-Reconstructed 3D Models''
    by Andrea Maggiordomo, Paolo Cignoni and Marco Tarini.

    TextureDefrag is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    TextureDefrag is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with TextureDefrag. If not, see <https://www.gnu.org/licenses/>.
*******************************************************************************/

#ifndef SEAM_REMOVER_H
#define SEAM_REMOVER_H

#include <vector>
#include <memory>
#include <queue>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include <vcg/space/point3.h>
#include <vcg/space/point2.h>

#include "types.h"
#include "mesh.h"
#include "mesh_graph.h"
#include "matching.h"
#include "arap.h"

#include "seams.h"
#include "intersection.h"

typedef std::unordered_map<Mesh::VertexPointer, double> OffsetMap;

struct AlgoParameters {
    double matchingThreshold         = 2.0;
    // If > 0, uses an absolute tolerance in texels (UVs are in pixel space after `ScaleTextureCoordinatesToImage`)
    // for both seam matching feasibility and post-optimization distortion checks.
    double maxErrorTexels            = 0.0;
    double offsetFactor              = 5.0;
    double boundaryTolerance         = 0.2;
    double distortionTolerance       = 0.5;
    double globalDistortionThreshold = 0.025;
    double reductionFactor           = 0.8;
    bool   reduce                    = false;
    double timelimit                 = 0;
    bool   visitComponents           = true;
    double expb                      = 1.0;
    double UVBorderLengthReduction   = 0.0;
    bool   ignoreOnReject            = false;
    double resolutionScaling         = 1.0;
    int    rotationNum               = 4;
    // If true, rank/attempt operations primarily by seam-edge removal (within tolerances).
    bool   minimizeSeamEdges         = false;
};

struct SeamData {
    ClusteredSeamHandle csh;

    ChartHandle a;
    ChartHandle b;

    std::vector<vcg::Point2d> texcoorda;
    std::vector<vcg::Point2d> texcoordb;
    std::vector<int> vertexinda;
    std::vector<int> vertexindb;

    std::map<Mesh::VertexPointer, Mesh::VertexPointer> mrep;
    std::map<SeamMesh::VertexPointer, std::vector<Mesh::VertexPointer>> evec;

    typedef std::pair<std::vector<Mesh::FacePointer>, std::vector<int>> FanInfo;
    std::map<Mesh::VertexPointer, FanInfo> vfmap;

    std::unordered_set<Mesh::VertexPointer> verticesWithinThreshold;
    std::unordered_set<Mesh::FacePointer> optimizationArea;
    std::vector<vcg::Point2d> texcoordoptVert;
    std::vector<vcg::Point2d> texcoordoptWedge;

    double inputNegativeArea;
    double inputAbsoluteArea;

    double inputUVBorderLength;

    double inputArapNum;
    double inputArapDenom;

    double outputArapNum;
    double outputArapDenom;

    ARAPSolveInfo si;

    Mesh shell;

    std::vector<HalfEdgePair> intersectionOpt;
    std::vector<HalfEdgePair> intersectionBoundary;
    std::vector<HalfEdgePair> intersectionInternal;

    std::unordered_set<Mesh::VertexPointer> fixedVerticesFromIntersectingEdges;

    SeamData() : a{nullptr}, b{nullptr}, inputNegativeArea{0}, inputAbsoluteArea{0} {}
};

// ============================================================================
// Parallel Architecture Data Structures
// ============================================================================

enum CheckStatus {
    PASS=0,
    FAIL_LOCAL_OVERLAP,
    FAIL_GLOBAL_OVERLAP_BEFORE,
    FAIL_GLOBAL_OVERLAP_AFTER_OPT, // border of the optimization area self-intersects
    FAIL_GLOBAL_OVERLAP_AFTER_BND, // border of the optimzation area hit the fixed border
    FAIL_DISTORTION_LOCAL,
    FAIL_DISTORTION_GLOBAL,
    FAIL_TOPOLOGY,  // shell genus is > 0 or shell is closed
    FAIL_NUMERICAL_ERROR,
    UNKNOWN,
    FAIL_GLOBAL_OVERLAP_UNFIXABLE,
    _END
};

// TopologyDiff: Describes the merge result without modifying global mesh state.
// Worker threads produce this; main thread applies it during commit phase.
struct TopologyDiff {
    // 1. Alignment Transform (Result of ICP/Rigid Match)
    MatchingTransform transform;

    // 2. Vertex Merges (The "Topology" change)
    // Map: Old Vertex Pointer -> New Representative Vertex Pointer
    // Used to simulate the merged mesh locally without modifying global state.
    std::unordered_map<Mesh::VertexPointer, Mesh::VertexPointer> replacements;

    // 2b. Merge Groups (Reverse of replacements)
    // Map: Representative Vertex -> All vertices that merge into it
    // Used for BFS traversal across the virtual seam merge
    std::unordered_map<Mesh::VertexPointer, std::vector<Mesh::VertexPointer>> mergeGroups;

    // 3. New UV Positions (The "Geometry" change)
    // Map: Vertex Pointer -> New UV Position
    // Stores the result of Rigid Alignment + ARAP Optimization
    std::unordered_map<Mesh::VertexPointer, vcg::Point2d> newUVPositions;

    // 4. New Wedge UV Positions (per face-vertex)
    // Map: (FacePointer, vertex_index) -> New UV Position
    std::map<std::pair<Mesh::FacePointer, int>, vcg::Point2d> newWedgeUVPositions;

    // 5. The Offset Map (for optimization area propagation)
    OffsetMap offsetMap;

    // 6. Seam edge vertex mapping (needed for VF adjacency updates)
    std::map<SeamMesh::VertexPointer, std::vector<Mesh::VertexPointer>> evec;

    // 7. VF adjacency info for rollback/commit
    typedef std::pair<std::vector<Mesh::FacePointer>, std::vector<int>> FanInfo;
    std::map<Mesh::VertexPointer, FanInfo> vfmap;

    // 8. UV bounding box of the result (for spatial collision detection)
    vcg::Box2d resultBoundingBox;

    // Clear all data
    void Clear() {
        transform = MatchingTransform::Identity();
        replacements.clear();
        mergeGroups.clear();
        newUVPositions.clear();
        newWedgeUVPositions.clear();
        offsetMap.clear();
        evec.clear();
        vfmap.clear();
        resultBoundingBox.SetNull();
    }
};

// MergeJobResult: Carries data from Worker Thread back to Main Thread
struct MergeJobResult {
    // ID back to the operation
    ClusteredSeamHandle csh;

    // Outcome
    CheckStatus checkStatus;

    // Chart IDs involved (for locking/collision detection)
    RegionID chartIdA;
    RegionID chartIdB;

    // Resulting Data (Valid only if checkStatus == PASS)
    TopologyDiff diff;

    // Metrics for scoring updates
    double finalEnergy;
    double initialEnergy;
    int arapIterations;

    // ARAP energy contribution changes
    double inputArapNum;
    double inputArapDenom;
    double outputArapNum;
    double outputArapDenom;

    // Negative/Absolute area changes
    double inputNegativeArea;
    double inputAbsoluteArea;

    // UV border length before merge
    double inputUVBorderLength;

    // Optimization area faces (needed for commit phase)
    std::unordered_set<Mesh::FacePointer> optimizationArea;
    std::unordered_set<Mesh::VertexPointer> verticesWithinThreshold;

    // Stored coordinates for rollback (not needed in parallel - we don't modify mesh)
    // But we need these for the commit phase to properly update the mesh
    std::vector<vcg::Point2d> texcoorda;
    std::vector<vcg::Point2d> texcoordb;
    std::vector<int> vertexinda;
    std::vector<int> vertexindb;

    // Intersection data (for potential retry logic)
    std::vector<HalfEdgePair> intersectionOpt;
    std::vector<HalfEdgePair> intersectionBoundary;
    std::vector<HalfEdgePair> intersectionInternal;
    std::unordered_set<Mesh::VertexPointer> fixedVerticesFromIntersectingEdges;

    MergeJobResult() : csh{nullptr}, checkStatus{UNKNOWN}, chartIdA{INVALID_ID}, chartIdB{INVALID_ID},
                       finalEnergy{0}, initialEnergy{0}, arapIterations{0},
                       inputArapNum{0}, inputArapDenom{0}, outputArapNum{0}, outputArapDenom{0},
                       inputNegativeArea{0}, inputAbsoluteArea{0}, inputUVBorderLength{0} {}
};

// ParallelBatchStats: Statistics for parallel batch processing
struct ParallelBatchStats {
    int batchNumber;
    int batchSize;
    int accepted;
    int rejected;
    int collisions; // "Moving Walls" collisions detected during commit
    double selectionTimeMs;
    double evaluationTimeMs;
    double commitTimeMs;

    ParallelBatchStats() : batchNumber{0}, batchSize{0}, accepted{0}, rejected{0},
                           collisions{0}, selectionTimeMs{0}, evaluationTimeMs{0}, commitTimeMs{0} {}
};

// ============================================================================
// End Parallel Architecture Data Structures
// ============================================================================

struct CostInfo {
    enum MatchingValue {
        FEASIBLE=0,
        ZERO_AREA,
        UNFEASIBLE_BOUNDARY,
        UNFEASIBLE_MATCHING,
        REJECTED,
        _END
    };

    double cost;
    MatchingTransform matching;
    MatchingValue mvalue;
};

struct AlgoState {

    struct WeightedSeamCmp {
        bool operator()(const WeightedSeam& a, const WeightedSeam& b)
        {
            return a.second > b.second;
        }
    };

    std::priority_queue<WeightedSeam, std::vector<WeightedSeam>, WeightedSeamCmp> queue;
    std::unordered_map<ClusteredSeamHandle, double> cost;
    std::unordered_map<ClusteredSeamHandle, double> penalty;
    std::unordered_map<RegionID, std::set<ClusteredSeamHandle>> chartSeamMap;

    std::map<ClusteredSeamHandle, CheckStatus> status;

    std::map<int, std::set<ClusteredSeamHandle>> emap; // endpoint -> seams map

    std::unordered_map<ClusteredSeamHandle, MatchingTransform> transform; // the rigid matching computed for each currently active move
    std::unordered_map<ClusteredSeamHandle, CostInfo::MatchingValue> mvalue;

    std::unordered_map<RegionID, std::set<RegionID>> failed;

    SeamMesh sm;
    std::set<Mesh::FacePointer> changeSet;

    double arapNum;
    double arapDenom;

    double inputUVBorderLength;
    double currentUVBorderLength;
};

void PrepareMesh(Mesh& m, int *vndup);
AlgoStateHandle InitializeState(GraphHandle graph, const AlgoParameters& algoParameters);
void GreedyOptimization(GraphHandle graph, AlgoStateHandle state, const AlgoParameters& params);
void Finalize(GraphHandle graph, const std::string& outname, int *vndup);

// ============================================================================
// Parallel Optimization API
// ============================================================================

// Main parallel optimization function - replaces GreedyOptimization for multi-threaded execution
void GreedyOptimization_Parallel(GraphHandle graph, AlgoStateHandle state, const AlgoParameters& params);

// Configuration for parallel execution
struct ParallelConfig {
    int numThreads;           // Number of worker threads (0 = auto-detect)
    int batchMultiplier;      // Batch size = numThreads * batchMultiplier
    bool enableSpatialCheck;  // Enable "Moving Walls" collision detection
    bool deterministicOrder;  // Commit in priority order for determinism
    size_t heavyTaskThreshold; // Face count above which to throttle concurrency

    ParallelConfig() : numThreads{0}, batchMultiplier{4}, enableSpatialCheck{true},
                       deterministicOrder{true}, heavyTaskThreshold{50000} {}
};

// Set parallel configuration (call before GreedyOptimization_Parallel)
void SetParallelConfig(const ParallelConfig& config);

// Get current parallel configuration
ParallelConfig GetParallelConfig();

#endif // SEAM_REMOVER_H
