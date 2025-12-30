#ifndef HARMONIC_MAP_UTILS_H
#define HARMONIC_MAP_UTILS_H

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

namespace UVDefrag {

struct Triangle { int v[3]; };

class HarmonicMapSolver {
public:
    using Scalar = double;
    using Vector2 = Eigen::Vector2d;
    using SparseMat = Eigen::SparseMatrix<Scalar>;
    using Triplet = Eigen::Triplet<Scalar>;

    HarmonicMapSolver(const std::vector<Vector2>& vertices, 
                      const std::vector<Triangle>& faces,
                      const std::vector<int>& fixedIndices) 
    {
        int numVerts = (int)vertices.size();

        // 1. O(1) Mapping Construction
        // Map global vertex index -> local free/fixed index
        // -1 indicates the opposite type
        globalToLocalFree_.assign(numVerts, -1);
        globalToLocalFixed_.assign(numVerts, -1);
        
        // Mark fixed vertices
        for (int idx : fixedIndices) {
            globalToLocalFixed_[idx] = (int)fixedNodes_.size();
            fixedNodes_.push_back(idx);
        }

        // Identify free vertices
        for (int i = 0; i < numVerts; ++i) {
            if (globalToLocalFixed_[i] == -1) {
                globalToLocalFree_[i] = (int)freeNodes_.size();
                freeNodes_.push_back(i);
            }
        }

        // 2. Assembly (One pass over faces)
        std::vector<Triplet> triplets_ii;
        std::vector<Triplet> triplets_ib;
        
        // Reserve memory to prevent re-allocation (approx 7 entries per row avg)
        triplets_ii.reserve(freeNodes_.size() * 7);
        triplets_ib.reserve(freeNodes_.size() * 2);

        for (const auto& tri : faces) {
            int idx[3] = { tri.v[0], tri.v[1], tri.v[2] };
            Vector2 p[3] = { vertices[idx[0]], vertices[idx[1]], vertices[idx[2]] };

            // Compute Cotangent weights for the 3 edges
            // Edge 0 (1-2) is opposite vertex 0, etc.
            Scalar w[3];
            w[0] = ComputeCotan(p[1], p[2], p[0]); // Edge 1-2 (Angle at 0)
            w[1] = ComputeCotan(p[2], p[0], p[1]); // Edge 2-0 (Angle at 1)
            w[2] = ComputeCotan(p[0], p[1], p[2]); // Edge 0-1 (Angle at 2)

            // Distribute weights to the matrix
            // Edges are: (1,2), (2,0), (0,1)
            int edges[3][2] = {{1,2}, {2,0}, {0,1}};

            for(int k=0; k<3; ++k) {
                int u = idx[edges[k][0]];
                int v = idx[edges[k][1]];
                Scalar weight = 0.5 * w[k]; // 0.5 factor for Dirichlet energy

                // Determine if u, v are fixed or free
                int u_free = globalToLocalFree_[u];
                int v_free = globalToLocalFree_[v];
                int u_fixed = globalToLocalFixed_[u];
                int v_fixed = globalToLocalFixed_[v];

                // Case 1: Both Free -> L_ii (Off-diagonal)
                if (u_free != -1 && v_free != -1) {
                    triplets_ii.emplace_back(u_free, v_free, -weight);
                    triplets_ii.emplace_back(v_free, u_free, -weight);
                    triplets_ii.emplace_back(u_free, u_free, weight);
                    triplets_ii.emplace_back(v_free, v_free, weight);
                }
                // Case 2: u Free, v Fixed -> L_ib and diagonal of L_ii
                else if (u_free != -1 && v_fixed != -1) {
                    triplets_ib.emplace_back(u_free, v_fixed, -weight);
                    triplets_ii.emplace_back(u_free, u_free, weight);
                }
                // Case 3: u Fixed, v Free -> L_ib and diagonal of L_ii
                else if (u_fixed != -1 && v_free != -1) {
                    triplets_ib.emplace_back(v_free, u_fixed, -weight);
                    triplets_ii.emplace_back(v_free, v_free, weight);
                }
                // Case 4: Both Fixed -> Irrelevant for solving interior
            }
        }

        // 3. Build Matrices
        L_ib_ = SparseMat(freeNodes_.size(), fixedNodes_.size());
        L_ib_.setFromTriplets(triplets_ib.begin(), triplets_ib.end());

        SparseMat L_ii(freeNodes_.size(), freeNodes_.size());
        L_ii.setFromTriplets(triplets_ii.begin(), triplets_ii.end());

        // 4. Pre-Factorize
        solver_.compute(L_ii);
    }

    // Returns false if factorization failed (topology issue)
    bool IsValid() const { return solver_.info() == Eigen::Success; }

    /**
     * @brief Updates the full mesh positions based on the current positions of the fixed nodes.
     * The input 'positions' vector MUST contain the valid target positions at the fixed indices.
     */
    void Solve(std::vector<Vector2>& positions) {
        if (!IsValid()) return;

        // Extract boundary conditions (RHS)
        // b = -L_ib * u_boundary
        Eigen::MatrixX2d u_b(fixedNodes_.size(), 2);
        for(size_t i=0; i<fixedNodes_.size(); ++i) {
            u_b.row(i) = positions[fixedNodes_[i]];
        }

        Eigen::MatrixX2d rhs = -(L_ib_ * u_b);
        Eigen::MatrixX2d u_free = solver_.solve(rhs);

        // Update free nodes in the main vector
        for(size_t i=0; i<freeNodes_.size(); ++i) {
            positions[freeNodes_[i]] = u_free.row(i);
        }
    }

    /**
     * @brief Utility: Checks if any triangle in the mesh has flipped (negative signed area).
     * Used for the "Adaptive Back-off" check.
     */
    static bool HasInvertedTriangles(const std::vector<Vector2>& verts, const std::vector<Triangle>& faces) {
        for (const auto& tri : faces) {
            const Vector2& a = verts[tri.v[0]];
            const Vector2& b = verts[tri.v[1]];
            const Vector2& c = verts[tri.v[2]];

            // 2D Cross Product: (b-a) x (c-a)
            double area = (b.x() - a.x()) * (c.y() - a.y()) - 
                          (b.y() - a.y()) * (c.x() - a.x());

            // Check for inversion (epsilon to handle degenerate input safely)
            if (area <= 1e-12) return true;
        }
        return false;
    }

private:
    std::vector<int> freeNodes_;
    std::vector<int> fixedNodes_;
    std::vector<int> globalToLocalFree_;  // O(1) Lookup
    std::vector<int> globalToLocalFixed_; // O(1) Lookup
    
    Eigen::SimplicialLLT<SparseMat> solver_;
    SparseMat L_ib_;

    // Robust Cotangent Calculation
    Scalar ComputeCotan(const Vector2& a, const Vector2& b, const Vector2& c) {
        Vector2 u = b - a;
        Vector2 v = c - a;
        Scalar dot = u.dot(v);
        Scalar cross = u.x()*v.y() - u.y()*v.x();
        
        // CLAMP: Prevent division by zero or infinity on degenerate geometry
        // This is crucial for photogrammetry data.
        if (std::abs(cross) < 1e-8) {
            return 0.0; // Effectively disconnects this part of the triangle
        }
        return dot / cross;
    }
};

/**
 * @brief Helper to run the Composite Harmonic Mapping loop.
 * 
 * @param meshVerts The "Ambient Mesh" vertices (input/output).
 * @param meshFaces The "Ambient Mesh" connectivity.
 * @param boundaryIndices Indices of the boundary vertices.
 * @param targetPositions The pre-calculated straightened positions for the boundary.
 * @return true If successful and valid (no inversions), false if it failed.
 */
inline bool ApplyCompositeWarp(std::vector<Eigen::Vector2d>& meshVerts,
                        const std::vector<Triangle>& meshFaces,
                        const std::vector<int>& boundaryIndices,
                        const std::vector<Eigen::Vector2d>& targetPositions)
{
    // 1. Init Solver (Factorization happens here)
    HarmonicMapSolver solver(meshVerts, meshFaces, boundaryIndices);
    if (!solver.IsValid()) return false;

    // 2. Cache Start Positions
    std::vector<Eigen::Vector2d> startPositions;
    startPositions.reserve(boundaryIndices.size());
    for(int idx : boundaryIndices) startPositions.push_back(meshVerts[idx]);

    // 3. Time-Step Loop (Composite Mapping)
    // We assume 10 steps is a good balance of speed vs safety
    const int steps = 10;
    for (int i = 1; i <= steps; ++i) {
        double t = (double)i / (double)steps;

        // Move boundary
        for (size_t b = 0; b < boundaryIndices.size(); ++b) {
            int vIdx = boundaryIndices[b];
            meshVerts[vIdx] = (1.0 - t) * startPositions[b] + t * targetPositions[b];
        }

        // Relax interior
        solver.Solve(meshVerts);
    }

    // 4. Final Validity Check
    if (HarmonicMapSolver::HasInvertedTriangles(meshVerts, meshFaces)) {
        return false; // Trigger the back-off logic
    }

    return true;
}

} // namespace UVDefrag

#endif // HARMONIC_MAP_UTILS_H