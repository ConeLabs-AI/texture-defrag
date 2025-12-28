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

#ifndef ARAP_H
#define ARAP_H

#include "mesh.h"
#include "arap_simd.h"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <vector>
#include <array>
#include <chrono>


struct ARAPSolveInfo {
    double initialEnergy;
    double finalEnergy;
    int iterations;
    bool numericalError;

#ifdef ARAP_ENABLE_TIMING
    double timePrecompute_ms;
    double timeRotations_ms;
    double timeRHS_ms;
    double timeSolve_ms;
    double timeEnergy_ms;
    double timeTotal_ms;
#endif
};

// Structure-of-Arrays for SIMD-friendly batched processing
struct SoARotations {
    std::vector<double> r00, r01, r10, r11;

    void resize(size_t n) {
        r00.resize(n); r01.resize(n);
        r10.resize(n); r11.resize(n);
    }

    void clear() {
        r00.clear(); r01.clear();
        r10.clear(); r11.clear();
    }
};

struct SoALocalCoords {
    // Local frame coordinates per face (3 vertices, 2D each)
    // Vertex 0 is always at origin, so we only store vertices 1 and 2
    std::vector<double> x1, y1;  // vertex 1
    std::vector<double> x2, y2;  // vertex 2

    void resize(size_t n) {
        x1.resize(n); y1.resize(n);
        x2.resize(n); y2.resize(n);
    }

    void clear() {
        x1.clear(); y1.clear();
        x2.clear(); y2.clear();
    }
};

struct SoAJacobians {
    // Jacobian matrices per face: [[a,b],[c,d]]
    std::vector<double> a, b, c, d;

    void resize(size_t n) {
        a.resize(n); b.resize(n);
        c.resize(n); d.resize(n);
    }
};

class ARAP {

public:

    struct Cot {
        double v[3];
    };

private:

    Mesh& m;

    std::vector<int> fixed_i;
    std::vector<vcg::Point2d> fixed_pos;

    // Reduced system mapping
    std::vector<int> global_to_local_idx;
    std::vector<int> local_to_global_idx;
    int n_free_verts = 0;

    // The Solver (Persistent)
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;

    // SoA data for SIMD-optimized processing
    SoALocalCoords soa_local_coords;
    SoARotations soa_rotations;

    int max_iter;
    bool verbose = false;

    void ComputeSystemMatrix(Mesh& m, const std::vector<Cot>& cotan, Eigen::SparseMatrix<double>& L);
    void ComputeRHSSIMD(Mesh& m, const std::vector<Cot>& cotan, Eigen::VectorXd& bu, Eigen::VectorXd& bv);
    void PrecomputeData();
    void ComputeRotationsSIMD(Mesh& m);
    double CurrentEnergySIMD();

public:

    ARAP(Mesh& mesh);

    double CurrentEnergy();
    void FixVertex(Mesh::ConstVertexPointer vp, const vcg::Point2d& pos);
    void FixBoundaryVertices();
    int FixSelectedVertices();
    int FixRandomEdgeWithinTolerance(double tol);
    void SetMaxIterations(int n);
    void SetVerbose(bool v) { verbose = v; }

    ARAPSolveInfo Solve();

#ifdef ARAP_ENABLE_TIMING
    struct AggregateStats {
        double total_time_ms = 0;
        double precompute_ms = 0;
        double rotations_ms = 0;
        double rhs_ms = 0;
        double solve_ms = 0;
        double energy_ms = 0;
        long long total_iterations = 0;
        int call_count = 0;

        // Problem scale histograms
        long long count_small = 0;  // < 100 vertices
        long long count_medium = 0; // 100-1000 vertices
        long long count_large = 0;  // > 1000 vertices
    };
    static AggregateStats globalStats;
    static void PrintAggregateStats();
#endif

    static double ComputeEnergyFromStoredWedgeTC(Mesh& m, double *num, double *denom);
    static double ComputeEnergyFromStoredWedgeTC(const std::vector<Mesh::FacePointer>& fpVec, Mesh& m, double *num, double *denom);
    static double ComputeEnergy(const vcg::Point2d& x10, const vcg::Point2d& x20,
                                const vcg::Point2d& u10, const vcg::Point2d& u20,
                                double *area);
};


#endif // ARAP_H