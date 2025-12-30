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

#include "arap.h"

#include "mesh_attribute.h"
#include "logging.h"
#include "math_utils.h"

#include <iomanip>
#include <unordered_set>
#include <omp.h>


ARAP::ARAP(Mesh& mesh)
    : m{mesh},
      max_iter{100}
{
}

void ARAP::FixVertex(Mesh::ConstVertexPointer vp, const vcg::Point2d& pos)
{
    fixed_i.push_back(tri::Index(m, vp));
    fixed_pos.push_back(pos);
}

void ARAP::FixBoundaryVertices()
{
    for (auto& v : m.vert) {
        if (v.IsB()) {
            fixed_i.push_back(tri::Index(m, v));
            fixed_pos.push_back(v.T().P());
        }
    }
}

int ARAP::FixSelectedVertices()
{
    int nfixed = 0;
    for (auto& v : m.vert) {
        if (v.IsS()) {
            fixed_i.push_back(tri::Index(m, v));
            fixed_pos.push_back(v.T().P());
            nfixed++;
        }
    }
    return nfixed;
}

/* This function fixes the vertices of an edge that is within 2pct of the target
 * edge length */
int ARAP::FixRandomEdgeWithinTolerance(double tol)
{
    std::unordered_set<int> fixed;
    for (int i : fixed_i)
        fixed.insert(i);

    auto tsa = GetTargetShapeAttribute(m);
    for (auto& f : m.face) {
        for (int i = 0; i < 3; ++i) {
            double dcurr = (f.WT(i).P() - f.WT(f.Next(i)).P()).Norm();
            double dtarget = (tsa[f].P[i] - tsa[f].P[f.Next(i)]).Norm();
            if (std::abs((dcurr - dtarget) / dtarget) < tol) {
                if (fixed.count(tri::Index(m, f.V(i))) == 0 && fixed.count(tri::Index(m, f.V(f.Next(i)))) == 0) {
                    FixVertex(f.V(i), f.WT(i).P());
                    FixVertex(f.V(f.Next(i)), f.WT(f.Next(i)).P());
                    LOG_DEBUG << "Fixing vertices " << tri::Index(m, f.V(i)) << "   " << tri::Index(m, f.V(f.Next(i)));
                    return 2;
                }
            }
        }
    }
    return 0;
}

void ARAP::SetMaxIterations(int n)
{
    max_iter = n;
}

static std::vector<ARAP::Cot> ComputeCotangentVector(Mesh& m)
{
    std::vector<ARAP::Cot> cotan;
    cotan.resize(m.FN());
    auto tsa = GetTargetShapeAttribute(m);
    double eps = std::numeric_limits<double>::epsilon();
    #pragma omp parallel for if(m.FN() > 1000)
    for (int fi = 0; fi < m.FN(); ++fi) {
        auto& f = m.face[fi];
        ARAP::Cot c;
        for (int i = 0; i < 3; ++i) {
            int j = (i+1)%3;
            int k = (i+2)%3;
            double alpha_i = std::max(VecAngle(tsa[f].P[j] - tsa[f].P[i], tsa[f].P[k] - tsa[f].P[i]), eps);
            c.v[i] = 0.5 * std::tan(M_PI_2 - alpha_i);
        }
        cotan[fi] = c;
    }
    return cotan;
}

void ARAP::ComputeSystemMatrix(Mesh& m, const std::vector<Cot>& cotan, Eigen::SparseMatrix<double>& L)
{
    using Td = Eigen::Triplet<double>;

    L.resize(n_free_verts, n_free_verts);
    L.setZero();
    std::vector<Td> tri;

    #pragma omp parallel if(m.FN() > 1000)
    {
        std::vector<Td> tri_private;
        #pragma omp for nowait
        for (int fi = 0; fi < m.FN(); ++fi) {
            auto &f = m.face[fi];
            for (int i = 0; i < 3; ++i) {
                int global_i = (int) tri::Index(m, f.V0(i));
                int global_j = (int) tri::Index(m, f.V1(i));
                int global_k = (int) tri::Index(m, f.V2(i));

                int local_i = global_to_local_idx[global_i];
                int local_j = global_to_local_idx[global_j];
                int local_k = global_to_local_idx[global_k];

                if (local_i != -1) {
                    int j_idx = (i+1)%3;
                    int k_idx = (i+2)%3;
                    double weight_ij = cotan[fi].v[k_idx];
                    double weight_ik = cotan[fi].v[j_idx];

                    if (!std::isfinite(weight_ij)) weight_ij = 1e-8;
                    if (!std::isfinite(weight_ik)) weight_ik = 1e-8;

                    // i is Free, check j and k
                    if (local_j != -1) {
                        tri_private.push_back(Td(local_i, local_j, -weight_ij));
                    }
                    if (local_k != -1) {
                        tri_private.push_back(Td(local_i, local_k, -weight_ik));
                    }
                    tri_private.push_back(Td(local_i, local_i, weight_ij + weight_ik));
                }
            }
        }
        #pragma omp critical
        tri.insert(tri.end(), tri_private.begin(), tri_private.end());
    }

    L.setFromTriplets(tri.begin(), tri.end());
    L.makeCompressed();
}

void ARAP::ComputeRotations(Mesh& m, std::vector<Eigen::Matrix2d>& rotations)
{
    rotations.resize(m.FN());
    #pragma omp parallel for if(m.FN() > 1000)
    for (int fi = 0; fi < m.FN(); ++fi) {
        auto& f = m.face[fi];

        // 1. Use precomputed 2D rest shape coordinates
        const auto& t = local_frame_coords[fi];

        // 2. Compute Jacobian (Transformation from Rest to Current UV)
        // Uses the original stable math from math_utils.h
        auto u10 = f.WT(1).P() - f.WT(0).P();
        auto u20 = f.WT(2).P() - f.WT(0).P();
        Eigen::Matrix2d Jf = ComputeTransformationMatrix(
            t[1], t[2], // Using precomputed x10, x20
            Eigen::Vector2d(u10.X(), u10.Y()),
            Eigen::Vector2d(u20.X(), u20.Y())
        );

        // SAFETY CHECK: Handle degenerate faces (Zero Area).
        // ComputeTransformationMatrix involves an inverse(); if the triangle area is 0,
        // Jf will contain Inf/NaN. Feeding Inf to SVD breaks the solver.
        if (!Jf.allFinite()) {
            rotations[fi].setIdentity();
            continue;
        }

        // 3. Compute SVD using Eigen (Stable)
        // Eigen::JacobiSVD is robust and accurate.
        Eigen::JacobiSVD<Eigen::Matrix2d> svd(Jf, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix2d U = svd.matrixU();
        Eigen::Matrix2d V = svd.matrixV();

        // 4. Extract Rotation
        Eigen::Matrix2d R = U * V.transpose();

        // 5. Fix Reflection (Ensure determinant is +1)
        if (R.determinant() < 0) {
            U.col(1) *= -1;
            R = U * V.transpose();
        }

        rotations[fi] = R;
    }
}

void ARAP::ComputeRHS(Mesh& m, const std::vector<Eigen::Matrix2d>& rotations, const std::vector<Cot>& cotan, Eigen::VectorXd& bu, Eigen::VectorXd& bv)
{
    // 1. Clear the global target vectors
    bu.setZero(n_free_verts);
    bv.setZero(n_free_verts);

    // 2. Start Parallel Region
    #pragma omp parallel if(m.FN() > 1000)
    {
        // Each thread gets its own private buffer to accumulate forces
        Eigen::VectorXd bu_private = Eigen::VectorXd::Zero(n_free_verts);
        Eigen::VectorXd bv_private = Eigen::VectorXd::Zero(n_free_verts);

        // Distribute faces across threads
        #pragma omp for nowait
        for (int fi = 0; fi < m.FN(); ++fi) {
            auto &f = m.face[fi];
            const Eigen::Matrix2d& Rf = rotations[fi];

            // Use precomputed local frame coordinates
            const auto& t = local_frame_coords[fi];

            for (int i = 0; i < 3; ++i) {
                int global_i = (int) tri::Index(m, f.V0(i));
                int local_i = global_to_local_idx[global_i];

                // If vertex i is fixed/boundary, it doesn't have an equation in the solver
                if (local_i == -1) continue;

                int j_local_idx = (i+1)%3;
                int k_local_idx = (i+2)%3;

                // Use the exact cotangent weights from the original math
                double weight_ij = cotan[fi].v[k_local_idx];
                double weight_ik = cotan[fi].v[j_local_idx];

                // Standard ARAP Right-Hand Side calculation: sum( w * R * (t_i - t_j) )
                Eigen::Vector2d x_ij = t[i] - t[j_local_idx];
                Eigen::Vector2d x_ik = t[i] - t[k_local_idx];
                Eigen::Vector2d rhs_rot = (weight_ij * Rf) * x_ij + (weight_ik * Rf) * x_ik;

                bu_private(local_i) += rhs_rot.x();
                bv_private(local_i) += rhs_rot.y();

                // Boundary Condition: If neighbor is fixed, its position is a constant
                int global_j = (int) tri::Index(m, f.V1(i));
                if (global_to_local_idx[global_j] == -1) {
                    vcg::Point2d fixed_pos_j = m.vert[global_j].T().P();
                    bu_private(local_i) += weight_ij * fixed_pos_j.X();
                    bv_private(local_i) += weight_ij * fixed_pos_j.Y();
                }

                int global_k = (int) tri::Index(m, f.V2(i));
                if (global_to_local_idx[global_k] == -1) {
                    vcg::Point2d fixed_pos_k = m.vert[global_k].T().P();
                    bu_private(local_i) += weight_ik * fixed_pos_k.X();
                    bv_private(local_i) += weight_ik * fixed_pos_k.Y();
                }
            }
        }

        // 3. Merge thread-local results into the main vectors
        #pragma omp critical
        {
            bu += bu_private;
            bv += bv_private;
        }
    }
}

double ARAP::ComputeEnergy(const vcg::Point2d& x10, const vcg::Point2d& x20,
                           const vcg::Point2d& u10, const vcg::Point2d& u20,
                           double *area)
{
    *area = std::abs(x10 ^ x20);
    Eigen::Matrix2d Jf = ComputeTransformationMatrix(x10, x20, u10, u20);
    Eigen::Matrix2d U, V;
    Eigen::Vector2d sigma;
    Eigen::JacobiSVD<Eigen::Matrix2d> svd;
    svd.compute(Jf, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U = svd.matrixU(); V = svd.matrixV(); sigma = svd.singularValues();
    return std::pow(sigma[0] - 1.0, 2.0) + std::pow(sigma[1] - 1.0, 2.0);
}

double ARAP::ComputeEnergyFromStoredWedgeTC(const std::vector<Mesh::FacePointer>& fpVec, Mesh& m, double *num, double *denom)
{
    double n = 0;
    double d = 0;
    auto tsa = GetWedgeTexCoordStorageAttribute(m);
    #pragma omp parallel for reduction(+:n, d) if(fpVec.size() > 1000)
    for (std::size_t i = 0; i < fpVec.size(); ++i) {
        auto fptr = fpVec[i];
        vcg::Point2d x10 = tsa[fptr].tc[1].P() - tsa[fptr].tc[0].P();
        vcg::Point2d x20 = tsa[fptr].tc[2].P() - tsa[fptr].tc[0].P();
        vcg::Point2d u10 = fptr->WT(1).P() - fptr->WT(0).P();
        vcg::Point2d u20 = fptr->WT(2).P() - fptr->WT(0).P();
        double area;
        double energy = ComputeEnergy(x10, x20, u10, u20, &area);
        if (area > 0) {
            n += (area * energy);
            d += area;
        }
    }
    if (num)
        *num = n;
    if (denom)
        *denom = d;
    return n / d;
}

double ARAP::ComputeEnergyFromStoredWedgeTC(Mesh& m, double *num, double *denom)
{
    double e = 0;
    double total_area = 0;
    auto tsa = GetWedgeTexCoordStorageAttribute(m);
    #pragma omp parallel for reduction(+:e, total_area) if(m.FN() > 1000)
    for (int fi = 0; fi < m.FN(); ++fi) {
        auto& f = m.face[fi];
        vcg::Point2d x10 = tsa[f].tc[1].P() - tsa[f].tc[0].P();
        vcg::Point2d x20 = tsa[f].tc[2].P() - tsa[f].tc[0].P();
        double area_f = std::abs(x10 ^ x20);
        if (area_f > 0) {
            Eigen::Matrix2d Jf = ComputeTransformationMatrix(x10, x20, f.WT(1).P() - f.WT(0).P(), f.WT(2).P() - f.WT(0).P());
            Eigen::Matrix2d U, V;
            Eigen::Vector2d sigma;
            Eigen::JacobiSVD<Eigen::Matrix2d> svd;
            svd.compute(Jf, Eigen::ComputeFullU | Eigen::ComputeFullV);
            U = svd.matrixU(); V = svd.matrixV(); sigma = svd.singularValues();
            total_area += area_f;
            e += area_f * (std::pow(sigma[0] - 1.0, 2.0) + std::pow(sigma[1] - 1.0, 2.0));
        }
    }
    if (num)
        *num = e;
    if (denom)
        *denom = total_area;
    return e / total_area;
}

double ARAP::CurrentEnergy()
{
    double e = 0;
    double total_area = 0;
    auto tsa = GetTargetShapeAttribute(m);

    // THRESHOLD: Only parallelize if we have enough work (>1000 faces)
    // Otherwise run in serial to avoid OpenMP overhead on small charts.
    #pragma omp parallel for reduction(+:e, total_area) if(m.FN() > 1000)
    for (int fi = 0; fi < m.FN(); ++fi) {
        auto& f = m.face[fi];

        // 1. Retrieve Precomputed Data (Commit 3)
        const auto& t = local_frame_coords[fi];
        const Eigen::Vector2d& x10 = t[1];
        const Eigen::Vector2d& x20 = t[2];

        // 2. Compute 3D Rest Area (Cross Product)
        double area_f = std::abs(x10.x() * x20.y() - x20.x() * x10.y()) * 0.5;

        // Skip degenerate/zero-area faces
        if (area_f <= 1e-12) continue;

        // 3. Convert Current UVs to Eigen
        vcg::Point2d uv10_vcg = f.WT(1).P() - f.WT(0).P();
        vcg::Point2d uv20_vcg = f.WT(2).P() - f.WT(0).P();
        Eigen::Vector2d u10(uv10_vcg.X(), uv10_vcg.Y());
        Eigen::Vector2d u20(uv20_vcg.X(), uv20_vcg.Y());

        // 4. Compute Jacobian
        Eigen::Matrix2d Jf = ComputeTransformationMatrix(x10, x20, u10, u20);

        // Safety check
        if (!Jf.allFinite()) continue;

        // 5. SVD & Energy
        Eigen::JacobiSVD<Eigen::Matrix2d> svd(Jf, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector2d sigma = svd.singularValues();

        total_area += area_f;
        e += area_f * (std::pow(sigma[0] - 1.0, 2.0) + std::pow(sigma[1] - 1.0, 2.0));
    }

    return (total_area > 0) ? e / total_area : 0;
}

ARAPSolveInfo ARAP::Solve()
{
    ARAPSolveInfo si = {0, 0, 0, false};
    std::vector<Cot> cotan = ComputeCotangentVector(m);

    PrecomputeData();

    // 1. Establish the mapping [0, N_free]
    global_to_local_idx.assign(m.VN(), -1);
    local_to_global_idx.clear();
    n_free_verts = 0;

    // We use a set for O(log N) lookup or a bool array for O(1)
    std::vector<bool> is_fixed(m.VN(), false);
    for (int idx : fixed_i) {
        if (idx >= 0 && idx < m.VN())
            is_fixed[idx] = true;
    }

    for (int i = 0; i < m.VN(); ++i) {
        if (!is_fixed[i]) {
            global_to_local_idx[i] = n_free_verts;
            local_to_global_idx.push_back(i);
            n_free_verts++;
        }
    }

    // Ensure fixed vertices are at their fixed positions
    for (unsigned i = 0; i < fixed_i.size(); ++i) {
        m.vert[fixed_i[i]].T().P() = fixed_pos[i];
    }

    // 2. Build and Factorize the System Matrix
    Eigen::SparseMatrix<double> L;
    ComputeSystemMatrix(m, cotan, L);

    solver.compute(L);
    if (solver.info() != Eigen::Success) {
        LOG_WARN << "ARAP: Matrix factorization failed: " << solver.info();
        si.numericalError = true;
        return si;
    }

    si.initialEnergy = CurrentEnergy();
    LOG_DEBUG << "ARAP: Starting energy is " << si.initialEnergy;

    double e = si.initialEnergy;
    bool converged = false;
    int iter = 0;

    std::vector<Eigen::Matrix2d> rotations;
    Eigen::VectorXd bu, bv;

    while (!converged && iter < max_iter) {
        ComputeRotations(m, rotations);
        ComputeRHS(m, rotations, cotan, bu, bv);

        Eigen::VectorXd xu = solver.solve(bu);
        if (solver.info() != Eigen::Success) {
            LOG_WARN << "ARAP: solve(u) failed";
            si.numericalError = true;
            break;
        }

        Eigen::VectorXd xv = solver.solve(bv);
        if (solver.info() != Eigen::Success) {
            LOG_WARN << "ARAP: solve(v) failed";
            si.numericalError = true;
            break;
        }

        // 3. Unpack back to Mesh
        #pragma omp parallel for if(n_free_verts > 1000)
        for (int local_i = 0; local_i < n_free_verts; ++local_i) {
            int global_i = local_to_global_idx[local_i];
            m.vert[global_i].T().P().X() = xu(local_i);
            m.vert[global_i].T().P().Y() = xv(local_i);
        }

        #pragma omp parallel for if(m.FN() > 1000)
        for (int fi = 0; fi < m.FN(); ++fi) {
            auto& f = m.face[fi];
            for (int i = 0; i < 3; ++i) {
                f.WT(i).P() = f.V(i)->T().P();
            }
        }

        double e_curr = CurrentEnergy();
        si.finalEnergy = e_curr;

        double delta_e = e - e_curr;
        if (delta_e < 1e-8) {
            LOG_DEBUG << "ARAP: convergence reached (change in the energy value is too small)";
            converged = true;
        }

        e = e_curr;
        iter++;
    }

    si.iterations = iter;

    if (iter == max_iter) {
        LOG_DEBUG << "ARAP: iteration limit reached";
    }

    LOG_DEBUG << "ARAP: Energy after optimization is " << CurrentEnergy() << " (" << iter << " iterations)";

    // Final update for wedges
    for (auto& f : m.face) {
        for (int i = 0; i < 3; ++i) {
            f.WT(i).P() = f.cV(i)->T().P();
        }
    }

    return si;
}

void ARAP::PrecomputeData()
{
    local_frame_coords.resize(m.FN());
    auto tsa = GetTargetShapeAttribute(m);
    #pragma omp parallel for if(m.FN() > 1000)
    for (int fi = 0; fi < m.FN(); ++fi) {
        auto& f = m.face[fi];
        vcg::Point2d x10, x20;

        // Calculate the 2D isometry once
        LocalIsometry(tsa[f].P[1] - tsa[f].P[0], tsa[f].P[2] - tsa[f].P[0], x10, x20);

        local_frame_coords[fi][0] = Eigen::Vector2d::Zero();
        local_frame_coords[fi][1] = Eigen::Vector2d(x10.X(), x10.Y());
        local_frame_coords[fi][2] = Eigen::Vector2d(x20.X(), x20.Y());
    }
}