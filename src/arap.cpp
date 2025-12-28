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

#ifdef ARAP_ENABLE_TIMING
#include <chrono>
#define ARAP_TIMER_START(name) auto timer_##name = std::chrono::high_resolution_clock::now()
#define ARAP_TIMER_END(name, var) var = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - timer_##name).count()

ARAP::AggregateStats ARAP::globalStats;

void ARAP::PrintAggregateStats() {
    if (globalStats.call_count == 0) return;
    LOG_INFO << "======= ARAP AGGREGATE STATS (" << globalStats.call_count << " calls) =======";
    LOG_INFO << "Total Time:  " << std::fixed << std::setprecision(3) << globalStats.total_time_ms << " ms";
    LOG_INFO << "  Precompute: " << globalStats.precompute_ms << " ms";
    if (globalStats.total_iterations > 0) {
        LOG_INFO << "  Rotations:  " << globalStats.rotations_ms << " ms (" << (globalStats.rotations_ms / globalStats.total_iterations) << " ms/iter)";
        LOG_INFO << "  RHS:        " << globalStats.rhs_ms << " ms (" << (globalStats.rhs_ms / globalStats.total_iterations) << " ms/iter)";
        LOG_INFO << "  Solve:      " << globalStats.solve_ms << " ms (" << (globalStats.solve_ms / globalStats.total_iterations) << " ms/iter)";
    }
    LOG_INFO << "  Energy:     " << globalStats.energy_ms << " ms";
    LOG_INFO << "Total Iters: " << globalStats.total_iterations;
    LOG_INFO << "Problem Sizes: <100: " << globalStats.count_small 
             << ", 100-1k: " << globalStats.count_medium 
             << ", >1k: " << globalStats.count_large;
    LOG_INFO << "=================================================";
}
#else
#define ARAP_TIMER_START(name)
#define ARAP_TIMER_END(name, var)
#endif


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
    #pragma omp parallel for
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

    #pragma omp parallel
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

static void ComputeRotations(Mesh& m, std::vector<Eigen::Matrix2d>& rotations)
{
    auto tsa = GetTargetShapeAttribute(m);
    rotations.resize(m.FN());
    #pragma omp parallel for
    for (int fi = 0; fi < m.FN(); ++fi) {
        auto& f = m.face[fi];
        vcg::Point2d x10, x20;
        LocalIsometry(tsa[f].P[1] - tsa[f].P[0], tsa[f].P[2] - tsa[f].P[0], x10, x20);
        Eigen::Matrix2d Jf = ComputeTransformationMatrix(x10, x20, f.WT(1).P() - f.WT(0).P(), f.WT(2).P() - f.WT(0).P());
        Eigen::Matrix2d U, V;
        Eigen::Vector2d sigma;
        Eigen::JacobiSVD<Eigen::Matrix2d> svd;
        svd.compute(Jf, Eigen::ComputeFullU | Eigen::ComputeFullV);
        U = svd.matrixU(); V = svd.matrixV(); sigma = svd.singularValues();
        Eigen::MatrixXd R = U * V.transpose();
        if (R.determinant() < 0) {
            U.col(U.cols() - 1) *= -1;
            R = U * V.transpose();
        }

        rotations[fi] = R;
    }
}

void ARAP::ComputeRotationsSIMD(Mesh& m)
{
    const int nf = m.FN();
    soa_rotations.resize(nf);

    auto tsa = GetTargetShapeAttribute(m);

#ifdef ARAP_USE_AVX2
    // Process 4 faces at a time with AVX2
    constexpr int batch_size = 4;
    const int nf_aligned = (nf / batch_size) * batch_size;

    // Temporary buffers for batched processing
    alignas(32) double jf_a[4], jf_b[4], jf_c[4], jf_d[4];
    alignas(32) double u00[4], u01[4], u10[4], u11[4];
    alignas(32) double v00[4], v01[4], v10[4], v11[4];
    alignas(32) double sigma0[4], sigma1[4];
    alignas(32) double r00[4], r01[4], r10[4], r11[4];

    #pragma omp parallel for firstprivate(jf_a, jf_b, jf_c, jf_d, u00, u01, u10, u11, v00, v01, v10, v11, sigma0, sigma1, r00, r01, r10, r11)
    for (int fi = 0; fi < nf_aligned; fi += batch_size) {
        // Load and compute Jacobians for 4 faces
        for (int k = 0; k < batch_size; ++k) {
            int idx = fi + k;
            auto& f = m.face[idx];

            // Get local frame coordinates
            double x10_x = soa_local_coords.x1[idx];
            double x10_y = soa_local_coords.y1[idx];
            double x20_x = soa_local_coords.x2[idx];
            double x20_y = soa_local_coords.y2[idx];

            // Current UV coordinates
            double u10_x = f.WT(1).P().X() - f.WT(0).P().X();
            double u10_y = f.WT(1).P().Y() - f.WT(0).P().Y();
            double u20_x = f.WT(2).P().X() - f.WT(0).P().X();
            double u20_y = f.WT(2).P().Y() - f.WT(0).P().Y();

            // Compute Jacobian: J = G * F^-1 where F = [x10, x20], G = [u10, u20]
            // F^-1 = (1/det) * [[x20_y, -x20_x], [-x10_y, x10_x]]
            double det = x10_x * x20_y - x20_x * x10_y;
            if (std::abs(det) < 1e-300) det = 1e-300;
            double inv_det = 1.0 / det;

            double f_inv00 = x20_y * inv_det;
            double f_inv01 = -x20_x * inv_det;
            double f_inv10 = -x10_y * inv_det;
            double f_inv11 = x10_x * inv_det;

            // J = G * F^-1
            jf_a[k] = u10_x * f_inv00 + u20_x * f_inv10;
            jf_b[k] = u10_x * f_inv01 + u20_x * f_inv11;
            jf_c[k] = u10_y * f_inv00 + u20_y * f_inv10;
            jf_d[k] = u10_y * f_inv01 + u20_y * f_inv11;
        }

        // Batched SVD
        arap_simd::batch_svd2x2_avx2(jf_a, jf_b, jf_c, jf_d,
                                      u00, u01, u10, u11,
                                      v00, v01, v10, v11,
                                      sigma0, sigma1);

        // Batched rotation extraction: R = U * V^T
        arap_simd::batch_rotation_from_svd_avx2(u00, u01, u10, u11,
                                                 v00, v01, v10, v11,
                                                 r00, r01, r10, r11);

        // Store rotations
        for (int k = 0; k < batch_size; ++k) {
            int idx = fi + k;
            soa_rotations.r00[idx] = r00[k];
            soa_rotations.r01[idx] = r01[k];
            soa_rotations.r10[idx] = r10[k];
            soa_rotations.r11[idx] = r11[k];
        }
    }

    // Handle remainder faces with scalar code
    for (int fi = nf_aligned; fi < nf; ++fi) {
        auto& f = m.face[fi];

        double x10_x = soa_local_coords.x1[fi];
        double x10_y = soa_local_coords.y1[fi];
        double x20_x = soa_local_coords.x2[fi];
        double x20_y = soa_local_coords.y2[fi];

        double u10_x = f.WT(1).P().X() - f.WT(0).P().X();
        double u10_y = f.WT(1).P().Y() - f.WT(0).P().Y();
        double u20_x = f.WT(2).P().X() - f.WT(0).P().X();
        double u20_y = f.WT(2).P().Y() - f.WT(0).P().Y();

        double det = x10_x * x20_y - x20_x * x10_y;
        if (std::abs(det) < 1e-300) det = 1e-300;
        double inv_det = 1.0 / det;

        double f_inv00 = x20_y * inv_det;
        double f_inv01 = -x20_x * inv_det;
        double f_inv10 = -x10_y * inv_det;
        double f_inv11 = x10_x * inv_det;

        double a = u10_x * f_inv00 + u20_x * f_inv10;
        double b = u10_x * f_inv01 + u20_x * f_inv11;
        double c = u10_y * f_inv00 + u20_y * f_inv10;
        double d = u10_y * f_inv01 + u20_y * f_inv11;

        double u00_s, u01_s, u10_s, u11_s;
        double v00_s, v01_s, v10_s, v11_s;
        double s0, s1;

        arap_simd::scalar::svd2x2(a, b, c, d,
                                   u00_s, u01_s, u10_s, u11_s,
                                   v00_s, v01_s, v10_s, v11_s,
                                   s0, s1);

        arap_simd::scalar::rotation_from_svd(u00_s, u01_s, u10_s, u11_s,
                                              v00_s, v01_s, v10_s, v11_s,
                                              soa_rotations.r00[fi], soa_rotations.r01[fi],
                                              soa_rotations.r10[fi], soa_rotations.r11[fi]);
    }
#else
    // Scalar fallback
    #pragma omp parallel for
    for (int fi = 0; fi < nf; ++fi) {
        auto& f = m.face[fi];

        double x10_x = soa_local_coords.x1[fi];
        double x10_y = soa_local_coords.y1[fi];
        double x20_x = soa_local_coords.x2[fi];
        double x20_y = soa_local_coords.y2[fi];

        double u10_x = f.WT(1).P().X() - f.WT(0).P().X();
        double u10_y = f.WT(1).P().Y() - f.WT(0).P().Y();
        double u20_x = f.WT(2).P().X() - f.WT(0).P().X();
        double u20_y = f.WT(2).P().Y() - f.WT(0).P().Y();

        double det = x10_x * x20_y - x20_x * x10_y;
        if (std::abs(det) < 1e-300) det = 1e-300;
        double inv_det = 1.0 / det;

        double f_inv00 = x20_y * inv_det;
        double f_inv01 = -x20_x * inv_det;
        double f_inv10 = -x10_y * inv_det;
        double f_inv11 = x10_x * inv_det;

        double a = u10_x * f_inv00 + u20_x * f_inv10;
        double b = u10_x * f_inv01 + u20_x * f_inv11;
        double c = u10_y * f_inv00 + u20_y * f_inv10;
        double d = u10_y * f_inv01 + u20_y * f_inv11;

        double u00_s, u01_s, u10_s, u11_s;
        double v00_s, v01_s, v10_s, v11_s;
        double s0, s1;

        arap_simd::scalar::svd2x2(a, b, c, d,
                                   u00_s, u01_s, u10_s, u11_s,
                                   v00_s, v01_s, v10_s, v11_s,
                                   s0, s1);

        arap_simd::scalar::rotation_from_svd(u00_s, u01_s, u10_s, u11_s,
                                              v00_s, v01_s, v10_s, v11_s,
                                              soa_rotations.r00[fi], soa_rotations.r01[fi],
                                              soa_rotations.r10[fi], soa_rotations.r11[fi]);
    }
#endif
}

void ARAP::ComputeRHS(Mesh& m, const std::vector<Eigen::Matrix2d>& rotations, const std::vector<Cot>& cotan, Eigen::VectorXd& bu, Eigen::VectorXd& bv)
{
    bu.setZero(n_free_verts);
    bv.setZero(n_free_verts);

    #pragma omp parallel
    {
        Eigen::VectorXd bu_private = Eigen::VectorXd::Zero(n_free_verts);
        Eigen::VectorXd bv_private = Eigen::VectorXd::Zero(n_free_verts);
        #pragma omp for nowait
        for (int fi = 0; fi < m.FN(); ++fi) {
            auto &f = m.face[fi];
            const Eigen::Matrix2d& Rf = rotations[fi];
            const auto& t = local_frame_coords[fi];

            for (int i = 0; i < 3; ++i) {
                int global_i = (int) tri::Index(m, f.V0(i));
                int local_i = global_to_local_idx[global_i];

                if (local_i != -1) {
                    int j_local_idx = (i+1)%3;
                    int k_local_idx = (i+2)%3;

                    int global_j = (int) tri::Index(m, f.V1(i));
                    int global_k = (int) tri::Index(m, f.V2(i));
                    int local_j = global_to_local_idx[global_j];
                    int local_k = global_to_local_idx[global_k];

                    double weight_ij = cotan[fi].v[k_local_idx];
                    double weight_ik = cotan[fi].v[j_local_idx];

                    if (!std::isfinite(weight_ij)) weight_ij = 1e-8;
                    if (!std::isfinite(weight_ik)) weight_ik = 1e-8;

                    // 1. Standard ARAP Rotation Forces
                    Eigen::Vector2d x_ij = t[i] - t[j_local_idx];
                    Eigen::Vector2d x_ik = t[i] - t[k_local_idx];
                    Eigen::Vector2d rhs_rot = (weight_ij * Rf) * x_ij + (weight_ik * Rf) * x_ik;

                    bu_private(local_i) += rhs_rot.x();
                    bv_private(local_i) += rhs_rot.y();

                    // 2. Fixed Vertex Pull (Boundary Conditions)
                    if (local_j == -1) {
                        vcg::Point2d fixed_pos_j = m.vert[global_j].T().P();
                        bu_private(local_i) += weight_ij * fixed_pos_j.X();
                        bv_private(local_i) += weight_ij * fixed_pos_j.Y();
                    }
                    if (local_k == -1) {
                        vcg::Point2d fixed_pos_k = m.vert[global_k].T().P();
                        bu_private(local_i) += weight_ik * fixed_pos_k.X();
                        bv_private(local_i) += weight_ik * fixed_pos_k.Y();
                    }
                }
            }
        }
        #pragma omp critical
        {
            bu += bu_private;
            bv += bv_private;
        }
    }
}

void ARAP::ComputeRHSSIMD(Mesh& m, const std::vector<Cot>& cotan, Eigen::VectorXd& bu, Eigen::VectorXd& bv)
{
    bu.setZero(n_free_verts);
    bv.setZero(n_free_verts);

    #pragma omp parallel
    {
        Eigen::VectorXd bu_private = Eigen::VectorXd::Zero(n_free_verts);
        Eigen::VectorXd bv_private = Eigen::VectorXd::Zero(n_free_verts);
        #pragma omp for nowait
        for (int fi = 0; fi < m.FN(); ++fi) {
            auto &f = m.face[fi];

            // Get rotation from SoA storage
            double rf00 = soa_rotations.r00[fi];
            double rf01 = soa_rotations.r01[fi];
            double rf10 = soa_rotations.r10[fi];
            double rf11 = soa_rotations.r11[fi];

            // Get local frame coordinates
            double t0_x = 0.0, t0_y = 0.0;  // vertex 0 at origin
            double t1_x = soa_local_coords.x1[fi];
            double t1_y = soa_local_coords.y1[fi];
            double t2_x = soa_local_coords.x2[fi];
            double t2_y = soa_local_coords.y2[fi];

            for (int i = 0; i < 3; ++i) {
                int global_i = (int) tri::Index(m, f.V0(i));
                int local_i = global_to_local_idx[global_i];

                if (local_i != -1) {
                    int j_local_idx = (i+1)%3;
                    int k_local_idx = (i+2)%3;

                    int global_j = (int) tri::Index(m, f.V1(i));
                    int global_k = (int) tri::Index(m, f.V2(i));
                    int local_j = global_to_local_idx[global_j];
                    int local_k = global_to_local_idx[global_k];

                    double weight_ij = cotan[fi].v[k_local_idx];
                    double weight_ik = cotan[fi].v[j_local_idx];

                    if (!std::isfinite(weight_ij)) weight_ij = 1e-8;
                    if (!std::isfinite(weight_ik)) weight_ik = 1e-8;

                    // Get the local coordinates for vertices i, j, k
                    double ti_x, ti_y, tj_x, tj_y, tk_x, tk_y;
                    if (i == 0) { ti_x = t0_x; ti_y = t0_y; }
                    else if (i == 1) { ti_x = t1_x; ti_y = t1_y; }
                    else { ti_x = t2_x; ti_y = t2_y; }

                    if (j_local_idx == 0) { tj_x = t0_x; tj_y = t0_y; }
                    else if (j_local_idx == 1) { tj_x = t1_x; tj_y = t1_y; }
                    else { tj_x = t2_x; tj_y = t2_y; }

                    if (k_local_idx == 0) { tk_x = t0_x; tk_y = t0_y; }
                    else if (k_local_idx == 1) { tk_x = t1_x; tk_y = t1_y; }
                    else { tk_x = t2_x; tk_y = t2_y; }

                    // Edge vectors in local frame
                    double x_ij_x = ti_x - tj_x;
                    double x_ij_y = ti_y - tj_y;
                    double x_ik_x = ti_x - tk_x;
                    double x_ik_y = ti_y - tk_y;

                    // rhs_rot = (weight_ij * Rf) * x_ij + (weight_ik * Rf) * x_ik
                    // (scalar * 2x2 matrix) * 2x1 vector
                    double rhs_rot_x = weight_ij * (rf00 * x_ij_x + rf01 * x_ij_y)
                                     + weight_ik * (rf00 * x_ik_x + rf01 * x_ik_y);
                    double rhs_rot_y = weight_ij * (rf10 * x_ij_x + rf11 * x_ij_y)
                                     + weight_ik * (rf10 * x_ik_x + rf11 * x_ik_y);

                    bu_private(local_i) += rhs_rot_x;
                    bv_private(local_i) += rhs_rot_y;

                    // Fixed vertex contributions
                    if (local_j == -1) {
                        vcg::Point2d fixed_pos_j = m.vert[global_j].T().P();
                        bu_private(local_i) += weight_ij * fixed_pos_j.X();
                        bv_private(local_i) += weight_ij * fixed_pos_j.Y();
                    }
                    if (local_k == -1) {
                        vcg::Point2d fixed_pos_k = m.vert[global_k].T().P();
                        bu_private(local_i) += weight_ik * fixed_pos_k.X();
                        bv_private(local_i) += weight_ik * fixed_pos_k.Y();
                    }
                }
            }
        }
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
    #pragma omp parallel for reduction(+:n, d)
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
    #pragma omp parallel for reduction(+:e, total_area)
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
    #pragma omp parallel for reduction(+:e, total_area)
    for (int fi = 0; fi < m.FN(); ++fi) {
        auto& f = m.face[fi];
        vcg::Point2d x10, x20;
        LocalIsometry(tsa[f].P[1] - tsa[f].P[0], tsa[f].P[2] - tsa[f].P[0], x10, x20);
        Eigen::Matrix2d Jf = ComputeTransformationMatrix(x10, x20, f.WT(1).P() - f.WT(0).P(), f.WT(2).P() - f.WT(0).P());
        Eigen::Matrix2d U, V;
        Eigen::Vector2d sigma;
        Eigen::JacobiSVD<Eigen::Matrix2d> svd;
        svd.compute(Jf, Eigen::ComputeFullU | Eigen::ComputeFullV);
        U = svd.matrixU(); V = svd.matrixV(); sigma = svd.singularValues();
        double area_f = 0.5 * ((tsa[f].P[1] - tsa[f].P[0]) ^ (tsa[f].P[2] - tsa[f].P[0])).Norm();
        total_area += area_f;
        e += area_f * (std::pow(sigma[0] - 1.0, 2.0) + std::pow(sigma[1] - 1.0, 2.0));
    }
    return e / total_area;
}

double ARAP::CurrentEnergySIMD()
{
    double e = 0;
    double total_area = 0;
    auto tsa = GetTargetShapeAttribute(m);
    const int nf = m.FN();

#ifdef ARAP_USE_AVX2
    constexpr int batch_size = 4;
    const int nf_aligned = (nf / batch_size) * batch_size;

    #pragma omp parallel reduction(+:e, total_area)
    {
        alignas(32) double jf_a[4], jf_b[4], jf_c[4], jf_d[4];
        alignas(32) double u00[4], u01[4], u10[4], u11[4];
        alignas(32) double v00[4], v01[4], v10[4], v11[4];
        alignas(32) double sigma0[4], sigma1[4];

        #pragma omp for nowait
        for (int fi = 0; fi < nf_aligned; fi += batch_size) {
            double batch_e = 0;
            double batch_area = 0;

            // Load and compute Jacobians for 4 faces
            for (int k = 0; k < batch_size; ++k) {
                int idx = fi + k;
                auto& f = m.face[idx];

                double x10_x = soa_local_coords.x1[idx];
                double x10_y = soa_local_coords.y1[idx];
                double x20_x = soa_local_coords.x2[idx];
                double x20_y = soa_local_coords.y2[idx];

                double u10_x = f.WT(1).P().X() - f.WT(0).P().X();
                double u10_y = f.WT(1).P().Y() - f.WT(0).P().Y();
                double u20_x = f.WT(2).P().X() - f.WT(0).P().X();
                double u20_y = f.WT(2).P().Y() - f.WT(0).P().Y();

                double det = x10_x * x20_y - x20_x * x10_y;
                if (std::abs(det) < 1e-300) det = 1e-300;
                double inv_det = 1.0 / det;

                double f_inv00 = x20_y * inv_det;
                double f_inv01 = -x20_x * inv_det;
                double f_inv10 = -x10_y * inv_det;
                double f_inv11 = x10_x * inv_det;

                jf_a[k] = u10_x * f_inv00 + u20_x * f_inv10;
                jf_b[k] = u10_x * f_inv01 + u20_x * f_inv11;
                jf_c[k] = u10_y * f_inv00 + u20_y * f_inv10;
                jf_d[k] = u10_y * f_inv01 + u20_y * f_inv11;

                // Compute face area
                double area_f = 0.5 * ((tsa[f].P[1] - tsa[f].P[0]) ^ (tsa[f].P[2] - tsa[f].P[0])).Norm();
                batch_area += area_f;

                // We'll compute energy after SVD
            }

            // Batched SVD
            arap_simd::batch_svd2x2_avx2(jf_a, jf_b, jf_c, jf_d,
                                          u00, u01, u10, u11,
                                          v00, v01, v10, v11,
                                          sigma0, sigma1);

            // Compute energy for each face in batch
            for (int k = 0; k < batch_size; ++k) {
                int idx = fi + k;
                auto& f = m.face[idx];
                double area_f = 0.5 * ((tsa[f].P[1] - tsa[f].P[0]) ^ (tsa[f].P[2] - tsa[f].P[0])).Norm();
                double s0_diff = sigma0[k] - 1.0;
                double s1_diff = sigma1[k] - 1.0;
                batch_e += area_f * (s0_diff * s0_diff + s1_diff * s1_diff);
            }

            e += batch_e;
            total_area += batch_area;
        }
    }

    // Handle remainder with scalar code
    for (int fi = nf_aligned; fi < nf; ++fi) {
        auto& f = m.face[fi];

        double x10_x = soa_local_coords.x1[fi];
        double x10_y = soa_local_coords.y1[fi];
        double x20_x = soa_local_coords.x2[fi];
        double x20_y = soa_local_coords.y2[fi];

        double u10_x = f.WT(1).P().X() - f.WT(0).P().X();
        double u10_y = f.WT(1).P().Y() - f.WT(0).P().Y();
        double u20_x = f.WT(2).P().X() - f.WT(0).P().X();
        double u20_y = f.WT(2).P().Y() - f.WT(0).P().Y();

        double det = x10_x * x20_y - x20_x * x10_y;
        if (std::abs(det) < 1e-300) det = 1e-300;
        double inv_det = 1.0 / det;

        double f_inv00 = x20_y * inv_det;
        double f_inv01 = -x20_x * inv_det;
        double f_inv10 = -x10_y * inv_det;
        double f_inv11 = x10_x * inv_det;

        double a = u10_x * f_inv00 + u20_x * f_inv10;
        double b = u10_x * f_inv01 + u20_x * f_inv11;
        double c = u10_y * f_inv00 + u20_y * f_inv10;
        double d = u10_y * f_inv01 + u20_y * f_inv11;

        double u00_s, u01_s, u10_s, u11_s;
        double v00_s, v01_s, v10_s, v11_s;
        double s0, s1;

        arap_simd::scalar::svd2x2(a, b, c, d,
                                   u00_s, u01_s, u10_s, u11_s,
                                   v00_s, v01_s, v10_s, v11_s,
                                   s0, s1);

        double area_f = 0.5 * ((tsa[f].P[1] - tsa[f].P[0]) ^ (tsa[f].P[2] - tsa[f].P[0])).Norm();
        total_area += area_f;
        double s0_diff = s0 - 1.0;
        double s1_diff = s1 - 1.0;
        e += area_f * (s0_diff * s0_diff + s1_diff * s1_diff);
    }
#else
    // Scalar fallback
    #pragma omp parallel for reduction(+:e, total_area)
    for (int fi = 0; fi < nf; ++fi) {
        auto& f = m.face[fi];

        double x10_x = soa_local_coords.x1[fi];
        double x10_y = soa_local_coords.y1[fi];
        double x20_x = soa_local_coords.x2[fi];
        double x20_y = soa_local_coords.y2[fi];

        double u10_x = f.WT(1).P().X() - f.WT(0).P().X();
        double u10_y = f.WT(1).P().Y() - f.WT(0).P().Y();
        double u20_x = f.WT(2).P().X() - f.WT(0).P().X();
        double u20_y = f.WT(2).P().Y() - f.WT(0).P().Y();

        double det = x10_x * x20_y - x20_x * x10_y;
        if (std::abs(det) < 1e-300) det = 1e-300;
        double inv_det = 1.0 / det;

        double f_inv00 = x20_y * inv_det;
        double f_inv01 = -x20_x * inv_det;
        double f_inv10 = -x10_y * inv_det;
        double f_inv11 = x10_x * inv_det;

        double a = u10_x * f_inv00 + u20_x * f_inv10;
        double b = u10_x * f_inv01 + u20_x * f_inv11;
        double c = u10_y * f_inv00 + u20_y * f_inv10;
        double d = u10_y * f_inv01 + u20_y * f_inv11;

        double u00_s, u01_s, u10_s, u11_s;
        double v00_s, v01_s, v10_s, v11_s;
        double s0, s1;

        arap_simd::scalar::svd2x2(a, b, c, d,
                                   u00_s, u01_s, u10_s, u11_s,
                                   v00_s, v01_s, v10_s, v11_s,
                                   s0, s1);

        double area_f = 0.5 * ((tsa[f].P[1] - tsa[f].P[0]) ^ (tsa[f].P[2] - tsa[f].P[0])).Norm();
        total_area += area_f;
        double s0_diff = s0 - 1.0;
        double s1_diff = s1 - 1.0;
        e += area_f * (s0_diff * s0_diff + s1_diff * s1_diff);
    }
#endif

    return e / total_area;
}

ARAPSolveInfo ARAP::Solve()
{
    ARAPSolveInfo si = {0, 0, 0, false};

#ifdef ARAP_ENABLE_TIMING
    si.timePrecompute_ms = 0;
    si.timeRotations_ms = 0;
    si.timeRHS_ms = 0;
    si.timeSolve_ms = 0;
    si.timeEnergy_ms = 0;
    si.timeTotal_ms = 0;
    ARAP_TIMER_START(total);
#endif

    std::vector<Cot> cotan = ComputeCotangentVector(m);

    ARAP_TIMER_START(precompute);
    PrecomputeData();
    ARAP_TIMER_END(precompute, si.timePrecompute_ms);

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

    ARAP_TIMER_START(init_energy);
    si.initialEnergy = CurrentEnergySIMD();
    ARAP_TIMER_END(init_energy, si.timeEnergy_ms);

    LOG_DEBUG << "ARAP: Starting energy is " << si.initialEnergy;

    double e = si.initialEnergy;
    bool converged = false;
    int iter = 0;

    Eigen::VectorXd bu, bv;

#ifdef ARAP_ENABLE_TIMING
    double total_rot_time = 0, total_rhs_time = 0, total_solve_time = 0, total_energy_time = si.timeEnergy_ms;
#endif

    while (!converged && iter < max_iter) {
        ARAP_TIMER_START(rot);
        ComputeRotationsSIMD(m);
        ARAP_TIMER_END(rot, si.timeRotations_ms);

        ARAP_TIMER_START(rhs);
        ComputeRHSSIMD(m, cotan, bu, bv);
        ARAP_TIMER_END(rhs, si.timeRHS_ms);

        ARAP_TIMER_START(solve);
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
        ARAP_TIMER_END(solve, si.timeSolve_ms);

        // 3. Unpack back to Mesh
        #pragma omp parallel for
        for (int local_i = 0; local_i < n_free_verts; ++local_i) {
            int global_i = local_to_global_idx[local_i];
            m.vert[global_i].T().P().X() = xu(local_i);
            m.vert[global_i].T().P().Y() = xv(local_i);
        }

        #pragma omp parallel for
        for (int fi = 0; fi < m.FN(); ++fi) {
            auto& f = m.face[fi];
            for (int i = 0; i < 3; ++i) {
                f.WT(i).P() = f.V(i)->T().P();
            }
        }

        ARAP_TIMER_START(energy);
        double e_curr = CurrentEnergySIMD();
        ARAP_TIMER_END(energy, si.timeEnergy_ms);

        si.finalEnergy = e_curr;

        double delta_e = e - e_curr;
        if (verbose && iter < 50) {
            LOG_INFO << "    [ARAP] Iteration " << iter << ": energy " << e_curr << " (delta " << delta_e << ")";
        }

        if (delta_e < 1e-8) {
            LOG_DEBUG << "ARAP: convergence reached (change in the energy value is too small)";
            converged = true;
        }

        e = e_curr;
        iter++;

#ifdef ARAP_ENABLE_TIMING
        total_rot_time += si.timeRotations_ms;
        total_rhs_time += si.timeRHS_ms;
        total_solve_time += si.timeSolve_ms;
        total_energy_time += si.timeEnergy_ms;
#endif
    }

    si.iterations = iter;

#ifdef ARAP_ENABLE_TIMING
    si.timeRotations_ms = total_rot_time;
    si.timeRHS_ms = total_rhs_time;
    si.timeSolve_ms = total_solve_time;
    si.timeEnergy_ms = total_energy_time;
#endif

    if (iter == max_iter) {
        LOG_DEBUG << "ARAP: iteration limit reached";
    }

    LOG_DEBUG << "ARAP: Energy after optimization is " << CurrentEnergySIMD() << " (" << iter << " iterations)";

    // Final update for wedges
    for (auto& f : m.face) {
        for (int i = 0; i < 3; ++i) {
            f.WT(i).P() = f.cV(i)->T().P();
        }
    }

#ifdef ARAP_ENABLE_TIMING
    ARAP_TIMER_END(total, si.timeTotal_ms);

    globalStats.total_time_ms += si.timeTotal_ms;
    globalStats.precompute_ms += si.timePrecompute_ms;
    globalStats.rotations_ms += si.timeRotations_ms;
    globalStats.rhs_ms += si.timeRHS_ms;
    globalStats.solve_ms += si.timeSolve_ms;
    globalStats.energy_ms += si.timeEnergy_ms;
    globalStats.total_iterations += si.iterations;
    globalStats.call_count++;

    if (n_free_verts < 100) globalStats.count_small++;
    else if (n_free_verts < 1000) globalStats.count_medium++;
    else globalStats.count_large++;
#endif

    return si;
}

void ARAP::PrecomputeData()
{
    const int nf = m.FN();
    local_frame_coords.resize(nf);

    // Allocate SoA structures
    soa_local_coords.resize(nf);
    soa_rotations.resize(nf);

    auto tsa = GetTargetShapeAttribute(m);
    #pragma omp parallel for
    for (int fi = 0; fi < nf; ++fi) {
        auto& f = m.face[fi];
        Eigen::Vector2d x_10, x_20;
        LocalIsometry(tsa[f].P[1] - tsa[f].P[0], tsa[f].P[2] - tsa[f].P[0], x_10, x_20);

        // Store in AoS format (for legacy code compatibility)
        local_frame_coords[fi][0] = Eigen::Vector2d::Zero();
        local_frame_coords[fi][1] = x_10;
        local_frame_coords[fi][2] = x_20;

        // Store in SoA format (for SIMD code)
        soa_local_coords.x1[fi] = x_10[0];
        soa_local_coords.y1[fi] = x_10[1];
        soa_local_coords.x2[fi] = x_20[0];
        soa_local_coords.y2[fi] = x_20[1];
    }
}