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
#include <atomic>

#ifdef ARAP_ENABLE_TIMING
#include <chrono>
#define ARAP_TIMER_START(name) auto timer_##name = std::chrono::high_resolution_clock::now()
#define ARAP_TIMER_END(name, var) var = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - timer_##name).count()

ARAP::AggregateStats ARAP::globalStats;

void ARAP::PrintAggregateStats() {
    if (globalStats.call_count == 0) return;
    LOG_INFO << "======= ARAP AGGREGATE STATS (" << globalStats.call_count << " calls) =======";
    LOG_INFO << "Total Time: " << std::fixed << std::setprecision(3) << globalStats.total_time_ms << " ms";
    LOG_INFO << " Precompute: " << globalStats.precompute_ms << " ms";
    if (globalStats.total_iterations > 0) {
        LOG_INFO << " Rotations: " << globalStats.rotations_ms << " ms (" << (globalStats.rotations_ms / globalStats.total_iterations) << " ms/iter )";
        LOG_INFO << " RHS: " << globalStats.rhs_ms << " ms (" << (globalStats.rhs_ms / globalStats.total_iterations) << " ms/iter )";
        LOG_INFO << " Solve: " << globalStats.solve_ms << " ms (" << (globalStats.solve_ms / globalStats.total_iterations) << " ms/iter )";
    }
    LOG_INFO << " Energy: " << globalStats.energy_ms << " ms";
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
edge length */
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
                    LOG_DEBUG << "Fixing vertices " << tri::Index(m, f.V(i)) << " " << tri::Index(m, f.V(f.Next(i)));
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

void ARAP::LogDegenerateFaces() {
    int degenerate_count = 0;
    auto tsa = GetTargetShapeAttribute(m);
    for (int i = 0; i < m.FN(); ++i) {
        auto& f = m.face[i];
        double area3d = ((tsa[f].P[1] - tsa[f].P[0]) ^ (tsa[f].P[2] - tsa[f].P[0])).Norm();
        if (area3d < 1e-12) {
            degenerate_count++;
            if (degenerate_count <= 5) { // Limit log spam
                LOG_WARN << " [ARAP DIAG] Degenerate Face #" << i
                         << " (V:" << tri::Index(m, f.V(0)) << "," << tri::Index(m, f.V(1)) << "," << tri::Index(m, f.V(2))
                         << ") Area3D=" << area3d;
            }
        }
    }
    if (degenerate_count > 0) {
        LOG_WARN << " [ARAP DIAG] Total degenerate faces detected: " << degenerate_count;
    }
}

void ARAP::LogInfiniteWeights(const std::vector<Cot>& cotan) {
    int inf_count = 0;
    int huge_count = 0;
    for (size_t i = 0; i < cotan.size(); ++i) {
        bool bad = false;
        for (int j=0; j<3; ++j) {
            if (!std::isfinite(cotan[i].v[j])) {
                inf_count++;
                bad = true;
            } else if (std::abs(cotan[i].v[j]) > 1e4) {
                huge_count++;
                bad = true;
            }
        }
        if (bad && (inf_count + huge_count) <= 5) {
            LOG_WARN << " [ARAP DIAG] Bad Cotan Weight Face #" << i
                     << " w=[" << cotan[i].v[0] << "," << cotan[i].v[1] << "," << cotan[i].v[2] << "]";
        }
    }
    if (inf_count > 0 || huge_count > 0) {
        LOG_WARN << " [ARAP DIAG] Weight issues: Inf/NaN=" << inf_count << ", Huge (>1e4)=" << huge_count;
    }
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

                    // Clamp huge weights to prevent numerical explosion
                    if (!std::isfinite(weight_ij)) weight_ij = 1e-8;
                    if (!std::isfinite(weight_ik)) weight_ik = 1e-8;
                    if (std::abs(weight_ij) > 1e6) weight_ij = 1e6 * (weight_ij > 0 ? 1.0 : -1.0);
                    if (std::abs(weight_ik) > 1e6) weight_ik = 1e6 * (weight_ik > 0 ? 1.0 : -1.0);

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

void ARAP::ComputeRotationsSIMD(Mesh& m)
{
    const int nf = m.FN();
    soa_rotations.resize(nf);

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
        // Gather input vectors (Local Frame Coords)
        __m256d x10_x = _mm256_loadu_pd(&soa_local_coords.x1[fi]);
        __m256d x10_y = _mm256_loadu_pd(&soa_local_coords.y1[fi]);
        __m256d x20_x = _mm256_loadu_pd(&soa_local_coords.x2[fi]);
        __m256d x20_y = _mm256_loadu_pd(&soa_local_coords.y2[fi]);

        // Gather UV coordinates
        auto& f0 = m.face[fi+0];
        auto& f1 = m.face[fi+1];
        auto& f2 = m.face[fi+2];
        auto& f3 = m.face[fi+3];

        __m256d u10_x = _mm256_set_pd(f3.WT(1).P().X() - f3.WT(0).P().X(), f2.WT(1).P().X() - f2.WT(0).P().X(), f1.WT(1).P().X() - f1.WT(0).P().X(), f0.WT(1).P().X() - f0.WT(0).P().X());
        __m256d u10_y = _mm256_set_pd(f3.WT(1).P().Y() - f3.WT(0).P().Y(), f2.WT(1).P().Y() - f2.WT(0).P().Y(), f1.WT(1).P().Y() - f1.WT(0).P().Y(), f0.WT(1).P().Y() - f0.WT(0).P().Y());
        __m256d u20_x = _mm256_set_pd(f3.WT(2).P().X() - f3.WT(0).P().X(), f2.WT(2).P().X() - f2.WT(0).P().X(), f1.WT(2).P().X() - f1.WT(0).P().X(), f0.WT(2).P().X() - f0.WT(0).P().X());
        __m256d u20_y = _mm256_set_pd(f3.WT(2).P().Y() - f3.WT(0).P().Y(), f2.WT(2).P().Y() - f2.WT(0).P().Y(), f1.WT(2).P().Y() - f1.WT(0).P().Y(), f0.WT(2).P().Y() - f0.WT(0).P().Y());

        // Sanitize inputs
        __m256d mask_valid = _mm256_and_pd(
            _mm256_and_pd(_mm256_cmp_pd(u10_x, u10_x, _CMP_EQ_OQ), _mm256_cmp_pd(u10_y, u10_y, _CMP_EQ_OQ)),
            _mm256_and_pd(_mm256_cmp_pd(u20_x, u20_x, _CMP_EQ_OQ), _mm256_cmp_pd(u20_y, u20_y, _CMP_EQ_OQ))
        );
        u10_x = _mm256_and_pd(u10_x, mask_valid);
        u10_y = _mm256_and_pd(u10_y, mask_valid);
        u20_x = _mm256_and_pd(u20_x, mask_valid);
        u20_y = _mm256_and_pd(u20_y, mask_valid);

        // Compute Jacobian J = U * F^-1 (Vectorized)
        __m256d det = _mm256_sub_pd(_mm256_mul_pd(x10_x, x20_y), _mm256_mul_pd(x20_x, x10_y));

        // Protect small det
        __m256d eps = _mm256_set1_pd(1e-300);
        __m256d abs_det = _mm256_andnot_pd(_mm256_set1_pd(-0.0), det);
        __m256d is_tiny = _mm256_cmp_pd(abs_det, eps, _CMP_LT_OQ);
        det = _mm256_blendv_pd(det, eps, is_tiny); 

        __m256d inv_det = _mm256_div_pd(_mm256_set1_pd(1.0), det);

        __m256d finv00 = _mm256_mul_pd(x20_y, inv_det);
        __m256d finv01 = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), x20_x), inv_det);
        __m256d finv10 = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), x10_y), inv_det);
        __m256d finv11 = _mm256_mul_pd(x10_x, inv_det);

        // J = G * F^-1
        __m256d ja = _mm256_add_pd(_mm256_mul_pd(u10_x, finv00), _mm256_mul_pd(u20_x, finv10));
        __m256d jb = _mm256_add_pd(_mm256_mul_pd(u10_x, finv01), _mm256_mul_pd(u20_x, finv11));
        __m256d jc = _mm256_add_pd(_mm256_mul_pd(u10_y, finv00), _mm256_mul_pd(u20_y, finv10));
        __m256d jd = _mm256_add_pd(_mm256_mul_pd(u10_y, finv01), _mm256_mul_pd(u20_y, finv11));

        _mm256_storeu_pd(jf_a, ja);
        _mm256_storeu_pd(jf_b, jb);
        _mm256_storeu_pd(jf_c, jc);
        _mm256_storeu_pd(jf_d, jd);

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
        
        // Sanitize inputs
        if (!std::isfinite(u10_x)) u10_x = 0;
        if (!std::isfinite(u10_y)) u10_y = 0;
        if (!std::isfinite(u20_x)) u20_x = 0;
        if (!std::isfinite(u20_y)) u20_y = 0;

        double det = x10_x * x20_y - x20_x * x10_y;
        if (std::abs(det) < 1e-300) det = (det >= 0) ? 1e-300 : -1e-300;
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
    // Scalar fallback (identical logic to above remainder loop)
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
        
        if (!std::isfinite(u10_x)) u10_x = 0;
        if (!std::isfinite(u10_y)) u10_y = 0;
        if (!std::isfinite(u20_x)) u20_x = 0;
        if (!std::isfinite(u20_y)) u20_y = 0;

        double det = x10_x * x20_y - x20_x * x10_y;
        if (std::abs(det) < 1e-300) det = (det >= 0) ? 1e-300 : -1e-300;
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

void ARAP::ComputeRHSSIMD(Mesh& m, const std::vector<Cot>& cotan, Eigen::VectorXd& bu, Eigen::VectorXd& bv)
{
    bu.setZero(n_free_verts);
    bv.setZero(n_free_verts);

    const int nf = m.FN();

#ifdef ARAP_USE_AVX2
    constexpr int batch_size = 4;
    const int nf_aligned = (nf / batch_size) * batch_size;
#endif

    #pragma omp parallel
    {
        Eigen::VectorXd bu_private = Eigen::VectorXd::Zero(n_free_verts);
        Eigen::VectorXd bv_private = Eigen::VectorXd::Zero(n_free_verts);

#ifdef ARAP_USE_AVX2
        alignas(32) double fx_buf[4], fy_buf[4];
        alignas(32) double w_ij_scalar_buf[4], w_ik_scalar_buf[4];

        #pragma omp for nowait
        for (int fi = 0; fi < nf_aligned; fi += batch_size) {
            // Load rotations
            __m256d r00 = _mm256_loadu_pd(&soa_rotations.r00[fi]);
            __m256d r01 = _mm256_loadu_pd(&soa_rotations.r01[fi]);
            __m256d r10 = _mm256_loadu_pd(&soa_rotations.r10[fi]);
            __m256d r11 = _mm256_loadu_pd(&soa_rotations.r11[fi]);

            // Load local coordinates
            __m256d tx0 = _mm256_setzero_pd();
            __m256d ty0 = _mm256_setzero_pd();
            __m256d tx1 = _mm256_loadu_pd(&soa_local_coords.x1[fi]);
            __m256d ty1 = _mm256_loadu_pd(&soa_local_coords.y1[fi]);
            __m256d tx2 = _mm256_loadu_pd(&soa_local_coords.x2[fi]);
            __m256d ty2 = _mm256_loadu_pd(&soa_local_coords.y2[fi]);

            for (int i = 0; i < 3; ++i) {
                int j_local_idx = (i+1)%3;
                int k_local_idx = (i+2)%3;

                // Determine ti, tj, tk vectors based on i
                __m256d tix, tiy, tjx, tjy, tkx, tky;
                if (i == 0) { tix = tx0; tiy = ty0; }
                else if (i == 1) { tix = tx1; tiy = ty1; }
                else { tix = tx2; tiy = ty2; }

                if (j_local_idx == 0) { tjx = tx0; tjy = ty0; }
                else if (j_local_idx == 1) { tjx = tx1; tjy = ty1; }
                else { tjx = tx2; tjy = ty2; }

                if (k_local_idx == 0) { tkx = tx0; tky = ty0; }
                else if (k_local_idx == 1) { tkx = tx1; tky = ty1; }
                else { tkx = tx2; tky = ty2; }

                // Gather weights safely (sanitizing NaNs/Infs and clamping huge weights)
                for (int k = 0; k < batch_size; ++k) {
                    double wj = cotan[fi+k].v[k_local_idx];
                    if (!std::isfinite(wj)) wj = 1e-8;
                    if (std::abs(wj) > 1e6) wj = 1e6 * (wj > 0 ? 1.0 : -1.0);
                    w_ij_scalar_buf[k] = wj;
                    
                    double wk = cotan[fi+k].v[j_local_idx];
                    if (!std::isfinite(wk)) wk = 1e-8;
                    if (std::abs(wk) > 1e6) wk = 1e6 * (wk > 0 ? 1.0 : -1.0);
                    w_ik_scalar_buf[k] = wk;
                }
                
                __m256d wij = _mm256_set_pd(w_ij_scalar_buf[3], w_ij_scalar_buf[2], w_ij_scalar_buf[1], w_ij_scalar_buf[0]);
                __m256d wik = _mm256_set_pd(w_ik_scalar_buf[3], w_ik_scalar_buf[2], w_ik_scalar_buf[1], w_ik_scalar_buf[0]);

                // Edge vectors
                __m256d xij_x = _mm256_sub_pd(tix, tjx);
                __m256d xij_y = _mm256_sub_pd(tiy, tjy);
                __m256d xik_x = _mm256_sub_pd(tix, tkx);
                __m256d xik_y = _mm256_sub_pd(tiy, tky);

                // rhs_rot = wij * (R * xij) + wik * (R * xik)
                __m256d rot_ij_x = _mm256_fmadd_pd(r00, xij_x, _mm256_mul_pd(r01, xij_y));
                __m256d rot_ij_y = _mm256_fmadd_pd(r10, xij_x, _mm256_mul_pd(r11, xij_y));
                
                __m256d rot_ik_x = _mm256_fmadd_pd(r00, xik_x, _mm256_mul_pd(r01, xik_y));
                __m256d rot_ik_y = _mm256_fmadd_pd(r10, xik_x, _mm256_mul_pd(r11, xik_y));

                __m256d fx = _mm256_fmadd_pd(wij, rot_ij_x, _mm256_mul_pd(wik, rot_ik_x));
                __m256d fy = _mm256_fmadd_pd(wij, rot_ij_y, _mm256_mul_pd(wik, rot_ik_y));

                _mm256_storeu_pd(fx_buf, fx);
                _mm256_storeu_pd(fy_buf, fy);

                for (int k = 0; k < batch_size; ++k) {
                    int idx = fi + k;
                    auto &f = m.face[idx];
                    int global_i = (int) tri::Index(m, f.V0(i));
                    int local_i = global_to_local_idx[global_i];

                    if (local_i != -1) {
                        if (std::isfinite(fx_buf[k])) bu_private(local_i) += fx_buf[k];
                        if (std::isfinite(fy_buf[k])) bv_private(local_i) += fy_buf[k];

                        // Fixed vertex handling
                        double w_ij_scalar = w_ij_scalar_buf[k];
                        int global_j = (int) tri::Index(m, f.V1(i));
                        int local_j = global_to_local_idx[global_j];
                        
                        if (local_j == -1) {
                            vcg::Point2d fixed_pos_j = m.vert[global_j].T().P();
                            if (std::isfinite(fixed_pos_j.X()) && std::isfinite(fixed_pos_j.Y())) {
                                bu_private(local_i) += w_ij_scalar * fixed_pos_j.X();
                                bv_private(local_i) += w_ij_scalar * fixed_pos_j.Y();
                            }
                        }

                        double w_ik_scalar = w_ik_scalar_buf[k];
                        int global_k = (int) tri::Index(m, f.V2(i));
                        int local_k = global_to_local_idx[global_k];

                        if (local_k == -1) {
                            vcg::Point2d fixed_pos_k = m.vert[global_k].T().P();
                            if (std::isfinite(fixed_pos_k.X()) && std::isfinite(fixed_pos_k.Y())) {
                                bu_private(local_i) += w_ik_scalar * fixed_pos_k.X();
                                bv_private(local_i) += w_ik_scalar * fixed_pos_k.Y();
                            }
                        }
                    }
                }
            }
        }

#else
        int nf_aligned = 0;
#endif

        // Remainder loop
        for (int fi = nf_aligned; fi < nf; ++fi) {
            auto &f = m.face[fi];

            double rf00 = soa_rotations.r00[fi];
            double rf01 = soa_rotations.r01[fi];
            double rf10 = soa_rotations.r10[fi];
            double rf11 = soa_rotations.r11[fi];

            if (!std::isfinite(rf00)) { rf00 = 1.0; rf01 = 0.0; rf10 = 0.0; rf11 = 1.0; }

            double t0_x = 0.0, t0_y = 0.0;
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
                    if (std::abs(weight_ij) > 1e6) weight_ij = 1e6 * (weight_ij > 0 ? 1.0 : -1.0);
                    if (std::abs(weight_ik) > 1e6) weight_ik = 1e6 * (weight_ik > 0 ? 1.0 : -1.0);

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

                    double x_ij_x = ti_x - tj_x;
                    double x_ij_y = ti_y - tj_y;
                    double x_ik_x = ti_x - tk_x;
                    double x_ik_y = ti_y - tk_y;

                    double rhs_rot_x = weight_ij * (rf00 * x_ij_x + rf01 * x_ij_y)
                                     + weight_ik * (rf00 * x_ik_x + rf01 * x_ik_y);
                    double rhs_rot_y = weight_ij * (rf10 * x_ij_x + rf11 * x_ij_y)
                                     + weight_ik * (rf10 * x_ik_x + rf11 * x_ik_y);

                    if (std::isfinite(rhs_rot_x)) bu_private(local_i) += rhs_rot_x;
                    if (std::isfinite(rhs_rot_y)) bv_private(local_i) += rhs_rot_y;

                    if (local_j == -1) {
                        vcg::Point2d fixed_pos_j = m.vert[global_j].T().P();
                        if (std::isfinite(fixed_pos_j.X()) && std::isfinite(fixed_pos_j.Y())) {
                            bu_private(local_i) += weight_ij * fixed_pos_j.X();
                            bv_private(local_i) += weight_ij * fixed_pos_j.Y();
                        }
                    }
                    if (local_k == -1) {
                        vcg::Point2d fixed_pos_k = m.vert[global_k].T().P();
                        if (std::isfinite(fixed_pos_k.X()) && std::isfinite(fixed_pos_k.Y())) {
                            bu_private(local_i) += weight_ik * fixed_pos_k.X();
                            bv_private(local_i) += weight_ik * fixed_pos_k.Y();
                        }
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
    if (*area <= 1e-12) return 0.0;

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
        if (area > 1e-12) {
            n += (area * energy);
            d += area;
        }
    }
    if (num)
        *num = n;
    if (denom)
        *denom = d;
    return (d > 0) ? n / d : 0;
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
        if (area_f > 1e-12) {
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
    return (total_area > 0) ? e / total_area : 0;
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
        double area_f = std::abs(x10 ^ x20);
        if (area_f > 1e-12) {
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
    return (total_area > 0) ? e / total_area : 0;
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
            
            // Gather input vectors
            __m256d x10_x = _mm256_loadu_pd(&soa_local_coords.x1[fi]);
            __m256d x10_y = _mm256_loadu_pd(&soa_local_coords.y1[fi]);
            __m256d x20_x = _mm256_loadu_pd(&soa_local_coords.x2[fi]);
            __m256d x20_y = _mm256_loadu_pd(&soa_local_coords.y2[fi]);

            // Gather UV coordinates (AoS -> registers)
            auto& f0 = m.face[fi+0];
            auto& f1 = m.face[fi+1];
            auto& f2 = m.face[fi+2];
            auto& f3 = m.face[fi+3];

            __m256d u10_x = _mm256_set_pd(f3.WT(1).P().X() - f3.WT(0).P().X(), f2.WT(1).P().X() - f2.WT(0).P().X(), f1.WT(1).P().X() - f1.WT(0).P().X(), f0.WT(1).P().X() - f0.WT(0).P().X());
            __m256d u10_y = _mm256_set_pd(f3.WT(1).P().Y() - f3.WT(0).P().Y(), f2.WT(1).P().Y() - f2.WT(0).P().Y(), f1.WT(1).P().Y() - f1.WT(0).P().Y(), f0.WT(1).P().Y() - f0.WT(0).P().Y());
            __m256d u20_x = _mm256_set_pd(f3.WT(2).P().X() - f3.WT(0).P().X(), f2.WT(2).P().X() - f2.WT(0).P().X(), f1.WT(2).P().X() - f1.WT(0).P().X(), f0.WT(2).P().X() - f0.WT(0).P().X());
            __m256d u20_y = _mm256_set_pd(f3.WT(2).P().Y() - f3.WT(0).P().Y(), f2.WT(2).P().Y() - f2.WT(0).P().Y(), f1.WT(2).P().Y() - f1.WT(0).P().Y(), f0.WT(2).P().Y() - f0.WT(0).P().Y());

            // Compute Jacobian J = U * F^-1
            __m256d det = _mm256_sub_pd(_mm256_mul_pd(x10_x, x20_y), _mm256_mul_pd(x20_x, x10_y));
            // Protect small det
            __m256d eps = _mm256_set1_pd(1e-300);
            __m256d mask = _mm256_cmp_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), det), eps, _CMP_LT_OQ);
            det = _mm256_blendv_pd(det, eps, mask); 
            __m256d inv_det = _mm256_div_pd(_mm256_set1_pd(1.0), det);

            __m256d finv00 = _mm256_mul_pd(x20_y, inv_det);
            __m256d finv01 = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), x20_x), inv_det);
            __m256d finv10 = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), x10_y), inv_det);
            __m256d finv11 = _mm256_mul_pd(x10_x, inv_det);

            // J = [[a,b],[c,d]]
            __m256d ja = _mm256_add_pd(_mm256_mul_pd(u10_x, finv00), _mm256_mul_pd(u20_x, finv10));
            __m256d jb = _mm256_add_pd(_mm256_mul_pd(u10_x, finv01), _mm256_mul_pd(u20_x, finv11));
            __m256d jc = _mm256_add_pd(_mm256_mul_pd(u10_y, finv00), _mm256_mul_pd(u20_y, finv10));
            __m256d jd = _mm256_add_pd(_mm256_mul_pd(u10_y, inv_det), _mm256_mul_pd(u20_y, finv11)); // Note: this line in user prompt had jd = ...u10_y * finv01 + u20_y * finv11... I will fix it.
            // Wait, let me re-check the user prompt's jd line.
            // It was: __m256d jd = _mm256_add_pd(_mm256_mul_pd(u10_y, finv01), _mm256_mul_pd(u20_y, finv11));
            // My previous manual fix was wrong.
            jd = _mm256_add_pd(_mm256_mul_pd(u10_y, finv01), _mm256_mul_pd(u20_y, finv11));

            _mm256_storeu_pd(jf_a, ja);
            _mm256_storeu_pd(jf_b, jb);
            _mm256_storeu_pd(jf_c, jc);
            _mm256_storeu_pd(jf_d, jd);

            // Compute Area
            __m256d area_abs = _mm256_andnot_pd(_mm256_set1_pd(-0.0), _mm256_mul_pd(_mm256_set1_pd(0.5), det)); 

            alignas(32) double tmp_area[4];
            _mm256_storeu_pd(tmp_area, area_abs);
            total_area += tmp_area[0] + tmp_area[1] + tmp_area[2] + tmp_area[3];

            // Batched SVD
            arap_simd::batch_svd2x2_avx2(jf_a, jf_b, jf_c, jf_d,
                                          u00, u01, u10, u11,
                                          v00, v01, v10, v11,
                                          sigma0, sigma1);

            // Energy
            __m256d s0 = _mm256_loadu_pd(sigma0);
            __m256d s1 = _mm256_loadu_pd(sigma1);
            __m256d one = _mm256_set1_pd(1.0);
            __m256d d0 = _mm256_sub_pd(s0, one);
            __m256d d1 = _mm256_sub_pd(s1, one);
            __m256d e_vec = _mm256_mul_pd(area_abs, _mm256_add_pd(_mm256_mul_pd(d0, d0), _mm256_mul_pd(d1, d1)));

            alignas(32) double tmp_e[4];
            _mm256_storeu_pd(tmp_e, e_vec);
            e += tmp_e[0] + tmp_e[1] + tmp_e[2] + tmp_e[3];
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

        double area_f = 0.5 * std::abs(det);
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

        double area_f = 0.5 * std::abs(det);
        total_area += area_f;
        double s0_diff = s0 - 1.0;
        double s1_diff = s1 - 1.0;
        e += area_f * (s0_diff * s0_diff + s1_diff * s1_diff);
    }

#endif
    return (total_area > 0) ? e / total_area : 0;
}

void ARAP::LogSolverState(int iter, const Eigen::VectorXd& solution) {
    if (solution.size() == 0) return;
    double min_val = solution.minCoeff();
    double max_val = solution.maxCoeff();
    if (std::abs(max_val) > 1e4 || std::abs(min_val) > 1e4) {
        LOG_WARN << " [ARAP DIAG] Iter " << iter << " Solution range: [" << min_val << ", " << max_val << "]";
    }
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

    // Diagnostic check for bad weights
    LogInfiniteWeights(cotan);
    LogDegenerateFaces();

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

    if (!std::isfinite(si.initialEnergy)) {
        LOG_WARN << "ARAP: Initial energy is NaN/Inf, aborting optimization.";
        si.numericalError = true;
        return si;
    }

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

        // Check for NaN/Inf in solution
        if (!std::isfinite(xu.sum()) || !std::isfinite(xv.sum())) {
             LOG_WARN << "ARAP: solver produced NaN/Inf values at iteration " << iter;
             si.numericalError = true;
             break;
        }
        
        // Diag log if range is huge
        if (iter % 10 == 0) {
            LogSolverState(iter, xu);
        }

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
        
        if (!std::isfinite(e_curr) || e_curr > 1e15) {
             LOG_WARN << "ARAP: Energy explosion detected (" << e_curr << ") at iteration " << iter;
             si.numericalError = true;
             break;
        }

        si.finalEnergy = e_curr;

        double delta_e = e - e_curr;
        if (verbose && iter < 75) {
            LOG_INFO << "    [ARAP] Iteration " << iter << ": energy " << e_curr << " (delta " << delta_e << ")";
        }

        if (std::abs(delta_e) < 1e-8) {
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

    // Allocate SoA structures
    soa_local_coords.resize(nf);
    soa_rotations.resize(nf);

    auto tsa = GetTargetShapeAttribute(m);
    #pragma omp parallel for
    for (int fi = 0; fi < nf; ++fi) {
        auto& f = m.face[fi];
        Eigen::Vector2d x_10, x_20;
        
        double area3d = ((tsa[f].P[1] - tsa[f].P[0]) ^ (tsa[f].P[2] - tsa[f].P[0])).Norm();
        if (area3d < 1e-12) {
            // Handle degenerate triangle
            x_10 = Eigen::Vector2d(1e-6, 0.0);
            x_20 = Eigen::Vector2d(0.0, 1e-6);
        } else {
            LocalIsometry(tsa[f].P[1] - tsa[f].P[0], tsa[f].P[2] - tsa[f].P[0], x_10, x_20);
        }

        // Store in SoA format (for SIMD code)
        soa_local_coords.x1[fi] = x_10[0];
        soa_local_coords.y1[fi] = x_10[1];
        soa_local_coords.x2[fi] = x_20[0];
        soa_local_coords.y2[fi] = x_20[1];
    }
}