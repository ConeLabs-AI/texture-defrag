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

#ifndef ARAP_SIMD_H
#define ARAP_SIMD_H

#include <cmath>
#include <algorithm>

// Platform detection
// AVX2 and FMA are required for the optimized path
#if defined(__AVX2__) && defined(__FMA__) && !defined(ARAP_DISABLE_SIMD)
    #define ARAP_USE_AVX2 1
    #include <immintrin.h>
#elif defined(_MSC_VER) && defined(__AVX2__) && !defined(ARAP_DISABLE_SIMD)
    // MSVC doesn't always define __FMA__ even when /arch:AVX2 is present
    #define ARAP_USE_AVX2 1
    #include <immintrin.h>
#endif

#ifdef ARAP_USE_AVX2
    #define ARAP_SIMD_WIDTH 4
#else
    #define ARAP_SIMD_WIDTH 1
#endif

namespace arap_simd {

#ifdef ARAP_USE_AVX2

// Vectorized atan2 approximation for AVX2
// Uses a polynomial approximation accurate to ~1e-7
inline __m256d atan2_avx2(__m256d y, __m256d x) {
    // Constants
    const __m256d pi = _mm256_set1_pd(3.14159265358979323846);
    const __m256d pi_2 = _mm256_set1_pd(1.57079632679489661923);
    const __m256d zero = _mm256_setzero_pd();
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d neg_one = _mm256_set1_pd(-1.0);

    // Polynomial coefficients for atan(x) on [-1,1]
    const __m256d c1 = _mm256_set1_pd(0.9998660);
    const __m256d c3 = _mm256_set1_pd(-0.3302995);
    const __m256d c5 = _mm256_set1_pd(0.1801410);
    const __m256d c7 = _mm256_set1_pd(-0.0851330);
    const __m256d c9 = _mm256_set1_pd(0.0208351);

    __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
    __m256d abs_y = _mm256_andnot_pd(_mm256_set1_pd(-0.0), y);

    // Compute atan(y/x) or atan(x/y) depending on which is smaller
    __m256d swap_mask = _mm256_cmp_pd(abs_y, abs_x, _CMP_GT_OQ);
    __m256d min_val = _mm256_blendv_pd(abs_y, abs_x, swap_mask);
    __m256d max_val = _mm256_blendv_pd(abs_x, abs_y, swap_mask);

    // Avoid division by zero
    max_val = _mm256_max_pd(max_val, _mm256_set1_pd(1e-300));

    __m256d t = _mm256_div_pd(min_val, max_val);
    __m256d t2 = _mm256_mul_pd(t, t);

    // Polynomial evaluation: atan(t) = t * (c1 + t^2*(c3 + t^2*(c5 + t^2*(c7 + t^2*c9))))
    __m256d result = c9;
    result = _mm256_fmadd_pd(result, t2, c7);
    result = _mm256_fmadd_pd(result, t2, c5);
    result = _mm256_fmadd_pd(result, t2, c3);
    result = _mm256_fmadd_pd(result, t2, c1);
    result = _mm256_mul_pd(result, t);

    // If we swapped, result = pi/2 - result
    result = _mm256_blendv_pd(result, _mm256_sub_pd(pi_2, result), swap_mask);

    // Handle quadrants based on signs of x and y
    __m256d x_neg = _mm256_cmp_pd(x, zero, _CMP_LT_OQ);
    __m256d y_neg = _mm256_cmp_pd(y, zero, _CMP_LT_OQ);

    // If x < 0, add or subtract pi
    __m256d adjust = _mm256_blendv_pd(pi, _mm256_sub_pd(zero, pi), y_neg);
    result = _mm256_blendv_pd(result, _mm256_add_pd(result, adjust), x_neg);

    // If y < 0 and x >= 0, negate result
    __m256d negate_mask = _mm256_andnot_pd(x_neg, y_neg);
    result = _mm256_blendv_pd(result, _mm256_sub_pd(zero, result), negate_mask);

    return result;
}

// Vectorized sin/cos using polynomial approximation
inline void sincos_avx2(__m256d x, __m256d& sin_out, __m256d& cos_out) {
    // Range reduction to [-pi, pi]
    const __m256d pi = _mm256_set1_pd(3.14159265358979323846);
    const __m256d two_pi = _mm256_set1_pd(6.28318530717958647692);
    const __m256d inv_two_pi = _mm256_set1_pd(0.15915494309189533577);

    // Reduce to [-pi, pi]
    __m256d n = _mm256_round_pd(_mm256_mul_pd(x, inv_two_pi), _MM_FROUND_TO_NEAREST_INT);
    x = _mm256_fnmadd_pd(n, two_pi, x);

    __m256d x2 = _mm256_mul_pd(x, x);

    // Sin polynomial: x - x^3/6 + x^5/120 - x^7/5040 + x^9/362880
    const __m256d s1 = _mm256_set1_pd(1.0);
    const __m256d s3 = _mm256_set1_pd(-1.0/6.0);
    const __m256d s5 = _mm256_set1_pd(1.0/120.0);
    const __m256d s7 = _mm256_set1_pd(-1.0/5040.0);
    const __m256d s9 = _mm256_set1_pd(1.0/362880.0);

    __m256d sin_val = s9;
    sin_val = _mm256_fmadd_pd(sin_val, x2, s7);
    sin_val = _mm256_fmadd_pd(sin_val, x2, s5);
    sin_val = _mm256_fmadd_pd(sin_val, x2, s3);
    sin_val = _mm256_fmadd_pd(sin_val, x2, s1);
    sin_out = _mm256_mul_pd(sin_val, x);

    // Cos polynomial: 1 - x^2/2 + x^4/24 - x^6/720 + x^8/40320
    const __m256d c0 = _mm256_set1_pd(1.0);
    const __m256d c2 = _mm256_set1_pd(-1.0/2.0);
    const __m256d c4 = _mm256_set1_pd(1.0/24.0);
    const __m256d c6 = _mm256_set1_pd(-1.0/720.0);
    const __m256d c8 = _mm256_set1_pd(1.0/40320.0);

    __m256d cos_val = c8;
    cos_val = _mm256_fmadd_pd(cos_val, x2, c6);
    cos_val = _mm256_fmadd_pd(cos_val, x2, c4);
    cos_val = _mm256_fmadd_pd(cos_val, x2, c2);
    cos_out = _mm256_fmadd_pd(cos_val, x2, c0);
}

// Batched 2x2 SVD for 4 matrices using AVX2
// Input: 4 matrices in SoA format (a[4], b[4], c[4], d[4]) for [[a,b],[c,d]]
// Output: U (u00,u01,u10,u11), V (v00,v01,v10,v11), sigma (s0, s1)
inline void batch_svd2x2_avx2(
    const double* a, const double* b, const double* c, const double* d,
    double* u00, double* u01, double* u10, double* u11,
    double* v00, double* v01, double* v10, double* v11,
    double* sigma0, double* sigma1)
{
    // Load 4 matrices
    __m256d va = _mm256_loadu_pd(a);
    __m256d vb = _mm256_loadu_pd(b);
    __m256d vc = _mm256_loadu_pd(c);
    __m256d vd = _mm256_loadu_pd(d);

    // Closed-form 2x2 SVD
    // E = (a+d)/2, F = (a-d)/2, G = (b+c)/2, H = (b-c)/2
    __m256d half = _mm256_set1_pd(0.5);
    __m256d E = _mm256_mul_pd(_mm256_add_pd(va, vd), half);
    __m256d F = _mm256_mul_pd(_mm256_sub_pd(va, vd), half);
    __m256d G = _mm256_mul_pd(_mm256_add_pd(vb, vc), half);
    __m256d H = _mm256_mul_pd(_mm256_sub_pd(vb, vc), half);

    // Q = sqrt(E^2 + H^2), R = sqrt(F^2 + G^2)
    __m256d E2 = _mm256_mul_pd(E, E);
    __m256d H2 = _mm256_mul_pd(H, H);
    __m256d F2 = _mm256_mul_pd(F, F);
    __m256d G2 = _mm256_mul_pd(G, G);

    __m256d Q = _mm256_sqrt_pd(_mm256_add_pd(E2, H2));
    __m256d R = _mm256_sqrt_pd(_mm256_add_pd(F2, G2));

    // Singular values: s1 = Q + R, s2 = |Q - R|
    __m256d s1 = _mm256_add_pd(Q, R);
    __m256d s2_signed = _mm256_sub_pd(Q, R);
    __m256d s2 = _mm256_andnot_pd(_mm256_set1_pd(-0.0), s2_signed);  // abs

    _mm256_storeu_pd(sigma0, s1);
    _mm256_storeu_pd(sigma1, s2);

    // Angles: alpha = atan2(H, E), beta = atan2(G, F)
    // phi = (beta - alpha) / 2, theta = (beta + alpha) / 2
    __m256d alpha = atan2_avx2(H, E);
    __m256d beta  = atan2_avx2(G, F);
    __m256d phi   = _mm256_mul_pd(_mm256_sub_pd(beta, alpha), half);
    __m256d theta = _mm256_mul_pd(_mm256_add_pd(beta, alpha), half);

    // U = [[cos(phi), -sin(phi)], [sin(phi), cos(phi)]] if s2 >= 0
    // else flip sign of second column
    __m256d sin_phi, cos_phi;
    sincos_avx2(phi, sin_phi, cos_phi);

    __m256d sin_theta, cos_theta;
    sincos_avx2(theta, sin_theta, cos_theta);

    // Check if we need to flip (when Q - R < 0)
    __m256d flip_mask = _mm256_cmp_pd(s2_signed, _mm256_setzero_pd(), _CMP_LT_OQ);
    __m256d neg_one = _mm256_set1_pd(-1.0);
    __m256d flip_sign = _mm256_blendv_pd(_mm256_set1_pd(1.0), neg_one, flip_mask);

    // U matrix
    _mm256_storeu_pd(u00, cos_phi);
    _mm256_storeu_pd(u01, _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), sin_phi), flip_sign));
    _mm256_storeu_pd(u10, sin_phi);
    _mm256_storeu_pd(u11, _mm256_mul_pd(cos_phi, flip_sign));

    // V matrix: [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
    _mm256_storeu_pd(v00, cos_theta);
    _mm256_storeu_pd(v01, _mm256_sub_pd(_mm256_setzero_pd(), sin_theta));
    _mm256_storeu_pd(v10, sin_theta);
    _mm256_storeu_pd(v11, cos_theta);
}

// Compute rotation R = U * V^T for 4 matrices
// Returns rotations ensuring det(R) > 0
inline void batch_rotation_from_svd_avx2(
    const double* u00, const double* u01, const double* u10, const double* u11,
    const double* v00, const double* v01, const double* v10, const double* v11,
    double* r00, double* r01, double* r10, double* r11)
{
    __m256d vu00 = _mm256_loadu_pd(u00);
    __m256d vu01 = _mm256_loadu_pd(u01);
    __m256d vu10 = _mm256_loadu_pd(u10);
    __m256d vu11 = _mm256_loadu_pd(u11);

    __m256d vv00 = _mm256_loadu_pd(v00);
    __m256d vv01 = _mm256_loadu_pd(v01);
    __m256d vv10 = _mm256_loadu_pd(v10);
    __m256d vv11 = _mm256_loadu_pd(v11);

    // If det(U) < 0, flip U's second column (this matches Eigen's fix)
    __m256d detU = _mm256_fmsub_pd(vu00, vu11, _mm256_mul_pd(vu01, vu10));
    __m256d neg_mask = _mm256_cmp_pd(detU, _mm256_setzero_pd(), _CMP_LT_OQ);
    __m256d flip = _mm256_blendv_pd(_mm256_set1_pd(1.0), _mm256_set1_pd(-1.0), neg_mask);

    vu01 = _mm256_mul_pd(vu01, flip);
    vu11 = _mm256_mul_pd(vu11, flip);

    // R = U * V^T
    // r00 = u00*v00 + u01*v01, r01 = u00*v10 + u01*v11
    // r10 = u10*v00 + u11*v01, r11 = u10*v10 + u11*v11
    __m256d vr00 = _mm256_fmadd_pd(vu00, vv00, _mm256_mul_pd(vu01, vv01));
    __m256d vr01 = _mm256_fmadd_pd(vu00, vv10, _mm256_mul_pd(vu01, vv11));
    __m256d vr10 = _mm256_fmadd_pd(vu10, vv00, _mm256_mul_pd(vu11, vv01));
    __m256d vr11 = _mm256_fmadd_pd(vu10, vv10, _mm256_mul_pd(vu11, vv11));

    _mm256_storeu_pd(r00, vr00);
    _mm256_storeu_pd(r01, vr01);
    _mm256_storeu_pd(r10, vr10);
    _mm256_storeu_pd(r11, vr11);
}

// Batched 2x2 matrix inverse for 4 matrices
// Input: [[a,b],[c,d]], Output: (1/det) * [[d,-b],[-c,a]]
inline void batch_inverse2x2_avx2(
    const double* a, const double* b, const double* c, const double* d,
    double* inv_a, double* inv_b, double* inv_c, double* inv_d)
{
    __m256d va = _mm256_loadu_pd(a);
    __m256d vb = _mm256_loadu_pd(b);
    __m256d vc = _mm256_loadu_pd(c);
    __m256d vd = _mm256_loadu_pd(d);

    // det = a*d - b*c
    __m256d det = _mm256_fmsub_pd(va, vd, _mm256_mul_pd(vb, vc));

    // Avoid division by zero while preserving sign
    __m256d eps = _mm256_set1_pd(1e-300);
    __m256d neg_eps = _mm256_set1_pd(-1e-300);
    __m256d abs_det = _mm256_andnot_pd(_mm256_set1_pd(-0.0), det);
    __m256d is_tiny = _mm256_cmp_pd(abs_det, eps, _CMP_LT_OQ);
    __m256d sign_mask = _mm256_cmp_pd(det, _mm256_setzero_pd(), _CMP_LT_OQ);
    __m256d clamped_det = _mm256_blendv_pd(eps, neg_eps, sign_mask);
    det = _mm256_blendv_pd(det, clamped_det, is_tiny);

    __m256d inv_det = _mm256_div_pd(_mm256_set1_pd(1.0), det);

    _mm256_storeu_pd(inv_a, _mm256_mul_pd(vd, inv_det));
    _mm256_storeu_pd(inv_b, _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), vb), inv_det));
    _mm256_storeu_pd(inv_c, _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), vc), inv_det));
    _mm256_storeu_pd(inv_d, _mm256_mul_pd(va, inv_det));
}

// Batched 2x2 matrix multiplication: C = A * B for 4 matrix pairs
inline void batch_matmul2x2_avx2(
    const double* a00, const double* a01, const double* a10, const double* a11,
    const double* b00, const double* b01, const double* b10, const double* b11,
    double* c00, double* c01, double* c10, double* c11)
{
    __m256d va00 = _mm256_loadu_pd(a00);
    __m256d va01 = _mm256_loadu_pd(a01);
    __m256d va10 = _mm256_loadu_pd(a10);
    __m256d va11 = _mm256_loadu_pd(a11);

    __m256d vb00 = _mm256_loadu_pd(b00);
    __m256d vb01 = _mm256_loadu_pd(b01);
    __m256d vb10 = _mm256_loadu_pd(b10);
    __m256d vb11 = _mm256_loadu_pd(b11);

    // c00 = a00*b00 + a01*b10, c01 = a00*b01 + a01*b11
    // c10 = a10*b00 + a11*b10, c11 = a10*b01 + a11*b11
    _mm256_storeu_pd(c00, _mm256_fmadd_pd(va00, vb00, _mm256_mul_pd(va01, vb10)));
    _mm256_storeu_pd(c01, _mm256_fmadd_pd(va00, vb01, _mm256_mul_pd(va01, vb11)));
    _mm256_storeu_pd(c10, _mm256_fmadd_pd(va10, vb00, _mm256_mul_pd(va11, vb10)));
    _mm256_storeu_pd(c11, _mm256_fmadd_pd(va10, vb01, _mm256_mul_pd(va11, vb11)));
}

// Batched 2x2 matrix-vector multiply: y = A * x for 4 pairs
inline void batch_matvec2x2_avx2(
    const double* a00, const double* a01, const double* a10, const double* a11,
    const double* x0, const double* x1,
    double* y0, double* y1)
{
    __m256d va00 = _mm256_loadu_pd(a00);
    __m256d va01 = _mm256_loadu_pd(a01);
    __m256d va10 = _mm256_loadu_pd(a10);
    __m256d va11 = _mm256_loadu_pd(a11);

    __m256d vx0 = _mm256_loadu_pd(x0);
    __m256d vx1 = _mm256_loadu_pd(x1);

    // y0 = a00*x0 + a01*x1, y1 = a10*x0 + a11*x1
    _mm256_storeu_pd(y0, _mm256_fmadd_pd(va00, vx0, _mm256_mul_pd(va01, vx1)));
    _mm256_storeu_pd(y1, _mm256_fmadd_pd(va10, vx0, _mm256_mul_pd(va11, vx1)));
}

// Batched scalar * matrix for 4 matrices
inline void batch_scale_matrix2x2_avx2(
    const double* scale,
    const double* a00, const double* a01, const double* a10, const double* a11,
    double* out00, double* out01, double* out10, double* out11)
{
    __m256d vs = _mm256_loadu_pd(scale);

    _mm256_storeu_pd(out00, _mm256_mul_pd(vs, _mm256_loadu_pd(a00)));
    _mm256_storeu_pd(out01, _mm256_mul_pd(vs, _mm256_loadu_pd(a01)));
    _mm256_storeu_pd(out10, _mm256_mul_pd(vs, _mm256_loadu_pd(a10)));
    _mm256_storeu_pd(out11, _mm256_mul_pd(vs, _mm256_loadu_pd(a11)));
}

#endif // ARAP_USE_AVX2

// Scalar fallback implementations
namespace scalar {

inline void svd2x2(
    double a, double b, double c, double d,
    double& u00, double& u01, double& u10, double& u11,
    double& v00, double& v01, double& v10, double& v11,
    double& sigma0, double& sigma1)
{
    // Closed-form 2x2 SVD
    double E = (a + d) * 0.5;
    double F = (a - d) * 0.5;
    double G = (b + c) * 0.5;
    double H = (b - c) * 0.5;

    double Q = std::sqrt(E*E + H*H);
    double R = std::sqrt(F*F + G*G);

    sigma0 = Q + R;
    double s2_signed = Q - R;
    sigma1 = std::abs(s2_signed);

    double alpha = std::atan2(H, E);
    double beta  = std::atan2(G, F);
    double phi   = 0.5 * (beta - alpha);
    double theta = 0.5 * (beta + alpha);

    double sin_phi = std::sin(phi);
    double cos_phi = std::cos(phi);
    double sin_theta = std::sin(theta);
    double cos_theta = std::cos(theta);

    double flip_sign = (s2_signed < 0) ? -1.0 : 1.0;

    u00 = cos_phi;
    u01 = -sin_phi * flip_sign;
    u10 = sin_phi;
    u11 = cos_phi * flip_sign;

    v00 = cos_theta;
    v01 = -sin_theta;
    v10 = sin_theta;
    v11 = cos_theta;
}

inline void rotation_from_svd(
    double u00, double u01, double u10, double u11,
    double v00, double v01, double v10, double v11,
    double& r00, double& r01, double& r10, double& r11)
{
    // If det(U) < 0, flip U's second column
    double detU = u00*u11 - u01*u10;
    if (detU < 0) {
        u01 = -u01;
        u11 = -u11;
    }

    // R = U * V^T
    r00 = u00*v00 + u01*v01;
    r01 = u00*v10 + u01*v11;
    r10 = u10*v00 + u11*v01;
    r11 = u10*v10 + u11*v11;
}

inline void inverse2x2(
    double a, double b, double c, double d,
    double& inv_a, double& inv_b, double& inv_c, double& inv_d)
{
    double det = a*d - b*c;
    if (std::abs(det) < 1e-300) {
        det = (det < 0) ? -1e-300 : 1e-300;
    }
    double inv_det = 1.0 / det;

    inv_a = d * inv_det;
    inv_b = -b * inv_det;
    inv_c = -c * inv_det;
    inv_d = a * inv_det;
}

} // namespace scalar

} // namespace arap_simd

#endif // ARAP_SIMD_H