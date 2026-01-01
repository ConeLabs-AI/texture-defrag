#ifndef RDP_EIGEN_H
#define RDP_EIGEN_H

#include <vector>
#include <cmath>
#include <Eigen/Dense>

namespace GeometryUtils {

    /**
     * Calculates squared perpendicular distance from point P to infinite line AB.
     * Dimension-agnostic (2D, 3D, 4D, etc).
     */
    template <typename Derived>
    double getPerpendicularDistSq(
        const Eigen::MatrixBase<Derived>& P,
        const Eigen::MatrixBase<Derived>& A,
        const Eigen::MatrixBase<Derived>& B) 
    {
        const auto AB = B - A;
        const auto AP = P - A;
        double ab_sq_norm = AB.squaredNorm();

        if (ab_sq_norm < 1e-12) return AP.squaredNorm();

        // Project AP onto AB to find the closest point on the infinite line
        double t = AP.dot(AB) / ab_sq_norm;
        const auto closest_point = A + (AB * t);

        return (P - closest_point).squaredNorm();
    }

    /**
     * RDP Recursive step. Standard implementation marking keep_flags.
     */
    template <typename T, typename Alloc>
    void rdpRecursive(
        const std::vector<T, Alloc>& points,
        size_t start,
        size_t end,
        double epsilon_sq,
        std::vector<bool>& keep_flags) 
    {
        if (end - start < 2) return;

        double max_dist_sq = 0.0;
        size_t index_max = 0;

        for (size_t i = start + 1; i < end; ++i) {
            double dist_sq = getPerpendicularDistSq(points[i], points[start], points[end]);
            if (dist_sq > max_dist_sq) {
                max_dist_sq = dist_sq;
                index_max = i;
            }
        }

        if (max_dist_sq > epsilon_sq) {
            keep_flags[index_max] = true;
            rdpRecursive(points, start, index_max, epsilon_sq, keep_flags);
            rdpRecursive(points, index_max, end, epsilon_sq, keep_flags);
        }
    }

    /**
     * Ramer-Douglas-Peucker polyline simplification.
     * Works with Eigen types (Vector2d, Vector3d, Vector4d).
     */
    template <typename T, typename Alloc = std::allocator<T>>
    std::vector<T, Alloc> simplifyRDP(const std::vector<T, Alloc>& points, double epsilon) {
        if (points.size() < 3) return points;

        std::vector<bool> keep_flags(points.size(), false);
        keep_flags[0] = true;
        keep_flags[points.size() - 1] = true;

        rdpRecursive<T, Alloc>(points, 0, points.size() - 1, epsilon * epsilon, keep_flags);

        std::vector<T, Alloc> result;
        result.reserve(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            if (keep_flags[i]) result.push_back(points[i]);
        }
        return result;
    }
}

#endif // RDP_EIGEN_H