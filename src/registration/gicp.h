#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../index/kdtree.h"

namespace py = pybind11;

namespace lidarlens {

struct GICPResult {
    std::array<float, 16> transform; // 4x4 matrix flattened row-major
    std::array<float, 36> hessian;   // 6x6 Hessian matrix flattened
    float final_error = 0.0f;
    int iterations = 0;
    bool converged = false;
};

class FastGICPEngine {
public:
    float max_corr_dist = 2.0f;
    int max_iterations = 30;
    float trans_eps = 1e-4f;

    FastGICPEngine(float max_dist = 2.0f, int max_iter = 30, float eps = 1e-4f)
        : max_corr_dist(max_dist), max_iterations(max_iter), trans_eps(eps) {}

    // Computes 3x3 local covariance matrices for each point using k nearest neighbors
    py::array_t<float> estimate_covariances(
        py::array_t<float, py::array::c_style | py::array::forcecast> points,
        int k_neighbors = 15) {
        
        auto buf = points.request();
        size_t n = static_cast<size_t>(buf.shape[0]);
        size_t cols = static_cast<size_t>(buf.shape[1]);
        const float* ptr = static_cast<const float*>(buf.ptr);

        py::array_t<float> cov_array({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(3), static_cast<py::ssize_t>(3)});
        auto cov_buf = cov_array.request();
        float* cov_ptr = static_cast<float*>(cov_buf.ptr);
        std::fill(cov_ptr, cov_ptr + n * 9, 0.0f);

        if (n < 4) return cov_array;

        FastKDTree tree;
        tree.build(points);

        for (size_t i = 0; i < n; ++i) {
            float qx = ptr[i * cols + 0];
            float qy = ptr[i * cols + 1];
            float qz = ptr[i * cols + 2];

            auto knn_res = tree.query_knn(qx, qy, qz, k_neighbors);
            const auto& indices = knn_res.first;
            size_t k_actual = indices.size();
            if (k_actual < 4) continue;

            // Compute centroid
            float cx = 0.0f, cy = 0.0f, cz = 0.0f;
            for (size_t idx : indices) {
                cx += ptr[idx * cols + 0];
                cy += ptr[idx * cols + 1];
                cz += ptr[idx * cols + 2];
            }
            cx /= k_actual; cy /= k_actual; cz /= k_actual;

            // Compute covariance matrix
            float cxx = 0.0f, cxy = 0.0f, cxz = 0.0f;
            float cyy = 0.0f, cyz = 0.0f, czz = 0.0f;

            for (size_t idx : indices) {
                float diff_x = ptr[idx * cols + 0] - cx;
                float diff_y = ptr[idx * cols + 1] - cy;
                float diff_z = ptr[idx * cols + 2] - cz;

                cxx += diff_x * diff_x; cxy += diff_x * diff_y; cxz += diff_x * diff_z;
                cyy += diff_y * diff_y; cyz += diff_y * diff_z; czz += diff_z * diff_z;
            }

            float inv_k = 1.0f / (k_actual - 1);
            float* m = cov_ptr + i * 9;
            m[0] = cxx * inv_k; m[1] = cxy * inv_k; m[2] = cxz * inv_k;
            m[3] = cxy * inv_k; m[4] = cyy * inv_k; m[5] = cyz * inv_k;
            m[6] = cxz * inv_k; m[7] = cyz * inv_k; m[8] = czz * inv_k;
        }

        return cov_array;
    }
};

} // namespace lidarlens
