#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../index/kdtree.h"

namespace py = pybind11;

namespace lidarlens {

class FastTemporalSOR {
public:
    int base_neighbors = 30;
    float base_std_ratio = 2.0f;
    float distance_scale = 0.05f;

    FastTemporalSOR(int k = 30, float std_ratio = 2.0f, float dist_scale = 0.05f)
        : base_neighbors(k), base_std_ratio(std_ratio), distance_scale(dist_scale) {}

    // Computes dynamic SOR mask for a single frame: returns (N,) uint8 array (1=keep, 0=outlier)
    py::array_t<uint8_t> filter(py::array_t<float, py::array::c_style | py::array::forcecast> input) {
        auto buf = input.request();
        size_t n = static_cast<size_t>(buf.shape[0]);
        size_t cols = static_cast<size_t>(buf.shape[1]);
        const float* ptr = static_cast<const float*>(buf.ptr);

        if (n == 0) {
            return py::array_t<uint8_t>(0);
        }

        FastKDTree tree;
        tree.build(input);

        std::vector<float> mean_distances(n, 0.0f);
        std::vector<float> point_ranges(n, 0.0f);
        double sum_mean_dist = 0.0;

        size_t k = static_cast<size_t>(std::max(2, std::min(base_neighbors, static_cast<int>(n))));

        for (size_t i = 0; i < n; ++i) {
            float x = ptr[i * cols + 0];
            float y = ptr[i * cols + 1];
            float z = ptr[i * cols + 2];
            float range = std::sqrt(x * x + y * y + z * z);
            point_ranges[i] = range;

            auto knn_res = tree.query_knn(x, y, z, k + 1);
            const auto& dists = knn_res.second;
            if (dists.size() > 1) {
                float sum_d = 0.0f;
                // Skip the 0-th neighbor (point itself)
                for (size_t j = 1; j < dists.size(); ++j) {
                    sum_d += dists[j];
                }
                float avg_d = sum_d / static_cast<float>(dists.size() - 1);
                mean_distances[i] = avg_d;
                sum_mean_dist += avg_d;
            } else {
                mean_distances[i] = 0.0f;
            }
        }

        double global_mean = sum_mean_dist / static_cast<double>(n);
        double variance = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double diff = mean_distances[i] - global_mean;
            variance += diff * diff;
        }
        double global_std = std::sqrt(variance / static_cast<double>(n));

        py::array_t<uint8_t> result(n);
        auto res_buf = result.request();
        uint8_t* out_ptr = static_cast<uint8_t*>(res_buf.ptr);

        for (size_t i = 0; i < n; ++i) {
            // Adaptive threshold based on distance:
            // Relax threshold for far points to preserve sparse valid returns
            float effective_ratio = base_std_ratio + distance_scale * point_ranges[i];
            double thresh = global_mean + effective_ratio * global_std;

            if (mean_distances[i] <= thresh) {
                out_ptr[i] = 1; // Keep
            } else {
                out_ptr[i] = 0; // Outlier
            }
        }

        return result;
    }
};

} // namespace lidarlens
