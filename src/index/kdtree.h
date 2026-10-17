#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>
#include <queue>
#include <cmath>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace lidarlens {

// 3D Morton (Z-order) code encoder for cache-coherent spatial hashing
inline uint64_t split_by_3(uint32_t a) {
    uint64_t x = a & 0x1fffff; // 21 bits
    x = (x | (x << 32)) & 0x1f00000000ffffULL;
    x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
    x = (x | (x << 8))  & 0x100f00f00f00f00fULL;
    x = (x | (x << 4))  & 0x10c30c30c30c30c3ULL;
    x = (x | (x << 2))  & 0x1249249249249249ULL;
    return x;
}

inline uint64_t morton_encode_3d(uint32_t x, uint32_t y, uint32_t z) {
    return (split_by_3(x) << 2) | (split_by_3(y) << 1) | split_by_3(z);
}

struct KDNode {
    size_t point_idx;
    int axis;
    int left = -1;
    int right = -1;
};

class FastKDTree {
public:
    std::vector<float> points_xyz; // Stored flattened: [N * 3]
    size_t num_points = 0;
    std::vector<KDNode> nodes;
    std::vector<size_t> indices;
    int root = -1;

    FastKDTree() = default;

    void build(py::array_t<float, py::array::c_style | py::array::forcecast> input) {
        auto buf = input.request();
        num_points = static_cast<size_t>(buf.shape[0]);
        size_t cols = static_cast<size_t>(buf.shape[1]);
        const float* ptr = static_cast<const float*>(buf.ptr);

        points_xyz.resize(num_points * 3);
        indices.resize(num_points);
        for (size_t i = 0; i < num_points; ++i) {
            points_xyz[i * 3 + 0] = ptr[i * cols + 0];
            points_xyz[i * 3 + 1] = ptr[i * cols + 1];
            points_xyz[i * 3 + 2] = ptr[i * cols + 2];
            indices[i] = i;
        }

        nodes.clear();
        nodes.reserve(num_points);
        root = build_recursive(0, num_points, 0);
    }

    // KNN query: returns (indices, distances_sq)
    std::pair<std::vector<size_t>, std::vector<float>> query_knn(
        float qx, float qy, float qz, size_t k) const {
        if (num_points == 0 || k == 0) return {{}, {}};

        using DistIdx = std::pair<float, size_t>;
        std::priority_queue<DistIdx> pq;

        knn_recursive(root, qx, qy, qz, k, pq);

        std::vector<size_t> out_idx(pq.size());
        std::vector<float> out_dist(pq.size());
        size_t idx = pq.size();
        while (!pq.empty()) {
            --idx;
            out_dist[idx] = std::sqrt(pq.top().first);
            out_idx[idx] = pq.top().second;
            pq.pop();
        }
        return {out_idx, out_dist};
    }

private:
    int build_recursive(size_t start, size_t end, int depth) {
        if (start >= end) return -1;

        int axis = depth % 3;
        size_t mid = start + (end - start) / 2;

        std::nth_element(
            indices.begin() + start,
            indices.begin() + mid,
            indices.begin() + end,
            [this, axis](size_t a, size_t b) {
                return points_xyz[a * 3 + axis] < points_xyz[b * 3 + axis];
            }
        );

        int node_idx = static_cast<int>(nodes.size());
        nodes.push_back({indices[mid], axis, -1, -1});

        int left = build_recursive(start, mid, depth + 1);
        int right = build_recursive(mid + 1, end, depth + 1);

        nodes[node_idx].left = left;
        nodes[node_idx].right = right;
        return node_idx;
    }

    void knn_recursive(int node_idx, float qx, float qy, float qz, size_t k,
                       std::priority_queue<std::pair<float, size_t>>& pq) const {
        if (node_idx < 0) return;

        const auto& node = nodes[node_idx];
        size_t p_idx = node.point_idx;
        float px = points_xyz[p_idx * 3 + 0];
        float py = points_xyz[p_idx * 3 + 1];
        float pz = points_xyz[p_idx * 3 + 2];

        float dx = qx - px;
        float dy = qy - py;
        float dz = qz - pz;
        float dist_sq = dx * dx + dy * dy + dz * dz;

        if (pq.size() < k) {
            pq.push({dist_sq, p_idx});
        } else if (dist_sq < pq.top().first) {
            pq.pop();
            pq.push({dist_sq, p_idx});
        }

        float q_val = (node.axis == 0) ? qx : (node.axis == 1 ? qy : qz);
        float p_val = (node.axis == 0) ? px : (node.axis == 1 ? py : pz);
        float diff = q_val - p_val;

        int first = (diff <= 0.0f) ? node.left : node.right;
        int second = (diff <= 0.0f) ? node.right : node.left;

        knn_recursive(first, qx, qy, qz, k, pq);

        if (pq.size() < k || (diff * diff) < pq.top().first) {
            knn_recursive(second, qx, qy, qz, k, pq);
        }
    }
};

} // namespace lidarlens
