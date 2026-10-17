#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

namespace lidarlens {

struct alignas(16) PointXYZI {
    float x;
    float y;
    float z;
    float intensity;

    PointXYZI() : x(0.0f), y(0.0f), z(0.0f), intensity(0.0f) {}
    PointXYZI(float x_, float y_, float z_, float i_) : x(x_), y(y_), z(z_), intensity(i_) {}
};

class LidarBuffer {
public:
    std::vector<PointXYZI> data;

    LidarBuffer() = default;
    explicit LidarBuffer(size_t capacity) {
        data.reserve(capacity);
    }

    void reserve(size_t n) {
        data.reserve(n);
    }

    void resize(size_t n) {
        data.resize(n);
    }

    void clear() {
        data.clear();
    }

    size_t size() const {
        return data.size();
    }

    bool empty() const {
        return data.empty();
    }

    void add_point(float x, float y, float z, float intensity) {
        data.emplace_back(x, y, z, intensity);
    }

    // Zero-copy NumPy view: exposes the underlying continuous C++ memory directly
    // to Python as a (N, 4) float32 ndarray with strides (16, 4) bytes.
    py::array_t<float> get_numpy_view() {
        return py::array_t<float>(
            { static_cast<ssize_t>(data.size()), static_cast<ssize_t>(4) },
            { static_cast<ssize_t>(sizeof(PointXYZI)), static_cast<ssize_t>(sizeof(float)) },
            reinterpret_cast<float*>(data.data()),
            py::cast(this)
        );
    }

    // Fast zero-copy or bulk copy ingestion from an input NumPy array
    void load_from_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> input) {
        auto buf = input.request();
        if (buf.ndim != 2 || buf.shape[1] < 3) {
            throw std::invalid_argument("Input array must have shape (N, 3) or (N, 4)");
        }

        size_t n = static_cast<size_t>(buf.shape[0]);
        size_t cols = static_cast<size_t>(buf.shape[1]);
        const float* ptr = static_cast<const float*>(buf.ptr);

        data.resize(n);
        for (size_t i = 0; i < n; ++i) {
            data[i].x = ptr[i * cols + 0];
            data[i].y = ptr[i * cols + 1];
            data[i].z = ptr[i * cols + 2];
            data[i].intensity = (cols >= 4) ? ptr[i * cols + 3] : 0.0f;
        }
    }

    // Fast C++ AABB bounds computation
    std::pair<std::vector<float>, std::vector<float>> compute_aabb() const {
        if (data.empty()) {
            return {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
        }
        float min_x = data[0].x, max_x = data[0].x;
        float min_y = data[0].y, max_y = data[0].y;
        float min_z = data[0].z, max_z = data[0].z;

        for (const auto& p : data) {
            min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
            min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
            min_z = std::min(min_z, p.z); max_z = std::max(max_z, p.z);
        }
        return {{min_x, min_y, min_z}, {max_x, max_y, max_z}};
    }
};

} // namespace lidarlens
