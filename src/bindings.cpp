#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "buffer/buffer_protocol.h"
#include "index/kdtree.h"
#include "filters/temporal_sor.h"
#include "registration/gicp.h"

namespace py = pybind11;

PYBIND11_MODULE(_lidarlens_cpp, m) {
    m.doc() = "LidarLens High-Performance C++20 Core Extensions";

    // ── Morton Code ──
    m.def("morton_encode_3d", &lidarlens::morton_encode_3d, "Compute 3D Morton (Z-order) code");

    // ── LidarBuffer ──
    py::class_<lidarlens::LidarBuffer>(m, "LidarBuffer")
        .def(py::init<>())
        .def(py::init<size_t>(), py::arg("capacity"))
        .def("reserve", &lidarlens::LidarBuffer::reserve, py::arg("n"))
        .def("resize", &lidarlens::LidarBuffer::resize, py::arg("n"))
        .def("clear", &lidarlens::LidarBuffer::clear)
        .def("size", &lidarlens::LidarBuffer::size)
        .def("empty", &lidarlens::LidarBuffer::empty)
        .def("add_point", &lidarlens::LidarBuffer::add_point,
             py::arg("x"), py::arg("y"), py::arg("z"), py::arg("intensity") = 0.0f)
        .def("get_numpy_view", &lidarlens::LidarBuffer::get_numpy_view,
             "Expose contiguous (N, 4) float32 memory to NumPy without copying")
        .def("load_from_numpy", &lidarlens::LidarBuffer::load_from_numpy,
             py::arg("array"), "Load (N, 3) or (N, 4) NumPy array into contiguous buffer")
        .def("compute_aabb", &lidarlens::LidarBuffer::compute_aabb,
             "Compute axis-aligned bounding box (min_xyz, max_xyz)");

    // ── FastKDTree ──
    py::class_<lidarlens::FastKDTree>(m, "FastKDTree")
        .def(py::init<>())
        .def("build", &lidarlens::FastKDTree::build, py::arg("points"))
        .def("query_knn", &lidarlens::FastKDTree::query_knn,
             py::arg("qx"), py::arg("qy"), py::arg("qz"), py::arg("k"));

    // ── FastTemporalSOR ──
    py::class_<lidarlens::FastTemporalSOR>(m, "FastTemporalSOR")
        .def(py::init<int, float, float>(),
             py::arg("k") = 30, py::arg("std_ratio") = 2.0f, py::arg("dist_scale") = 0.05f)
        .def_readwrite("base_neighbors", &lidarlens::FastTemporalSOR::base_neighbors)
        .def_readwrite("base_std_ratio", &lidarlens::FastTemporalSOR::base_std_ratio)
        .def_readwrite("distance_scale", &lidarlens::FastTemporalSOR::distance_scale)
        .def("filter", &lidarlens::FastTemporalSOR::filter, py::arg("points"));

    // ── FastGICPEngine ──
    py::class_<lidarlens::FastGICPEngine>(m, "FastGICPEngine")
        .def(py::init<float, int, float>(),
             py::arg("max_dist") = 2.0f, py::arg("max_iter") = 30, py::arg("eps") = 1e-4f)
        .def("estimate_covariances", &lidarlens::FastGICPEngine::estimate_covariances,
             py::arg("points"), py::arg("k_neighbors") = 15);
}
