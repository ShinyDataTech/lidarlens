import os
import sys
from setuptools import setup, find_packages

c_modules = []
has_pybind = False

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    has_pybind = True
except ImportError:
    pass

if has_pybind:
    extra_compile_args = []
    if sys.platform == "win32":
        extra_compile_args = ["/std:c++20", "/O2", "/EHsc", "/utf-8"]
    else:
        extra_compile_args = ["-std=c++20", "-O3", "-march=native"]

    ext_module = Pybind11Extension(
        "lidarlens._lidarlens_cpp",
        sources=[
            "src/bindings.cpp",
            "src/buffer/buffer_protocol.cpp",
            "src/index/kdtree.cpp",
            "src/filters/temporal_sor.cpp",
            "src/registration/gicp.cpp",
        ],
        include_dirs=["src"],
        extra_compile_args=extra_compile_args,
        cxx_std=20,
    )
    c_modules = [ext_module]

cmdclass = {}
if has_pybind:
    cmdclass["build_ext"] = build_ext

setup(
    name="lidarlens",
    version="0.2.0",
    packages=find_packages(include=["lidarlens*", "pylidarlens*"]),
    ext_modules=c_modules,
    cmdclass=cmdclass,
)
