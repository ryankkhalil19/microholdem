"""Build script for the C++ extension. Everything else lives in pyproject.toml."""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "microholdem._core",
        ["src/microholdem/_core.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3"],
    )
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
