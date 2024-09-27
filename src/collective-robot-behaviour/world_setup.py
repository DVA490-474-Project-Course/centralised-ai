from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'world',  # Name of the module in Python
        sources=['world.cpp'],  # Source file
        include_dirs=[pybind11.get_include()],  # pybind11 headers
        language='c++',
        extra_compile_args=['-std=c++20'],  # Enforce C++20 standard
    ),
]

setup(
    name='world',
    version='1.0',
    ext_modules=ext_modules,
)
