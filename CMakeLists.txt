cmake_minimum_required(VERSION 3.22)
project(bmp_spmm)

set(BMP_SPMM_VERSION_MAJOR 0)
set(BMP_SPMM_VERSION_MAJOR 1)

enable_language(CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS ON)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# link_directories(${BOOST_ROOT}/lib)
add_subdirectory(include)
add_subdirectory(examples)
