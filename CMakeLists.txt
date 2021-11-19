option(WITH_PYTHON_EXTENSION "Enable python interface compilation" ON)
option(WITH_CUDA_PBD "Enable PBD CUDA optimization" ON)
option(WITH_OPENVDB "Enable OpenVDB powered remesher" OFF)

cmake_minimum_required(VERSION 3.21)
project(fluidsim LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe")

if (WITH_PYTHON_EXTENSION)
  #  message("-- Enable pybind11 for python interface compilation")
  find_package(pybind11 REQUIRED)
endif ()

find_package(CUDA REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND OR OpenMP_CXX_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

# Copy models to build directory
set(models "${PROJECT_SOURCE_DIR}/models")
file(COPY models DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

enable_testing()
add_subdirectory("${PROJECT_SOURCE_DIR}/src")
add_subdirectory("${PROJECT_SOURCE_DIR}/samples")
add_subdirectory("${PROJECT_SOURCE_DIR}/tests")

