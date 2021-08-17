cmake_minimum_required(VERSION 3.21)
project(pbf3d LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=znver2 -mtune=znver2 -pipe")

find_package(CUDA REQUIRED)

enable_testing()

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

add_subdirectory("${PROJECT_SOURCE_DIR}/src")
add_subdirectory("${PROJECT_SOURCE_DIR}/samples")
add_subdirectory("${PROJECT_SOURCE_DIR}/tests")
