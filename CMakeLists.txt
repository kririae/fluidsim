cmake_minimum_required(VERSION 3.21)
project(fluidsim LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe")

find_package(CUDA REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND OR OpenMP_CXX_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

# Copy models to build directory
set(models "${PROJECT_SOURCE_DIR}/models")
file(COPY models DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

enable_testing()
add_subdirectory("${PROJECT_SOURCE_DIR}/src")
add_subdirectory("${PROJECT_SOURCE_DIR}/samples")
add_subdirectory("${PROJECT_SOURCE_DIR}/tests")
