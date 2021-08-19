cmake_minimum_required(VERSION 3.21)
project(pbf3d)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=znver2 -mtune=znver2 -pipe")

enable_testing()

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

# Copy models to build directory
set(models "${PROJECT_SOURCE_DIR}/models")
file(COPY models DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory("${PROJECT_SOURCE_DIR}/src")
add_subdirectory("${PROJECT_SOURCE_DIR}/samples")
add_subdirectory("${PROJECT_SOURCE_DIR}/tests")
