//
// Created by kr2 on 8/10/21.
//

#ifndef PBF3D_SRC_COMMON_CUH_
#define PBF3D_SRC_COMMON_CUH_

// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

constexpr float border = 20.0f;
constexpr float epsilon = 1e-5;
constexpr bool rotate = false;
constexpr float radius = 1.8f;
// other pbf parameters are defined in pbd.cpp

using color = glm::vec3;
using vec3 = glm::vec3;
using ivec3 = glm::ivec3;
template<typename T>
using dvector = thrust::device_vector<T, thrust::device_allocator<T>>;
template<typename T> using hvector = thrust::host_vector<T>;
using namespace std::literals::string_literals;

vec3 color_ramp(float t, const color &col_left, const color &col_right);

[[maybe_unused]] __host__ __device__ float fpow(float a, int b);

class Random {
 public:
  Random() noexcept;
  float rand();

 private:
  std::random_device rd{};
  std::mt19937 mt;
  std::uniform_real_distribution<float> dist;
};

extern Random rd_global;

#endif  // PBF3D_SRC_COMMON_CUH_
