//
// Created by kr2 on 8/10/21.
//

#ifndef PBF3D_SRC_COMMON_HPP_
#define PBF3D_SRC_COMMON_HPP_

// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <random>

// TO BE RECONSTRUCT into `class`
constexpr float border = 20.0f;
constexpr float epsilon = 1e-5;
constexpr float radius = 2.0f;
constexpr float radius2 = radius * radius;
// other pbf parameters are defined in pbd.cpp

// For consideration of flexibility
using color = glm::vec3;
using vec3 = glm::vec3;
using ivec3 = glm::ivec3;
template<typename T> using hvector = std::vector<T>;
using namespace std::literals::string_literals;

#define CUDA_FUNC_DEC __attribute__((host)) __attribute__((device))
#define TODO() \
  do { \
    std::cerr << __FILE__ << " " << __LINE__ << " " << __func__ \
              << " is not implemented" << std::endl; \
  } while (0)

// Function definition
[[maybe_unused]] CUDA_FUNC_DEC float fast_pow(float a, int b);
[[maybe_unused]] CUDA_FUNC_DEC vec3 color_ramp(float t,
                                               const color &col_left,
                                               const color &col_right);

#endif  // PBF3D_SRC_COMMON_HPP_
