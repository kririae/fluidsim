//
// Created by kr2 on 8/10/21.
//

#ifndef PBF3D_SRC_SHADER_HPP_
#define PBF3D_SRC_SHADER_HPP_

#include "common.hpp"
#include <string>

class Shader {
 public:
  Shader(const std::string &vert = "../src/p_vert.glsl",
         const std::string &frag = "../src/p_frag.glsl");

  virtual void del();
  virtual void use() const;
  virtual void set_bool(const std::string &name, bool value) const;
  virtual void set_int(const std::string &name, int value) const;
  virtual void set_float(const std::string &name, float value) const;
  virtual void set_vec3(const std::string &name,
                        float x0,
                        float x1,
                        float x2) const;
  virtual void set_vec3(const std::string &name, const vec3 &value) const;
  virtual void set_mat4(const std::string &name, const glm::mat4 &value) const;

  virtual ~Shader();

 protected:
  unsigned int id{};
};

#endif  // PBF3D_SRC_SHADER_HPP_
