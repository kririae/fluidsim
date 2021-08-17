//
// Created by kr2 on 8/10/21.
//

#include "shader.cuh"
#include <fstream>
#include <iostream>
#include <sstream>

Shader::Shader()
{
  auto open_file = [&](const GLchar *filePath) -> std::string {
    std::string done;
    std::ifstream shader_file;
    std::stringstream shader_file_stream;

    shader_file.open(filePath);
    if (!shader_file) {
      shader_file.close();
      throw std::ifstream::failure("Failed to find corresponding file.");
    }
    shader_file_stream << shader_file.rdbuf();
    shader_file.close();

    done = shader_file_stream.str();
    return done;
  };

  std::string v_code, f_code;
  try {
    v_code = open_file("../src/vert.glsl");
    f_code = open_file("../src/frag.glsl");
  }
  catch (std::ifstream::failure &e) {
    std::cerr << "failed to read shader files\n" << std::endl;
    glfwTerminate();
  }

  const char *v_code_c = v_code.c_str();
  const char *f_code_c = f_code.c_str();

  auto gl_compile_info = [](auto _shader) {
    int success;
    char info_log[512];
    glGetShaderiv(_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(_shader, 512, nullptr, info_log);
      std::cerr << "failed to compile _shader\n" << info_log << std::endl;
      glfwTerminate();
    }
  };

  unsigned int vertex_shader;  // generate vertex shader object
  vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex_shader, 1, &v_code_c, nullptr);
  glCompileShader(vertex_shader);

  gl_compile_info(vertex_shader);

  unsigned int fragment_shader;
  fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment_shader, 1, &f_code_c, nullptr);
  glCompileShader(fragment_shader);

  gl_compile_info(fragment_shader);

  this->id = glCreateProgram();
  glAttachShader(this->id, vertex_shader);
  glAttachShader(this->id, fragment_shader);
  glLinkProgram(this->id);

  int success;
  char info_log[512];
  glGetProgramiv(id, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(id, 512, nullptr, info_log);
    std::cerr << "failed to link shader\n" << info_log << std::endl;
  }

  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);
}

void Shader::del()
{
  glDeleteShader(id);
}

void Shader::use() const
{
  glUseProgram(this->id);
}

void Shader::set_bool(const std::string &name, bool value) const
{
  glUniform1i(glGetUniformLocation(id, name.c_str()), (int)value);
}

void Shader::set_int(const std::string &name, int value) const
{
  glUniform1i(glGetUniformLocation(id, name.c_str()), value);
}

void Shader::set_float(const std::string &name, float value) const
{
  glUniform1f(glGetUniformLocation(id, name.c_str()), value);
}

void Shader::set_vec3(const std::string &name,
                      float x0,
                      float x1,
                      float x2) const
{
  glUniform3f(glGetUniformLocation(id, name.c_str()), x0, x1, x2);
}

void Shader::set_vec3(const std::string &name, const vec3 &value) const
{
  glUniform3f(
      glGetUniformLocation(id, name.c_str()), value.x, value.y, value.z);
}
void Shader::set_mat4(const std::string &name, const glm::mat4 &value) const
{
  int val_loc = glGetUniformLocation(id, name.c_str());
  glUniformMatrix4fv(val_loc, 1, GL_FALSE, glm::value_ptr(value));
}

Shader::~Shader() = default;
