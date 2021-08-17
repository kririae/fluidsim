#version 330 core

layout(location = 0) in vec3 v_pos;
layout(location = 1) in float v_coeff;

out vec3 f_pos;
out float f_coeff;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
  gl_Position = projection * view * model * vec4(v_pos, 1.0f);

  // export to frag shader
  f_pos = vec3(model * vec4(v_pos, 1.0f));
  f_coeff = v_coeff;
}