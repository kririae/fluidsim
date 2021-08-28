#version 330 core

layout(location = 0) in vec3 v_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_tex_coord;

out vec3 f_pos;
out vec3 f_normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
  gl_Position = projection * view * model * vec4(v_pos, 1.0f);
  f_pos = vec3(model * vec4(v_pos, 1.0f));
  f_normal = mat3(transpose(inverse(model))) * v_normal;
}
