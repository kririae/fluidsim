#version 330 core

in vec3 f_pos;
in vec3 f_normal;
out vec4 FragColor;

uniform vec3 object_color;
uniform vec3 light_color;
uniform vec3 light_pos;
uniform vec3 view_pos;

void main() {
  float ambient_strength = 0.3;
  vec3 ambient = ambient_strength * light_color;

  vec3 norm = normalize(f_normal);
  vec3 light_dir = normalize(light_pos - f_pos);
  float diff = max(dot(norm, light_dir), 0.0);
  vec3 diffuse = diff * light_color;

  float specular_strength = 0.3;
  vec3 view_dir = normalize(view_pos - f_pos);
  vec3 halfway_dir = normalize(light_dir + view_dir);
  float spec = pow(max(dot(norm, halfway_dir), 0.0), 16.0);
  vec3 specular = specular_strength * spec * light_color;

  vec3 result = (ambient + diffuse + specular) * object_color;
  // vec3 result = (specular) * object_color;
  FragColor = vec4(result, 1.0f);
}
