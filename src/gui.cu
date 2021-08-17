//
// Created by kr2 on 8/9/21.
//

#include "common.cuh"
#include "gui.cuh"
#include "particle.cuh"
#include <iostream>
#include <sstream>

GUI::GUI(int WIDTH, int HEIGHT) : width(WIDTH), height(HEIGHT)
{
  // OpenGL initialization
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_SAMPLES, 4);

  // assign window
  window = glfwCreateWindow(WIDTH, HEIGHT, "pbf3d (fps: 60)", nullptr, nullptr);
  if (window == nullptr) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
  }

  glfwMakeContextCurrent(window);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
  }

  glViewport(0, 0, WIDTH, HEIGHT);

  glfwSetFramebufferSizeCallback(
      window, [](GLFWwindow *, int _width, int _height) {
        std::clog << "window size changed to " << _width << " " << _height
                  << std::endl;
        glViewport(0, 0, _width, _height);
      });
}

void GUI::main_loop(const std::function<void()> &callback)
{
  while (!glfwWindowShouldClose(window)) {
    glClear(GL_COLOR_BUFFER_BIT);
    callback();

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
}

RTGUI_particles::RTGUI_particles(int WIDTH, int HEIGHT)
    : GUI::GUI(WIDTH, HEIGHT)
{
}

void RTGUI_particles::set_particles(const hvector<SPHParticle> &_p)
{
  p = _p;

  if (VAO != 0) {
    glDeleteVertexArrays(1, &VAO);
  }

  if (VBO != 0) {
    glDeleteBuffers(1, &VBO);
  }

  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);

  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);

  std::vector<float> points;
  points.reserve(p.size() * 4);
  for (auto &i : p) {
    points.push_back(i.pos.x);
    points.push_back(i.pos.y);
    points.push_back(i.pos.z);
    points.push_back(i.rho);
  }

  glBufferData(GL_ARRAY_BUFFER,
               points.size() * sizeof(float),
               points.data(),
               GL_STREAM_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(
      0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (void *)nullptr);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(
      1, 1, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (void *)(3 * sizeof(float)));

  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void RTGUI_particles::main_loop(const std::function<void()> &callback)
{
  // Please call set_particles in callback function and return the newly
  // generated particles

  while (!glfwWindowShouldClose(window)) {
    // Do something with particles
    glClearColor(0.921f, 0.925f, 0.933f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    process_input();
    refresh_fps();
    callback();

    render_particles();

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
}

void RTGUI_particles::render_particles() const
{
  static float rotate_y = 0;
  if (rotate)
    rotate_y += 0.005;
  else
    rotate_y = 45.0f;

  if (rotate_y > 360.0f)
    rotate_y = 0;

  auto model = glm::translate(glm::mat4(1.0f), vec3(0));
  model = glm::scale(model, vec3(1 / border));
  model = glm::rotate(model, rotate_y, vec3(0.0f, 1.0f, 0.0f));

  auto camera_pos = vec3(0.0f, 3.0f, 6.0f);
  auto camera_center = vec3(0.0f);
  auto camera_up = vec3(0.0f, 1.0f, 0.0f);
  auto view = glm::lookAt(camera_pos, camera_center, camera_up);
  auto projection = glm::perspective(
      glm::radians(45.0f), static_cast<float>(width) / height, 0.1f, 100.0f);

  glBindVertexArray(VAO);

  shader.use();
  shader.set_mat4("model", model);
  shader.set_mat4("view", view);
  shader.set_mat4("projection", projection);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_MULTISAMPLE);
  glPointSize(1.8f);

  glDrawArrays(GL_POINTS, 0, p.size());
  glBindVertexArray(0);
}

void RTGUI_particles::del()
{
  shader.del();
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
}

void RTGUI_particles::process_input()
{
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS ||
      glfwGetKey(window, GLFW_KEY_CAPS_LOCK) == GLFW_PRESS) {
    std::clog << "pbf3d quited" << std::endl;
    glfwSetWindowShouldClose(window, true);
  }
}

void RTGUI_particles::refresh_fps() const
{
  static int n_frames = 0;
  static auto last_time = glfwGetTime();
  auto cur_time = glfwGetTime();
  auto delta = cur_time - last_time;
  ++n_frames;

  if (delta > 1.0f) {
    std::stringstream win_title;
    win_title << "pbf3d (fps: " << n_frames / delta << ")";
    glfwSetWindowTitle(window, win_title.str().c_str());
    n_frames = 0;
    last_time = cur_time;
  }
}
