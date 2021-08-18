//
// Created by kr2 on 8/9/21.
//

#include "gui.hpp"
#include "common.hpp"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "particle.hpp"
#include "pbd.hpp"
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
  glfwSwapInterval(1);  // Enable vsync

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
  }

  glViewport(0, 0, WIDTH, HEIGHT);

  // glfwSetFramebufferSizeCallback(
  //     window, [](GLFWwindow *, int _width, int _height) {
  //       std::clog << "window size changed to " << _width << " " << _height
  //                 << std::endl;
  //       glViewport(0, 0, _width, _height);
  //     });

  // Setup Dear ImGui context
  const char *glsl_version = "#version 330 core";
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);
  ImGui::StyleColorsClassic();

  std::cout << "gui finished initialization" << std::endl;
}

void GUI::main_loop(const std::function<void()> &callback)
{
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    glClear(GL_COLOR_BUFFER_BIT);
    callback();

    glfwSwapBuffers(window);
  }
}

RTGUI_particles::RTGUI_particles(int WIDTH, int HEIGHT)
    : GUI::GUI(WIDTH, HEIGHT)
{
}

void RTGUI_particles::set_solver(PBDSolver *_solver)
{
  solver = _solver;
}

void RTGUI_particles::set_particles(const std::vector<SPHParticle> &_p)
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

  if (mesh) {
    // TODO: construct mesh
  }
  else {
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
    glVertexAttribPointer(1,
                          1,
                          GL_FLOAT,
                          GL_FALSE,
                          sizeof(float) * 4,
                          (void *)(3 * sizeof(float)));
  }

  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void RTGUI_particles::main_loop(const std::function<void()> &callback)
{
  // Call set_particles in callback function and return the newly
  // generated particles

  std::cout << "entered main_loop" << std::endl;
  while (!glfwWindowShouldClose(window)) {
    // Do something with particles
    glfwPollEvents();
    glClearColor(0.16, 0.17, 0.2, 1);  // one dark
    // glClearColor(0.921f, 0.925f, 0.933f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    process_input();
    refresh_fps();
    callback();
    render_particles();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // ImGui widgets
    {
      // ImGui::SetNextWindowSize(ImVec2(360, 150), ImGuiCond_Always);
      ImGui::Begin(
          "PBD Controller", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
      ImGui::SliderFloat("rho_0", &(solver->rho_0), 5.0f, 20.0f);
      ImGui::SliderInt("iter", &(solver->iter), 1, 10);
      ImGui::SliderFloat("ext_f.x", &(solver->ext_f.x), -10.0f, 10.0f);
      ImGui::SliderFloat("ext_f.y", &(solver->ext_f.y), -10.0f, 10.0f);
      ImGui::SliderFloat("ext_f.z", &(solver->ext_f.z), -10.0f, 10.0f);
      ImGui::Checkbox("rotate", &rotate);
      ImGui::Checkbox("mesh", &mesh);
      ImGui::End();
    }

    {
      ImGuiIO &io = ImGui::GetIO();
      ImGui::Begin(
          "Program Information", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
      ImGui::Text("Framerate: %.1f", io.Framerate);
      ImGui::End();
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glfwSwapBuffers(window);
  }
}

void RTGUI_particles::render_particles() const
{
  static float rotate_y = 45.0f;
  if (rotate)
    rotate_y += 0.005;

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

  p_shader.use();
  p_shader.set_mat4("model", model);
  p_shader.set_mat4("view", view);
  p_shader.set_mat4("projection", projection);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_MULTISAMPLE);

  if (mesh) {
    // TODO: display mesh
  }
  else {
    glPointSize(1.8f);
    glDrawArrays(GL_POINTS, 0, p.size());
  }
  glBindVertexArray(0);
}

void RTGUI_particles::del()
{
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  p_shader.del();
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glfwDestroyWindow(window);
  glfwTerminate();
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
