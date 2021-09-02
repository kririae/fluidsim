//
// Created by kr2 on 8/9/21.
//

#include "gui.hpp"
#include "OBJ_Loader.h"
#include "common.hpp"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "mesher.hpp"
#include "particle.hpp"
#include "pbd.hpp"
#include "solver.hpp"
#include <chrono>
#include <iostream>
#include <memory>
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
  p_shader = std::make_unique<Shader>();
  m_shader = std::make_unique<Shader>("../src/m_vert.glsl"s,
                                      "../src/m_frag.glsl"s);

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

void RTGUI_particles::set_solver(PBDSolver *_solver)
{
  solver = _solver;
}

void RTGUI_particles::set_particles(const hvector<SPHParticle> &_p)
{
  p = _p;

  if (VAO != 0)
    glDeleteVertexArrays(1, &VAO);

  if (VBO != 0)
    glDeleteBuffers(1, &VBO);

  if (EBO != 0)
    glDeleteBuffers(1, &EBO);

  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);

  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);

  if (remesh) {
    // if (mesh == nullptr)
    construct_mesh();

    glBufferData(GL_ARRAY_BUFFER,
                 mesh->size() * sizeof(vec3),
                 mesh->data(),
                 GL_STREAM_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0, 3, GL_FLOAT, GL_FALSE, sizeof(vec3) * 3, (void *)nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1, 3, GL_FLOAT, GL_FALSE, sizeof(vec3) * 3, (void *)sizeof(vec3));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(
        2, 2, GL_FLOAT, GL_FALSE, sizeof(vec3) * 3, (void *)(2 * sizeof(vec3)));

    // Initialize indicies(EBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 indicies->size() * sizeof(uint),
                 indicies->data(),
                 GL_STREAM_DRAW);
  } else {
    hvector<float> points;
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
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void RTGUI_particles::main_loop(const std::function<void()> &callback)
{
  // Call set_particles in substep function and return the newly
  // generated particles

  std::cout << "entered main_loop" << std::endl;
  while (!glfwWindowShouldClose(window)) {
    ++frame;

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
      ImGuiIO &io = ImGui::GetIO();
      ImGui::Begin(
          "PBD Controller", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
      ImGui::Text("Framerate: %.1f", io.Framerate);
      if (solver) {
        ImGui::SliderFloat("rho_0", &(solver->rho_0), 5.0f, 40.0f);
        ImGui::SliderInt("iter", &(solver->iter), 1, 10);
        ImGui::SliderFloat("ext_f.x", &(solver->ext_f.x), -10.0f, 10.0f);
        ImGui::SliderFloat("ext_f.y", &(solver->ext_f.y), -10.0f, 10.0f);
        ImGui::SliderFloat("ext_f.z", &(solver->ext_f.z), -10.0f, 10.0f);
      }
      ImGui::Checkbox("rotate", &rotate);
      ImGui::Checkbox("remesh", &remesh);
      exportMesh |= ImGui::Button("Export Current");
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
  model = glm::rotate(model, rotate_y, vec3(0.0f, 1.0f, 0.0f));
  model = glm::scale(model, vec3(1 / border));

  auto camera_pos = vec3(0.0f, 3.0f, 6.0f);
  auto camera_center = vec3(0.0f);
  auto camera_up = vec3(0.0f, 1.0f, 0.0f);
  auto view = glm::lookAt(camera_pos, camera_center, camera_up);
  auto projection = glm::perspective(
      glm::radians(45.0f), static_cast<float>(width) / height, 0.1f, 100.0f);

  glBindVertexArray(VAO);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_MULTISAMPLE);

  assert(m_shader != nullptr);
  if (remesh) {
    assert(mesh != nullptr);
    assert(indicies != nullptr);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    m_shader->use();
    m_shader->set_mat4("model", model);
    m_shader->set_mat4("view", view);
    m_shader->set_mat4("projection", projection);
    m_shader->set_vec3("object_color", glm::vec3(1.0f));
    m_shader->set_vec3("light_color", glm::vec3(1.0f));
    m_shader->set_vec3("light_pos", camera_pos);
    m_shader->set_vec3("view_pos", camera_pos);
    // glDrawArrays(GL_TRIANGLES, 0, mesh->size() / 3);
    glDrawElements(GL_TRIANGLES, indicies->size(), GL_UNSIGNED_INT, (void *)0);
  } else {
    p_shader->use();
    p_shader->set_mat4("model", model);
    p_shader->set_mat4("view", view);
    p_shader->set_mat4("projection", projection);
    glPointSize(1.6f);
    glDrawArrays(GL_POINTS, 0, p.size());
  }

  glBindVertexArray(0);
}

void RTGUI_particles::del()
{
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  p_shader->del();
  m_shader->del();
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

void RTGUI_particles::construct_mesh()
{
  // Construct mesh from p

  // OBJ_Loader version
  // objl::Loader loader;
  // bool success = loader.LoadFile("../models/bunny/bunny.obj");
  // if (!success) {
  //   std::cerr << "obj file load failed" << std::endl;
  //   glfwTerminate();
  // }
  //
  // std::cout << "obj mesh size: " << loader.LoadedMeshes.size() << std::endl;
  // mesh = std::make_shared<hvector<vec3>>();
  // indicies = std::make_shared<hvector<uint>>(loader.LoadedMeshes[0].Indices);
  //
  // for (auto &i : loader.LoadedMeshes[0].Vertices) {
  //   mesh->emplace_back(i.Position.X, i.Position.Y, i.Position.Z);
  //   mesh->emplace_back(i.Normal.X, i.Normal.Y, i.Normal.Z);
  //   mesh->emplace_back(i.TextureCoordinate.X, i.TextureCoordinate.Y, 0.0f);
  // }

  // -------------------------------------
  // OpenVDB version, openvdb-style coding :)
  // Construct mesh through `data`

  std::cout << "--- meshing start ---" << std::endl;
  auto ms_start = std::chrono::system_clock::now();
  std::vector<openvdb::Vec3s> points;
  std::vector<openvdb::Vec3I> triangles;
  std::vector<openvdb::Vec4I> quads;
  particleToMesh(p, points, triangles, quads);
  std::cout << "n points: " << points.size() << std::endl;
  std::cout << "n triangles: " << triangles.size() << std::endl;
  std::cout << "n quads: " << quads.size() << std::endl;
  indicies = std::make_shared<hvector<uint>>();
  mesh = std::make_shared<hvector<vec3>>();

  for (auto &i : triangles) {
    indicies->push_back(i.x());
    indicies->push_back(i.y());
    indicies->push_back(i.z());
  }

  // TODO: calculate the convex hull
  for (auto &i : quads) {
    indicies->push_back(i.x());
    indicies->push_back(i.y());
    indicies->push_back(i.z());

    indicies->push_back(i.x());
    indicies->push_back(i.z());
    indicies->push_back(i.w());
  }

  for (auto &i : points) {
    mesh->emplace_back(i.x(), i.y(), i.z());
    mesh->emplace_back(0, 1, 0);
    mesh->emplace_back(0, 0, 0);
  }

  if (exportMesh) {
    export_mesh("pbf_sim_");
    // exportMesh = false;
  }

  auto ms_end = std::chrono::system_clock::now();
  std::chrono::duration<float> ms_diff = ms_end - ms_start;
  std::cout << "meshing complete: " << ms_diff.count() * 1000 << " ms"
            << std::endl;
  std::cout << std::endl;
}

void RTGUI_particles::export_mesh(std::string filename)
{
  assert(mesh != nullptr && indicies != nullptr);

  std::ofstream of;
  of.open(filename + std::to_string(frame) + ".obj");
  for (uint i = 0; i < mesh->size(); i += 3) {
    of << "v " << (*mesh)[i].x << " " << (*mesh)[i].y << " " << (*mesh)[i].z
       << std::endl;
  }
  assert(indicies->size() % 3 == 0);
  for (uint i = 0; i < indicies->size(); i += 3) {
    of << "f " << (*indicies)[i] + 1 << " " << (*indicies)[i + 1] + 1 << " "
       << (*indicies)[i + 2] + 1 << std::endl;
  }
  of.close();
}
