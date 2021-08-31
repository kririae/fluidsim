//
// Created by kr2 on 8/9/21.
//

#ifndef PBF3D_SRC_GUI_HPP_
#define PBF3D_SRC_GUI_HPP_

#include "shader.hpp"
#include <functional>
#include <memory>

class GLFWwindow;
class SPHParticle;
class Solver;
class PBDSolver;

class GUI {
 public:
  GUI(int WIDTH, int HEIGHT);
  GUI(const GUI &) = delete;
  GUI &operator=(const GUI &) = delete;
  virtual ~GUI() = default;
  virtual void main_loop(const std::function<void()> &callback);

 protected:
  GLFWwindow *window{};
  int width, height;
};

class RTGUI_particles : public GUI {
  // REAL-TIME GUI for Lagrange View stimulation(particles)
 public:
  RTGUI_particles(int WIDTH, int HEIGHT);
  ~RTGUI_particles() override = default;

  void set_particles(const hvector<SPHParticle> &_p);
  void set_solver(PBDSolver *_solver);
  // OBJ File will be exported as {filename}_{frame}.obj
  void export_mesh(std::string filename);
  void main_loop(const std::function<void()> &callback) override;
  void del();

 protected:
  hvector<SPHParticle> p{};
  // std::shared_ptr<hvector<SPHParticle>> p{};

  uint frame = 0;
  unsigned int VAO{}, VBO{}, EBO{};
  std::unique_ptr<Shader> p_shader{}, m_shader{};
  std::shared_ptr<hvector<vec3>> mesh;
  std::shared_ptr<hvector<uint>> indicies;
  PBDSolver *solver{};
  bool rotate = false;
  bool remesh = false;
  // float meta_radius = 1.0f;  //  metaball_radius
  bool exportMesh = false;

 private:
  void render_particles() const;
  void refresh_fps() const;
  void process_input();
  void construct_mesh();
};

#endif  // PBF3D_SRC_GUI_HPP_
