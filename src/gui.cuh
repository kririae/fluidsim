//
// Created by kr2 on 8/9/21.
//

#ifndef PBF3D_SRC_GUI_CUH_
#define PBF3D_SRC_GUI_CUH_

#include "shader.hpp"
#include <functional>
#include <thrust/host_vector.h>

class GLFWwindow;
class SPHParticle;
class Solver;

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

  void set_particles(const thrust::host_vector<SPHParticle> &_p);
  void main_loop(const std::function<void()> &callback) override;
  void del();

 protected:
  thrust::host_vector<SPHParticle> p{};
  void render_particles() const;
  unsigned int VAO{}, VBO{};
  Shader shader{};

 private:
  void refresh_fps() const;
  void process_input();
};

#endif  // PBF3D_SRC_GUI_CUH_
