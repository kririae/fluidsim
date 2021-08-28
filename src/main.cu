//
// Created by kr2 on 8/11/21.
//

#include "gui.hpp"
#include "particle.hpp"
#include "pbd.hpp"

constexpr int NParticles = 2000;
constexpr int WIDTH = 1280, HEIGHT = 720;

int main()
{
  // random generator initialization
  RTGUI_particles gui(WIDTH, HEIGHT);
  PBDSolver pbd(radius);
  gui.set_solver(&pbd);
  std::function<void()> callback = [obj = &pbd, gui = &gui] {
    obj->update_gui(gui);
  };

  constexpr int _range = static_cast<int>(border / 1.5f);
  constexpr float coeff = 0.6f;
  for (float x = -_range; x <= _range; x += coeff * radius)
    for (float y = -_range; y <= _range; y += coeff * radius)
      for (float z = -_range; z <= _range; z += coeff * radius)
        pbd.add_particle(SPHParticle(x, y, z));

  gui.main_loop(callback);
  gui.del();

  return 0;
}
