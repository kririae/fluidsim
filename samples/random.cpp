//
// Created by kr2 on 8/11/21.
//

#include "gui.hpp"
#include "particle.hpp"
#include "pbd.hpp"
#include <iostream>
#include <random>

constexpr int NParticles = 6000;
constexpr int WIDTH = 1280, HEIGHT = 720;

int main()
{
  // random generator initialization
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(-border, border);

  RTGUI_particles gui(WIDTH, HEIGHT);
  hvector<SPHParticle> p;
  p.reserve(NParticles);
  for (int i = 0; i < NParticles; ++i) {
    p.emplace_back(dist(mt), dist(mt), dist(mt));
  }

  gui.main_loop([&]() { gui.set_particles(p); });
  gui.del();

  return 0;
}
