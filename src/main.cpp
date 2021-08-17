#include "gui.hpp"
#include "particle.hpp"
#include "pbd.hpp"
#include <iostream>
#include <random>

constexpr int NParticles = 6000;
constexpr int WIDTH = 800, HEIGHT = 600;

int main()
{
  // random generator initialization
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  return 0;
}
