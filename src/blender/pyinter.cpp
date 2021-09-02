//
// Created by kr2 on 9/2/21.
//

// ON WORKING
#include "../gui.hpp"
#include "../pbd.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(fluidsim, m)
{
  py::class_<RTGUI_particles>(m, "RTGUI_particles")
      .def(py::init<int, int>())
      .def("set_particles", &RTGUI_particles::set_particles)
      .def("set_solver", &RTGUI_particles::set_solver)
      .def("export_mesh", &RTGUI_particles::export_mesh)
      .def("main_loop", &RTGUI_particles::main_loop)
      .def("del", &RTGUI_particles::del);

  py::class_<PBDSolver>(m, "PBDSolver")
      .def(py::init<float>())
      .def("update_gui", &PBDSolver::update_gui)
      .def("substep", &PBDSolver::substep)
      .def("add_particle", &PBDSolver::add_particle)
      .def("get_data", &PBDSolver::get_data);
}
