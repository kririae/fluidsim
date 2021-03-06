# fluidsim

> Note: Development of this repo has stopped, and the refactored version has been migrated to `FluidVisualization`.

My implementation of Position-Based Fluid[1] (or more?)  and infrastructures[2] for further use.

## Requirements

- GLFW (3.3 or above)
- ImGui (1.84.1 or above, hard requirement)
- CUDA (11.4.1 or above)
- OpenVDB (8.1.0 or above)
- g++ only (I don't know why yet)

## Usage

To build:

```bash
  $ make -j12
```

To test(after building):

```bash
  $ make test
```

## TODO

- [ ] other mathematical part of PBF (for accuracy)
- [x] normal reconstruction
- [ ] work as a blender plugin (for convenience)
- [ ] simple raytracing render, for caustics

... and other algorithms

## Reference

[1] Macklin M, Müller M. Position based fluids[J]. ACM Transactions on Graphics (TOG), 2013, 32(4): 104.

[2]: Ihmsen, M., Akinci, N., Becker, M., & Teschner, M. (2010). A Parallel SPH Implementation on Multi-Core CPUs.
Computer Graphics Forum, 30(1), 99–112.
