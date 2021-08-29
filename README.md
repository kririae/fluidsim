# fluidsim

> In progress with my lazy development

My implementation of Position-Based Fluid[1] (or more?)  and infrastructures[2] for further use.

## Requirements

- GLFW (3.3 or above)
- ImGui (1.84.1 or above, hard requirement)
- CUDA (11.4.1 or above)
- OpenVDB (8.1.0 or above)

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
- [ ] normal reconstruction
- [ ] work as a blender plugin (for convenience)
- [ ] simple raytracing render, for caustics

... and other algorithms

## Reference

[1] Macklin M, Müller M. Position based fluids[J]. ACM Transactions on Graphics (TOG), 2013, 32(4): 104.

[2]: Ihmsen, M., Akinci, N., Becker, M., & Teschner, M. (2010). A Parallel SPH Implementation on Multi-Core CPUs.
Computer Graphics Forum, 30(1), 99–112.
