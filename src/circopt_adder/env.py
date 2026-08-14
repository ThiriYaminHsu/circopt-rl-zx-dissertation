"""Gym environment wrapping a circuit generator for single-episode ZX-diagram optimisation.

TODO: migrate ZXOptEnv from Section 5 of the dissertation notebook (PyG Data observation
construction, reward shaping, episode termination).

Note (see docs/deviations_from_paper.md): the reset() baseline uses clifford_simp in
place of the original repo's zx.simplify.greedy_simp, which is not available in
pyzx 0.10.5.
"""
