"""ZX-diagram utilities: graph-like conversion, feasible-action enumeration, and rule application.

TODO: migrate from Section 3 of the dissertation notebook:
    - circuit_to_graphlike, gate_count, t_count, two_qubit_gate_count, extract_and_cleanup
    - find_feasible_lc, find_feasible_pivots
    - apply_local_complementation, apply_pivot

Note (see docs/deviations_from_paper.md): this module currently only implements the
Clifford-only interior-spider rules (LC on +-pi/2 spiders, pivoting on 0/pi spiders).
The original repo's gym_zx/envs/zx_env.py additionally implements pivot-boundary,
pivot-gadget, identity removal, and gadget fusion for full Clifford+T handling.
"""
