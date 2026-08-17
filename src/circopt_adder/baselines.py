"""Deterministic pyzx baselines (no training required).

Migrated from Section 10 of the dissertation notebook. Runs PyZX's own
simplification routines on the same evaluation circuits, for the same comparison
table structure as Table 2 in Riu et al. and Table 1 in AlphaTensor-Quantum.
"""

import pyzx as zx


def baseline_full_reduce(circuit: "zx.Circuit") -> "zx.Circuit":
    g = circuit.to_graph()
    zx.simplify.full_reduce(g)
    return zx.extract_circuit(g).to_basic_gates()


def baseline_basic_plus_full_reduce(circuit: "zx.Circuit") -> "zx.Circuit":
    c = zx.basic_optimization(circuit.to_basic_gates())
    g = c.to_graph()
    zx.simplify.full_reduce(g)
    c2 = zx.extract_circuit(g).to_basic_gates()
    return zx.basic_optimization(c2)
