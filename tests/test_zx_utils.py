"""Sanity tests for the Clifford-only ZX rewrite rules.

Checks that apply_local_complementation and apply_pivot preserve the circuit's
unitary (by comparing tensors before/after on small random Clifford diagrams), and
that find_feasible_lc / find_feasible_pivots only return genuinely-applicable
actions on a hand-built graph.
"""

import random
from fractions import Fraction

import pyzx as zx
import pytest

from circopt_adder.zx_utils import (
    apply_local_complementation,
    apply_pivot,
    find_feasible_lc,
    find_feasible_pivots,
)


def _random_clifford_graphlike(n_qubits=4, depth=20, seed=0):
    # zx.generate.cliffords has no seed parameter of its own -- it draws from
    # the global `random` module, so seeding that directly is what makes this
    # reproducible across test runs.
    random.seed(seed)
    g = zx.generate.cliffords(n_qubits, depth)
    zx.simplify.to_graph_like(g)
    return g


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_local_complementation_preserves_tensor(seed):
    g = _random_clifford_graphlike(seed=seed)
    targets = find_feasible_lc(g)
    if not targets:
        pytest.skip("no LC-eligible spider in this random graph")

    before = g.copy()
    apply_local_complementation(g, targets[0])

    assert zx.compare_tensors(before, g)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pivot_preserves_tensor(seed):
    g = _random_clifford_graphlike(seed=seed)
    pairs = find_feasible_pivots(g)
    if not pairs:
        pytest.skip("no pivot-eligible pair in this random graph")

    before = g.copy()
    v1, v2 = pairs[0]
    apply_pivot(g, v1, v2)

    assert zx.compare_tensors(before, g)


def test_find_feasible_lc_only_returns_interior_pi2_spiders():
    g = zx.Graph()
    b0 = g.add_vertex(zx.VertexType.BOUNDARY, qubit=0, row=0)
    v = g.add_vertex(zx.VertexType.Z, qubit=0, row=1, phase=Fraction(1, 2))
    b1 = g.add_vertex(zx.VertexType.BOUNDARY, qubit=0, row=2)
    g.add_edge((b0, v), zx.EdgeType.HADAMARD)
    g.add_edge((v, b1), zx.EdgeType.HADAMARD)
    g.set_inputs((b0,))
    g.set_outputs((b1,))

    # v is a boundary neighbour (interior check fails), so it must not be a
    # feasible LC target despite having the right phase and edge types.
    assert find_feasible_lc(g) == []


def test_find_feasible_pivots_rejects_non_hadamard_edge():
    g = zx.Graph()
    v1 = g.add_vertex(zx.VertexType.Z, qubit=0, row=0, phase=Fraction(0, 1))
    v2 = g.add_vertex(zx.VertexType.Z, qubit=1, row=0, phase=Fraction(1, 1))
    g.add_edge((v1, v2), zx.EdgeType.SIMPLE)

    assert find_feasible_pivots(g) == []
