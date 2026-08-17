"""ZX-diagram utilities: graph-like conversion, feasible-action enumeration, and rule application.

Migrated from Section 3 of the dissertation notebook. Direct graph-surgery
implementations of the two Clifford simplification rules used as the RL action
space: local complementation (interior spiders with phase +-pi/2) and pivoting
(interior spider pairs with phase 0 or pi, connected by a Hadamard edge), following
Duncan, Kissinger, Perdrix & van de Wetering (2020) and Eq. 2-3 of Riu et al. These
operate directly on a pyzx.Graph via its low-level edge/phase API rather than
pyzx.simplify's batch routines, because the RL setting needs to apply and score
*one* action at a time.

Note (see docs/deviations_from_paper.md): this module currently only implements the
Clifford-only interior-spider rules (LC on +-pi/2 spiders, pivoting on 0/pi spiders).
The original repo's gym_zx/envs/zx_env.py additionally implements pivot-boundary,
pivot-gadget, identity removal, and gadget fusion for full Clifford+T handling.
"""

from fractions import Fraction
from typing import List, Optional, Tuple

import pyzx as zx
from pyzx.graph.base import BaseGraph
from pyzx.utils import VertexType, EdgeType

# ---------------------------------------------------------------------------
# Circuit <-> graph-like ZX-diagram conversion
# ---------------------------------------------------------------------------


def circuit_to_graphlike(circuit: "zx.Circuit") -> BaseGraph:
    """Convert a pyzx Circuit into its graph-like ZX-diagram form."""
    g = circuit.to_graph()
    zx.simplify.to_graph_like(g)
    return g


def gate_count(circuit: "zx.Circuit") -> int:
    return len(circuit.gates)


def t_count(circuit: "zx.Circuit") -> int:
    """Counts T-phase gates by phase value, not gate name, since circuits
    that have passed through zx.extract_circuit() get their non-Clifford
    phases synthesized as generic ZPhase gates rather than named T/T*."""
    count = 0
    for gate in circuit.gates:
        if gate.name in ("T", "T*"):
            count += 1
        elif gate.name == "ZPhase" and hasattr(gate, "phase"):
            phase_mod = gate.phase % 2          # normalize to [0, 2)
            if phase_mod in (Fraction(1, 4), Fraction(7, 4)):
                count += 1
    return count


def two_qubit_gate_count(circuit: "zx.Circuit") -> int:
    """Counts gates acting on two qubits (CNOT, CZ, SWAP, etc.), matching the two-qubit gate metric reported in Fig. 8 of Riu et al."""
    return sum(1 for gate in circuit.gates if hasattr(gate, "control"))


def extract_and_cleanup(g: BaseGraph) -> Optional["zx.Circuit"]:
    """Extract a circuit from a graph-like diagram and run cheap gate-level
    cleanup (mirrors the `basic_optimization` post-processing step used
    before reward computation in Riu et al., Sec 3.3.2). Returns None if
    extraction fails (can happen mid-episode on malformed graphs)."""
    try:
        g_copy = g.copy()
        c = zx.extract_circuit(g_copy)
        c = zx.basic_optimization(c.to_basic_gates())
        return c
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Feasible-action enumeration
# ---------------------------------------------------------------------------
# Phase convention: pyzx stores phases as Fraction multiples of pi, i.e.
# g.phase(v) == Fraction(1, 2) means phase = pi/2.

PHASE_PI_2 = Fraction(1, 2)
PHASE_3PI_2 = Fraction(3, 2)
PHASE_0 = Fraction(0, 1)
PHASE_PI = Fraction(1, 1)


def _is_interior(g: BaseGraph, v: int) -> bool:
    """A spider is interior if none of its neighbours are boundary
    (input/output) vertices, matching the definition in Section 3.1.4 of
    Nogue's thesis."""
    return all(g.type(n) != VertexType.BOUNDARY for n in g.neighbors(v))


def _all_hadamard_edges(g: BaseGraph, v: int) -> bool:
    return all(g.edge_type(g.edge(v, n)) == EdgeType.HADAMARD for n in g.neighbors(v))


def find_feasible_lc(g: BaseGraph) -> List[int]:
    """Interior Z-spiders with phase +-pi/2, connected to all neighbours
    via Hadamard edges only -- eligible for local complementation."""
    candidates = []
    for v in g.vertices():
        if g.type(v) != VertexType.Z:
            continue
        if g.phase(v) not in (PHASE_PI_2, PHASE_3PI_2):
            continue
        if not _is_interior(g, v):
            continue
        if not _all_hadamard_edges(g, v):
            continue
        candidates.append(v)
    return candidates


def find_feasible_pivots(g: BaseGraph) -> List[Tuple[int, int]]:
    """Interior Z-spider pairs with phase in {0, pi}, connected to each
    other by a Hadamard edge -- eligible for pivoting."""
    candidates = []
    for v1, v2 in g.edges():
        if g.edge_type(g.edge(v1, v2)) != EdgeType.HADAMARD:
            continue
        if g.type(v1) != VertexType.Z or g.type(v2) != VertexType.Z:
            continue
        if g.phase(v1) not in (PHASE_0, PHASE_PI):
            continue
        if g.phase(v2) not in (PHASE_0, PHASE_PI):
            continue
        if not (_is_interior(g, v1) and _is_interior(g, v2)):
            continue
        candidates.append((v1, v2))
    return candidates


# NOTE (scope, see docs/deviations_from_paper.md): the two functions above only
# cover the Clifford fragment. To handle circuits containing T gates natively
# (rather than relying on them surviving untouched until extraction), add the
# non-Clifford pivot variants (p2, p3) and gadget-fusion (gf) from Riu et al.
# Eq. 6-8, which operate on "phase gadgets" (an alpha-phase spider connected by
# a single Hadamard edge to an otherwise-phaseless spider).


# ---------------------------------------------------------------------------
# Rule application (graph surgery)
# ---------------------------------------------------------------------------


def apply_local_complementation(g: BaseGraph, v: int) -> None:
    """Eq. 2, Riu et al. Removes `v`, complements the edge set among its
    neighbours, and updates their phases by -phase(v)."""
    phase_v = g.phase(v)
    neighbours = list(g.neighbors(v))

    for n in neighbours:
        g.add_to_phase(n, -phase_v)

    for i in range(len(neighbours)):
        for j in range(i + 1, len(neighbours)):
            a, b = neighbours[i], neighbours[j]
            if g.connected(a, b):
                g.remove_edge(g.edge(a, b))
            else:
                g.add_edge(g.edge(a, b), EdgeType.HADAMARD)

    g.remove_vertex(v)


def apply_pivot(g: BaseGraph, v1: int, v2: int) -> None:
    """Eq. 3, Riu et al. Removes `v1`, `v2` and complements edges between
    the three neighbourhood subsets: unique-to-v1, unique-to-v2, shared."""
    n1 = set(g.neighbors(v1)) - {v2}
    n2 = set(g.neighbors(v2)) - {v1}
    shared = n1 & n2
    only1 = n1 - shared
    only2 = n2 - shared

    j = g.phase(v1)  # coefficient of pi for v1 (0 or 1)
    k = g.phase(v2)

    for n in only1:
        g.add_to_phase(n, k)
    for n in only2:
        g.add_to_phase(n, j)
    for n in shared:
        g.add_to_phase(n, j + k + 1)

    def toggle_between(set_a, set_b):
        for a in set_a:
            for b in set_b:
                if a == b:
                    continue
                if g.connected(a, b):
                    g.remove_edge(g.edge(a, b))
                else:
                    g.add_edge(g.edge(a, b), EdgeType.HADAMARD)

    toggle_between(only1, only2)
    toggle_between(only1, shared)
    toggle_between(only2, shared)

    g.remove_vertex(v1)
    g.remove_vertex(v2)
