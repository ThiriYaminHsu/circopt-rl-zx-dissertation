"""Circuit generators injected into the Gym environment at construction time.

Migrated from Section 4 of the dissertation notebook. Two interchangeable generator
functions, both with signature `() -> zx.Circuit`. This is the *only* thing that
differs between Agent R and Agent A's training runs.
"""

import random
from fractions import Fraction
from typing import Callable, Optional, Tuple

import pyzx as zx

# ---------------------------------------------------------------------------
# Random Clifford+T generator (replicates Riu et al. Sec. 4 training regime)
# ---------------------------------------------------------------------------


def make_random_circuit_generator(n_qubits: int, n_gates: int, seed: Optional[int] = None) -> Callable[[], "zx.Circuit"]:
    # `seed` is accepted for interface parity with make_adder_generator, but
    # zx.generate.cliffordT has no seed hook of its own -- it draws from
    # pyzx's global RNG state, so this generator is not independently
    # reproducible across instances the way the adder generator is.
    rng = random.Random(seed)

    def _generate() -> "zx.Circuit":
        rng.randint(0, 2**31 - 1)
        # zx.generate.cliffordT samples CNOT/CZ, Hadamard, S, and T gates
        # with these exact probabilities, matching Riu et al.'s released
        # code (gym-zx/envs/zx_env.py reset()), not the paper's prose summary
        g = zx.generate.cliffordT(n_qubits, n_gates, p_t=0.17, p_s=0.24, p_hsh=0.25)
        zx.simplify.full_reduce(g)
        return zx.extract_circuit(g)

    return _generate


# ---------------------------------------------------------------------------
# Ripple-carry adder generator (Cuccaro et al. 2004 construction)
# ---------------------------------------------------------------------------
# Built directly with pyzx's Circuit/gate API rather than an external QASM
# file, so bit-width is a free parameter. This follows the standard
# Cuccaro ripple-carry adder wiring: n+1 qubits for the two n-bit operands
# (one register overwritten in place) plus 1 ancilla/carry qubit, using
# MAJ (Toffoli+CNOT) and UMA gate blocks.
#
# NOTE: Toffoli gates are compiled here via the explicit 7-T-gate Toffoli
# decomposition below -- this is deliberate: it matches the "naive
# compilation, no measurement-based uncomputation" baseline used throughout
# Riu et al. and AlphaTensor-Quantum, so any T-count reduction the agent
# finds is attributable to the ZX rewriting itself, not to a smarter Toffoli
# gadget chosen upstream.


def _tdg(c: "zx.Circuit", target: int) -> None:
    """T-dagger: pyzx has no 'T*' shortcut key, so build it directly as a
    ZPhase gate with phase -pi/4."""
    c.add_gate("ZPhase", target, phase=Fraction(-1, 4))


def _add_toffoli(c: "zx.Circuit", ctrl1: int, ctrl2: int, target: int) -> None:
    """Standard 7-T-gate Toffoli decomposition (Nielsen & Chuang Fig 4.9)."""
    c.add_gate("H", target)
    c.add_gate("CNOT", ctrl2, target)
    _tdg(c, target)
    c.add_gate("CNOT", ctrl1, target)
    c.add_gate("T", target)
    c.add_gate("CNOT", ctrl2, target)
    _tdg(c, target)
    c.add_gate("CNOT", ctrl1, target)
    c.add_gate("T", ctrl2)
    c.add_gate("T", target)
    c.add_gate("H", target)
    c.add_gate("CNOT", ctrl1, ctrl2)
    c.add_gate("T", ctrl1)
    _tdg(c, target)
    c.add_gate("CNOT", ctrl1, ctrl2)


def _maj(c: "zx.Circuit", a: int, b: int, cq: int) -> None:
    c.add_gate("CNOT", cq, b)
    c.add_gate("CNOT", cq, a)
    _add_toffoli(c, a, b, cq)


def _uma(c: "zx.Circuit", a: int, b: int, cq: int) -> None:
    _add_toffoli(c, a, b, cq)
    c.add_gate("CNOT", cq, a)
    c.add_gate("CNOT", a, b)


def ripple_carry_adder(n_bits: int) -> "zx.Circuit":
    """n-bit Cuccaro ripple-carry adder. Qubit layout:
    q[0]         = carry-in / carry-out
    q[1 .. n]    = register A (a_0 .. a_{n-1}), overwritten with sum
    q[n+1 .. 2n] = register B (b_0 .. b_{n-1})
    Total qubits = 2n + 1."""
    n_qubits = 2 * n_bits + 1
    c = zx.Circuit(n_qubits)

    carry = 0
    a = list(range(1, n_bits + 1))
    b = list(range(n_bits + 1, 2 * n_bits + 1))

    for i in range(n_bits):
        cq = carry if i == 0 else a[i - 1]
        _maj(c, a[i], b[i], cq)
    for i in reversed(range(n_bits)):
        cq = carry if i == 0 else a[i - 1]
        _uma(c, a[i], b[i], cq)

    return c


def make_adder_generator(min_bits: int, max_bits: int, exclude_bits: Tuple[int, ...] = (),
                          seed: Optional[int] = None) -> Callable[[], "zx.Circuit"]:
    """Returns a generator sampling a random bit-width in [min_bits, max_bits]
    (excluding `exclude_bits`, e.g. a held-out interpolation width) on every
    call, and building a fresh ripple-carry adder at that width."""
    rng = random.Random(seed)
    valid_widths = [n for n in range(min_bits, max_bits + 1) if n not in exclude_bits]
    assert valid_widths, "No valid bit-widths left after exclusion."

    def _generate() -> "zx.Circuit":
        n_bits = rng.choice(valid_widths)
        return ripple_carry_adder(n_bits)

    return _generate


def make_fixed_adder_generator(n_bits: int) -> Callable[[], "zx.Circuit"]:
    """Deterministic single-bit-width generator, used for evaluation."""
    def _generate() -> "zx.Circuit":
        return ripple_carry_adder(n_bits)
    return _generate
