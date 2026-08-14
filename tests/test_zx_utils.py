"""Sanity tests for the Clifford-only ZX rewrite rules.

TODO: once src/circopt_adder/zx_utils.py is populated, add tests that check
apply_local_complementation and apply_pivot preserve the circuit's unitary (e.g. by
comparing zx.compare_tensors before/after on small random Clifford diagrams), and
that find_feasible_lc / find_feasible_pivots only return genuinely-applicable actions.
"""

import pytest


def test_placeholder():
    """Remove once real tests are added."""
    assert True
