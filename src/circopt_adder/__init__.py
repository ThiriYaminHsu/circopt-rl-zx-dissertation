"""circopt_adder: RL agents for ZX-calculus circuit optimisation.

Dissertation project by Thiri Yamin Hsu (Tori), MSc Applied AI, WMG, University of Warwick.

Built on the environment/reward design of Riu, Nogue, Vilaplana, Garcia-Saez and Estarellas
(2025), "Reinforcement Learning Based Quantum Circuit Optimization via ZX-Calculus"
(Quantum, 9). Code: https://github.com/qilimanjaro-tech/Circopt-RL-ZXCalc

This package is an independent, reduced-scope reimplementation (Clifford-only action
space, single shared GATv2 actor-critic backbone) -- see docs/deviations_from_paper.md
for the full list of differences from the original repo.
"""

__version__ = "0.1.0"

from .config import Config, DEVICE, SEED, set_seed  # noqa: E402,F401
from .model import ActorCriticGNN  # noqa: E402,F401
from .env import ZXOptEnv  # noqa: E402,F401
