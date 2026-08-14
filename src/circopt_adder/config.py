"""Experiment configuration.

TODO: migrate the Config dataclass from Section 2 of the dissertation notebook here
(circuit-generation parameters, environment settings, GNN architecture sizes, PPO
hyperparameters, and training-run settings).
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Placeholder -- see notebooks/ for the current working version of this class."""

    seed: int = 42
