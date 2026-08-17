"""Train Agent R (random Clifford+T training regime).

Migrated from Section 8 of the dissertation notebook. Replicates the Riu et al.
training distribution. Run this first as a sanity check: if this agent doesn't learn
to reduce gate count on random circuits at all, the bug is in zx_utils/env/model/ppo,
not in anything adder-specific, and it's worth debugging here (smaller, faster
episodes) before touching the adder generator.

Requires the package to be installed (`pip install -e .` from the repo root).

Usage:
    python scripts/train_agent_r.py [--smoke-test]
"""

import argparse

from circopt_adder.config import DEVICE, SEED, Config, set_seed
from circopt_adder.env import ZXOptEnv
from circopt_adder.generators import make_random_circuit_generator
from circopt_adder.model import ActorCriticGNN
from circopt_adder.ppo import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                         help="Drastically reduce total_timesteps before committing to a full run.")
    args = parser.parse_args()

    set_seed(SEED)
    print("Using device:", DEVICE)

    cfg = Config()
    if args.smoke_test:
        cfg.total_timesteps = 10_000
        cfg.log_interval = 1

    generator = make_random_circuit_generator(
        n_qubits=cfg.n_qubits, n_gates=cfg.n_gates_random, seed=SEED,
    )
    env = ZXOptEnv(generator, cfg)
    policy = ActorCriticGNN(cfg).to(DEVICE)

    train(env, policy, cfg, run_name="agent_R_random")


if __name__ == "__main__":
    main()
