"""Train Agent A (adder-specific training regime).

Migrated from Section 9 of the dissertation notebook. Identical config to Agent R
except for the generator: bit-width is resampled every episode from
[adder_min_bits, adder_max_bits], excluding the interpolation held-out width(s)
defined in cfg.adder_heldout_interp_bits.

Requires the package to be installed (`pip install -e .` from the repo root).

Usage:
    python scripts/train_agent_a.py [--smoke-test]
"""

import argparse

from circopt_adder.config import DEVICE, SEED, Config, set_seed
from circopt_adder.env import ZXOptEnv
from circopt_adder.generators import make_adder_generator
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

    generator = make_adder_generator(
        min_bits=cfg.adder_min_bits,
        max_bits=cfg.adder_max_bits,
        exclude_bits=cfg.adder_heldout_interp_bits,
        seed=SEED,
    )
    env = ZXOptEnv(generator, cfg)
    policy = ActorCriticGNN(cfg).to(DEVICE)

    train(env, policy, cfg, run_name="agent_A_adder")


if __name__ == "__main__":
    main()
