"""Load trained checkpoints and run the full evaluation + plotting protocol.

Migrated from Sections 11-12 of the dissertation notebook. Runs both trained agents
(deterministic, argmax policy) plus the two deterministic pyzx baselines on:
  1. A batch of random circuits (sanity check against Riu et al.'s own numbers)
  2. Held-out interpolation adder bit-width(s)
  3. Extrapolation adder bit-widths never seen in training
  4. The original paper's benchmark circuits (Table 2, Riu et al.)

Requires the package to be installed (`pip install -e .` from the repo root) and both
agents already trained (scripts/train_agent_r.py, scripts/train_agent_a.py).

Usage:
    python scripts/evaluate.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # to import the sibling benchmarks/ package

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from benchmarks.fetch_paper_circuits import load_all as load_paper_benchmark_circuits
from circopt_adder.config import DEVICE, SEED, Config
from circopt_adder.evaluation import evaluate_all_methods
from circopt_adder.generators import make_random_circuit_generator, ripple_carry_adder
from circopt_adder.model import ActorCriticGNN


def _load_policy(cfg: Config, checkpoint_path: Path) -> ActorCriticGNN:
    policy = ActorCriticGNN(cfg).to(DEVICE)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    policy.eval()
    return policy


def build_eval_circuits(cfg: Config) -> dict:
    eval_circuits = {}

    # 1. random sanity-check circuits
    random_eval_gen = make_random_circuit_generator(cfg.n_qubits, cfg.n_gates_random, seed=SEED + 1)
    for i in range(5):
        eval_circuits[f"random_{i}"] = random_eval_gen()

    # 2. interpolation adders (held out bit-width, within training range)
    for n_bits in cfg.adder_heldout_interp_bits:
        eval_circuits[f"adder_interp_{n_bits}bit"] = ripple_carry_adder(n_bits)

    # 3. extrapolation adders (bit-widths never seen in training)
    for n_bits in cfg.adder_heldout_extrap_bits:
        eval_circuits[f"adder_extrap_{n_bits}bit"] = ripple_carry_adder(n_bits)

    # 4. original paper benchmark circuits (Riu et al. Table 2)
    paper_benchmark_circuits = load_paper_benchmark_circuits()
    print(f"Loaded {len(paper_benchmark_circuits)} paper benchmark circuits")
    eval_circuits.update(paper_benchmark_circuits)

    return eval_circuits


def _categorize(s: str) -> str:
    if s.startswith("random"):
        return "random"
    if s.startswith("adder"):
        return "adder"
    if s.startswith("paper"):
        return "paper"
    return "other"


def _pct_reduction(row, init_col, final_col):
    init = row[init_col]
    fin = row.get(final_col)
    return np.nan if pd.isna(fin) else (100.0 * (init - fin) / init if init else np.nan)


def build_result_tables(results_df: pd.DataFrame) -> dict:
    results_df = results_df.copy()
    results_df["category"] = results_df["circuit"].apply(_categorize)
    results_df["gate_reduction_pct"] = results_df.apply(lambda r: _pct_reduction(r, "initial_gates", "final_gates"), axis=1)
    results_df["twoq_reduction_pct"] = results_df.apply(lambda r: _pct_reduction(r, "initial_2q", "final_2q"), axis=1)
    results_df["t_reduction_pct"] = results_df.apply(lambda r: _pct_reduction(r, "initial_t", "final_t"), axis=1)

    display_cols = [c for c in ["circuit", "method", "initial_gates", "final_gates", "gate_reduction_pct",
                                 "initial_2q", "final_2q", "twoq_reduction_pct",
                                 "initial_t", "final_t", "t_reduction_pct", "error"]
                     if c in results_df.columns]

    def make_table(cat):
        return results_df[results_df["category"] == cat][display_cols].sort_values(["circuit", "method"]).reset_index(drop=True)

    return {
        "all": results_df,
        "random": make_table("random"),
        "adder": make_table("adder"),
        "paper": make_table("paper"),
    }


def plot_comparison(results_df: pd.DataFrame, cfg: Config, out_dir: Path) -> None:
    metric_col = "final_t" if cfg.reward_mode == "t_count" else "final_gates"
    init_col = "initial_t" if cfg.reward_mode == "t_count" else "initial_gates"

    plot_df = results_df.copy()
    plot_df["reduction_pct"] = plot_df.apply(lambda r: _pct_reduction(r, init_col, metric_col), axis=1)
    plot_df["eval_set"] = plot_df["circuit"].apply(
        lambda s: "random" if s.startswith("random") else ("paper" if s.startswith("paper") else ("interpolation" if "interp" in s else "extrapolation")))

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="eval_set", y="reduction_pct", hue="method", ax=ax)
    ax.set_ylabel(f"% reduction in {'T-count' if cfg.reward_mode == 't_count' else 'gate count'}")
    ax.set_title("Agent R (random-trained) vs Agent A (adder-trained) vs deterministic baselines")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_barplot.png", dpi=150)
    plt.close(fig)

    # Bit-width scaling view for the adder-only rows (interpolation + extrapolation)
    adder_rows = plot_df[plot_df["circuit"].str.startswith("adder")].copy()
    adder_rows["n_bits"] = adder_rows["circuit"].str.extract(r"(\d+)bit").astype(int)

    fig, ax = plt.subplots(figsize=(9, 5))
    for method in adder_rows["method"].unique():
        sub = adder_rows[adder_rows["method"] == method].sort_values("n_bits")
        ax.plot(sub["n_bits"], sub["reduction_pct"], marker="o", label=method)
    ax.axvline(x=cfg.adder_max_bits + 0.5, color="grey", linestyle="--", alpha=0.5,
               label="train/extrapolation boundary")
    ax.set_xlabel("adder bit-width")
    ax.set_ylabel("% reduction")
    ax.set_title("Scaling behaviour: interpolation vs extrapolation")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "adder_scaling.png", dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = Config()
    checkpoint_dir = REPO_ROOT / cfg.checkpoint_dir
    log_dir = REPO_ROOT / cfg.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    policies = {
        "agent_R_random": _load_policy(cfg, checkpoint_dir / "agent_R_random.pt"),
        "agent_A_adder": _load_policy(cfg, checkpoint_dir / "agent_A_adder.pt"),
    }

    eval_circuits = build_eval_circuits(cfg)
    results_df = evaluate_all_methods(eval_circuits, policies, cfg)
    results_df.to_csv(log_dir / "evaluation_results.csv", index=False)

    tables = build_result_tables(results_df)
    print("=== Random sanity-check circuits ===")
    print(tables["random"].to_string(index=False))
    print("\n=== Adder circuits (interpolation + extrapolation) ===")
    print(tables["adder"].to_string(index=False))
    print("\n=== Paper benchmark circuits (Riu et al. Table 2) ===")
    print(tables["paper"].to_string(index=False))

    plot_comparison(results_df, cfg, log_dir)
    print(f"\nSaved evaluation_results.csv and plots to {log_dir}")


if __name__ == "__main__":
    main()
