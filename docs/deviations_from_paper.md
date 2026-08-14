# Deviations from Riu et al. / the original Circopt-RL-ZXCalc repo

This project builds on the environment and reward design of Riu, Nogue, Vilaplana,
Garcia-Saez and Estarellas (2025), "Reinforcement Learning Based Quantum Circuit
Optimization via ZX-Calculus" (Quantum, 9), and its accompanying code at
https://github.com/qilimanjaro-tech/Circopt-RL-ZXCalc. It is an independent,
reduced-scope reimplementation, not a port of that repo. This document tracks the
known differences, so they can be cited directly in the dissertation's methodology
and limitations sections.

## Action space

The original repo (gym_zx/envs/zx_env.py) implements six action types: local
complementation (LC), three pivot variants (interior pivot, pivot-boundary, and
pivot-gadget -- the latter two needed for non-Clifford/T-phase spiders), identity
removal with spider fusion (ID), and gadget fusion (GF), plus STOP. This project
currently implements only LC, plain interior pivoting, and STOP. Circuits containing
T gates (e.g. the ripple-carry adders used here, which need Toffolis) will not fully
reduce until the non-Clifford pivot variants and gadget fusion are added.

## Network architecture

The original repo (rl_agent.py) uses two separate GATv2-based GNNs: an actor network
(5 layers, 128 hidden channels, 17 input node features, edge dim 7) and a critic
network (5 layers, 128 hidden channels, 12 input node features, edge dim 2), with the
critic pooled via AttentionalAggregation. This project uses a single shared GATv2
trunk (5 layers, 32 channels) with linear policy/value heads and mean-pool critic
aggregation -- a smaller, simpler design inspired by rather than copied from the
paper's Fig. 4.

## Environment / baselines

Reward shaping (per-step gate-count reduction normalised by a fixed max_compression
of 10, plus a terminal bonus vs. deterministic baselines, max_episode_len = 35) is
reproduced faithfully. Two baseline routines had to be substituted due to pyzx version
drift (this project pins pyzx==0.10.5):

- the original repo's greedy_simp ("korbinian" baseline) does not exist in pyzx
  0.10.5 and is replaced with clifford_simp;
- the original repo's terminal-step baseline uses flow_2Q_simp + extract_simple
  (a cflow-based reduction); this project uses teleport_reduce at reset time instead.

These are not equivalent algorithms, so baseline numbers produced here are not
directly comparable to the paper's reported figures.

## Training loop

The original repo does not publish a PPO training script -- only the environment, the
agent network class, and an evaluation script (agent_test.py) that loads a pretrained
checkpoint and runs vectorised, batched rollouts. The PPO loop in this project
(src/circopt_adder/ppo.py) is an independent implementation, single-environment and
per-transition rather than batched/vectorised, and is correspondingly slower.

## New contributions (not in the original repo)

- The ripple-carry adder circuit generator (generators.ripple_carry_adder) and the
  random-vs-structured training ablation (Agent R vs. Agent A) are original to this
  dissertation.
- A configurable reward metric (total_gates / t_count in addition to the paper's
  hardcoded twoqubits).
