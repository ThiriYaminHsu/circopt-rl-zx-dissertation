"""PPO training loop: rollout collection, GAE, clipped policy/value update.

TODO: migrate Transition, collect_rollout, compute_gae, ppo_update, and train from
Section 7 of the dissertation notebook.

Note: the original repo does not publish a training script -- only the environment,
the agent network class, and an evaluation script (agent_test.py) that loads a
pretrained checkpoint. This training loop is an independent implementation,
single-environment and per-transition rather than batched/vectorised.
"""
