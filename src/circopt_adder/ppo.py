"""PPO training loop: rollout collection, GAE, clipped policy/value update.

Migrated from Section 7 of the dissertation notebook. Single-environment rollout
collection for clarity. Loss terms follow Eq. 10-13 of Riu et al.: clipped policy
loss, clipped value loss, entropy bonus.

Because each episode's graph has a *different number of nodes* (action count changes
every step), batching rollouts requires padding or torch_geometric.data.Batch -- this
loop keeps things simple by doing the PPO update per-transition rather than batching
across transitions with different action-space sizes. This is slower than the paper's
setup but far easier to get correct first; vectorise once it's verified to learn
anything at all.

Note: the original repo does not publish a training script -- only the environment,
the agent network class, and an evaluation script (agent_test.py) that loads a
pretrained checkpoint. This training loop is an independent implementation,
single-environment and per-transition rather than batched/vectorised.

NOTE on scaling up: once training visibly reduces gate/T-count on toy circuits, the
main speed win is batching the PPO update over a torch_geometric.data.Batch of
same-epoch transitions instead of looping `for i in batch_idx`, and running
cfg.n_envs copies of ZXOptEnv in parallel via e.g. gymnasium.vector + manual stepping
(standard AsyncVectorEnv won't handle variable-sized Data observations out of the box).
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from .config import DEVICE, Config
from .env import ZXOptEnv
from .model import ActorCriticGNN


@dataclass
class Transition:
    data: Data
    action_idx: int
    log_prob: float
    value: float
    reward: float
    done: bool


def collect_rollout(env: ZXOptEnv, policy: ActorCriticGNN, n_steps: int) -> List[Transition]:
    transitions = []
    obs, _ = env.reset()
    for _ in range(n_steps):
        with torch.no_grad():
            obs_dev = obs.to(DEVICE)
            action_idx, log_prob, value = policy.act(obs_dev)
        next_obs, reward, terminated, truncated, info = env.step(action_idx)
        done = terminated or truncated
        transitions.append(Transition(obs, action_idx, log_prob.item(), value.item(), reward, done))
        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs
    return transitions


def compute_gae(transitions: List[Transition], gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    rewards = np.array([t.reward for t in transitions])
    values = np.array([t.value for t in transitions] + [0.0])
    dones = np.array([t.done for t in transitions])

    advantages = np.zeros_like(rewards)
    last_adv = 0.0
    for t in reversed(range(len(transitions))):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv
    returns = advantages + values[:-1]
    return advantages, returns


def ppo_update(policy: ActorCriticGNN, optimizer: torch.optim.Optimizer,
                transitions: List[Transition], advantages: np.ndarray, returns: np.ndarray,
                cfg: Config) -> dict:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    idxs = np.arange(len(transitions))

    stats = {"policy_loss": [], "value_loss": [], "entropy": []}

    for _ in range(cfg.n_epochs):
        np.random.shuffle(idxs)
        for start in range(0, len(idxs), cfg.minibatch_size):
            batch_idx = idxs[start:start + cfg.minibatch_size]

            policy_losses, value_losses, entropies = [], [], []
            for i in batch_idx:
                tr = transitions[i]
                data = tr.data.to(DEVICE)
                action_t = torch.tensor(tr.action_idx, device=DEVICE)
                log_prob, entropy, value = policy.evaluate_actions(data, action_t)

                old_log_prob = torch.tensor(tr.log_prob, device=DEVICE)
                adv = torch.tensor(advantages[i], device=DEVICE, dtype=torch.float32)
                ret = torch.tensor(returns[i], device=DEVICE, dtype=torch.float32)

                ratio = torch.exp(log_prob - old_log_prob)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2)
                value_loss = F.mse_loss(value, ret)

                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropies.append(entropy)

            policy_loss = torch.stack(policy_losses).mean()
            value_loss = torch.stack(value_losses).mean()
            entropy = torch.stack(entropies).mean()

            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.item())

    return {k: float(np.mean(v)) for k, v in stats.items()}


def train(env: ZXOptEnv, policy: ActorCriticGNN, cfg: Config, run_name: str) -> pd.DataFrame:
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    n_updates = cfg.total_timesteps // cfg.n_steps
    log_rows = []

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

    for update in range(n_updates):
        t0 = time.time()
        transitions = collect_rollout(env, policy, cfg.n_steps)
        advantages, returns = compute_gae(transitions, cfg.gamma, cfg.gae_lambda)

        # --- instrumentation ---
        failure_rate = env.n_extraction_failures / max(env.n_steps_total, 1)
        print(f"    [diag] adv mean={advantages.mean():.4f} std={advantages.std():.4f} "
              f"| extraction failure rate (cumulative)={failure_rate:.1%}")

        stats = ppo_update(policy, optimizer, transitions, advantages, returns, cfg)

        log_rows.append({
            "update": update,
            "timesteps": (update + 1) * cfg.n_steps,
            "mean_reward": float(np.mean([t.reward for t in transitions])),
            "policy_loss": stats["policy_loss"],
            "value_loss": stats["value_loss"],
            "entropy": stats["entropy"],
            "wall_time_s": time.time() - t0,
        })

        if update % cfg.log_interval == 0:
            print(f"[{run_name}] update {update}/{n_updates} "
                  f"reward={log_rows[-1]['mean_reward']:.3f} "
                  f"entropy={stats['entropy']:.3f} "
                  f"({log_rows[-1]['wall_time_s']:.1f}s)")

    df = pd.DataFrame(log_rows)
    df.to_csv(Path(cfg.log_dir) / f"{run_name}_train_log.csv", index=False)
    torch.save(policy.state_dict(), Path(cfg.checkpoint_dir) / f"{run_name}.pt")
    return df
