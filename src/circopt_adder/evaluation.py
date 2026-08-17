"""Evaluation protocol: run trained agents and baselines on held-out circuits.

Migrated from Section 11 of the dissertation notebook. Runs both trained agents
(deterministic, argmax policy -- no exploration at eval time) plus the deterministic
baselines on a set of evaluation circuits.

For the agents, the policy is run until STOP or cfg.max_episode_steps, and the
**best circuit seen during the episode** is reported (not just the final one) --
Riu et al. found this matters a lot (Section 4.2), since a single bad step late in
an episode can undo earlier gains.
"""

from typing import Dict

import pandas as pd
import torch

from .baselines import baseline_basic_plus_full_reduce, baseline_full_reduce
from .config import DEVICE, Config
from .env import ZXOptEnv
from .model import ActorCriticGNN
from .zx_utils import gate_count, t_count, two_qubit_gate_count


def run_agent_eval(policy: ActorCriticGNN, circuit: "object", cfg: Config, deterministic: bool = True):
    env = ZXOptEnv(lambda: circuit, cfg)
    obs, _ = env.reset()
    for _ in range(cfg.max_episode_steps):
        with torch.no_grad():
            obs_dev = obs.to(DEVICE)
            logits, _ = policy.forward(obs_dev)
            if deterministic:
                action_idx = int(torch.argmax(logits).item())
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action_idx = dist.sample().item()
        next_obs, reward, terminated, truncated, info = env.step(action_idx)
        if terminated or truncated:
            break
        obs = next_obs
    return env.best_circuit, gate_count(env.best_circuit), t_count(env.best_circuit), two_qubit_gate_count(env.best_circuit)


def evaluate_all_methods(circuits: Dict[str, "object"], policies: Dict[str, ActorCriticGNN], cfg: Config) -> pd.DataFrame:
    """`circuits`: {label: zx.Circuit}. `policies`: {label: ActorCriticGNN}."""
    rows = []
    for circ_label, circuit in circuits.items():
        initial_gates = gate_count(circuit)
        initial_t = t_count(circuit)
        initial_2q = two_qubit_gate_count(circuit)

        for method_label, fn in [
            ("full_reduce", baseline_full_reduce),
            ("basic_opt+full_reduce", baseline_basic_plus_full_reduce),
        ]:
            try:
                c_opt = fn(circuit)
                rows.append({"circuit": circ_label, "method": method_label,
                              "initial_gates": initial_gates, "final_gates": gate_count(c_opt),
                              "initial_2q": initial_2q, "final_2q": two_qubit_gate_count(c_opt),
                              "initial_t": initial_t, "final_t": t_count(c_opt)})
            except Exception as e:
                rows.append({"circuit": circ_label, "method": method_label, "error": str(e)})

        for policy_label, policy in policies.items():
            best_circuit, final_gates, final_t, final_2q = run_agent_eval(policy, circuit, cfg)
            rows.append({"circuit": circ_label, "method": policy_label,
                         "initial_gates": initial_gates, "final_gates": final_gates,
                         "initial_2q": initial_2q, "final_2q": final_2q,
                         "initial_t": initial_t, "final_t": final_t})

    return pd.DataFrame(rows)
