"""GATv2 actor-critic network.

Migrated from Section 6 of the dissertation notebook. Follows Riu et al. Section
3.2-3.3.1: 5 GATv2Conv message-passing layers, 32 channels, action logits read off
the action-node subset of the final layer's output, then masked-softmaxed over only
the nodes present in the current graph (masking is implicit -- the graph literally
doesn't contain infeasible-action nodes).

Note (see docs/deviations_from_paper.md): this is a single shared 32-channel GATv2
trunk with linear policy/value heads and mean-pool critic aggregation -- a
simplification of the original repo's rl_agent.py, which uses two separate
128-channel actor/critic GNNs with attentional aggregation pooling for the critic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool

from .config import Config


class ActorCriticGNN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        C = cfg.gnn_channels

        self.input_proj = nn.Linear(cfg.actor_node_feat_dim, C)
        self.convs = nn.ModuleList([
            GATv2Conv(C, C, edge_dim=cfg.edge_feat_dim, add_self_loops=True)
            for _ in range(cfg.gnn_layers)
        ])
        self.policy_head = nn.Linear(C, 1)   # one logit per node; we slice to action nodes
        self.value_proj = nn.Linear(C, C)
        self.value_head = nn.Linear(C, 1)

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = F.relu(self.input_proj(x))
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_attr=edge_attr))

        action_logits = self.policy_head(h).squeeze(-1)
        action_logits = action_logits[data.action_node_mask]

        if hasattr(data, "batch") and data.batch is not None:
            pooled = global_mean_pool(self.value_proj(h), data.batch)
        else:
            pooled = self.value_proj(h).mean(dim=0, keepdim=True)
        value = self.value_head(F.relu(pooled)).squeeze(-1)

        return action_logits, value

    def act(self, data: Data):
        """Single-environment inference: sample an action, return
        (action_idx, log_prob, value)."""
        logits, value = self.forward(data)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze()

    def evaluate_actions(self, data: Data, action_idx: torch.Tensor):
        logits, value = self.forward(data)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy()
        return log_prob, entropy, value.squeeze()
