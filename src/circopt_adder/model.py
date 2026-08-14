"""GATv2 actor-critic network.

TODO: migrate ActorCriticGNN from Section 6 of the dissertation notebook.

Note (see docs/deviations_from_paper.md): this is a single shared 32-channel GATv2
trunk with linear policy/value heads and mean-pool critic aggregation -- a
simplification of the original repo's rl_agent.py, which uses two separate
128-channel actor/critic GNNs with attentional aggregation pooling for the critic.
"""
