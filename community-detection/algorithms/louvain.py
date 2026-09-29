"""
Louvain (Blondel et al., 2008). Greedy modularity optimisation in two repeated phases:
  1. Local moving: move each node to the neighbouring community that most increases modularity
  2. Aggregation: collapse each community into a super-node and repeat on the smaller graph
Implementation: igraph's community_multilevel (compiled C, so runtimes are comparable).
"""
import random

NAME = "Louvain"
PARAMS = {"resolution": 1.0, "first_level_only": False}
SWEEP = {"resolution": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]}
VARIANTS = {
    "default": {},
    "first_level_only": {"first_level_only": True},   # ablation: phase 1 only, no aggregation
}


def detect(g, seed, resolution=1.0, first_level_only=False):
    random.seed(seed)   # igraph draws its randomness (node order) from Python's random module
    if first_level_only:
        levels = g.community_multilevel(resolution=resolution, return_levels=True)
        return levels[0].membership
    return g.community_multilevel(resolution=resolution).membership
