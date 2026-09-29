"""
Bayesian Stochastic Block Model (Peixoto, 2014/2017). A generative, statistical approach:
assume the graph was produced by an SBM, where the chance of an edge depends only on the
two nodes' groups, and find the partition that best COMPRESSES the graph (minimum
description length = Bayesian posterior mode). The number of groups K is chosen automatically,
because adding groups only pays off if it explains the edges well enough to justify the cost.
Degree correction lets nodes in the same group have very different degrees (hubs).
Implementation: graph-tool's minimize_blockmodel_dl (compiled C++).

Install: graph-tool is NOT on pip. See README "Installing graph-tool".
"""
import numpy as np

NAME = "Bayesian SBM"
PARAMS = {"deg_corr": True, "nested": False}
SWEEP = {}     # non-parametric: K is inferred, so there is nothing to tune. Discuss this!
VARIANTS = {
    "default": {},                        # degree-corrected, flat
    "no_deg_corr": {"deg_corr": False},   # ablation: plain SBM, should struggle with hubs
    "nested": {"nested": True},           # ablation: hierarchical prior, bottom level scored
}


def detect(g, seed, deg_corr=True, nested=False):
    import graph_tool.all as gt       # imported here so the other algorithms work without it
    gt.openmp_set_num_threads(1)      # single thread so seeds are reproducible
    gt.seed_rng(seed)
    np.random.seed(seed)
    gg = gt.Graph(directed=False)
    gg.add_vertex(g.vcount())
    gg.add_edge_list(g.get_edgelist())
    if nested:
        state = gt.minimize_nested_blockmodel_dl(gg, state_args=dict(deg_corr=deg_corr))
        blocks = state.get_levels()[0].get_blocks()
    else:
        state = gt.minimize_blockmodel_dl(gg, state_args=dict(deg_corr=deg_corr))
        blocks = state.get_blocks()
    return [int(blocks[v]) for v in gg.vertices()]
