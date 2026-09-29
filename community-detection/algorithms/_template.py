"""
TEMPLATE: copy this file to add or rewrite an algorithm. The harness only needs these 5 things.
"""

NAME = "MyAlgorithm"            # name shown in tables and figures

PARAMS = {}                     # default settings, e.g. {"resolution": 1.0}

SWEEP = {}                      # values to try in the parameter experiment,
                                # e.g. {"resolution": [0.25, 0.5, 1.0, 2.0]}. Leave {} if none.

VARIANTS = {"default": {}}      # ablation: named on/off switches for key components,
                                # e.g. {"default": {}, "no_refinement": {"refine": False}}


def detect(g, seed, **params):
    """
    g      : igraph.Graph (undirected, nodes 0..n-1). Do NOT read the labels.
    seed   : int, use it for ALL randomness so runs are reproducible.
    params : PARAMS, possibly overridden by SWEEP or VARIANTS.
    return : membership list, membership[i] = community id of node i
             (a list of node sets is also accepted)
    """
    raise NotImplementedError
