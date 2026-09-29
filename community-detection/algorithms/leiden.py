"""
Leiden (Traag et al., 2019). Louvain plus a REFINEMENT phase between moving and aggregation,
which guarantees every community is internally connected (Louvain does not).
  1. Fast local moving of nodes
  2. Refinement: split each community into well-connected sub-communities
  3. Aggregation based on the refined partition, repeat
Implementation: leidenalg with the modularity objective (RBConfiguration, resolution=1 is standard modularity).
"""
import leidenalg as la

NAME = "Leiden"
PARAMS = {"resolution": 1.0, "n_iterations": 2}   # 2 = leidenalg default
SWEEP = {"resolution": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]}
VARIANTS = {
    "default": {},                          # 2 passes (library default)
    "one_iteration": {"n_iterations": 1},   # ablation: a single pass
    "until_converged": {"n_iterations": -1},# ablation: keep going until nothing improves (much slower)
}


def detect(g, seed, resolution=1.0, n_iterations=2):
    part = la.find_partition(
        g, la.RBConfigurationVertexPartition,
        resolution_parameter=resolution, n_iterations=n_iterations, seed=seed,
    )
    return part.membership
