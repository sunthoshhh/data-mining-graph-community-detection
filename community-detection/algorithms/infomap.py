"""
Infomap (Rosvall & Bergstrom, 2008). Treats communities as regions where a random walker
gets trapped for a long time. Finds the partition that gives the shortest description
(map equation) of the walker's path: good communities = walker rarely leaves them.
Implementation: the official infomap package (compiled C++).
"""
from infomap import Infomap

NAME = "Infomap"
PARAMS = {"markov_time": 1.0, "num_trials": 1, "two_level": True}
SWEEP = {"markov_time": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]}   # larger = bigger communities
VARIANTS = {
    "default": {},
    "hierarchical": {"two_level": False},   # ablation: allow nested modules, score the top level
    "trials_10": {"num_trials": 10},        # ablation: best of 10 restarts vs 1
}


def detect(g, seed, markov_time=1.0, num_trials=1, two_level=True):
    # Infomap needs seed >= 1, so shift by one
    flags = f"--silent --seed {seed + 1} --num-trials {num_trials} --markov-time {markov_time}"
    if two_level:
        flags += " --two-level"
    im = Infomap(flags)
    im.add_nodes(range(g.vcount()))
    for e in g.es:
        im.add_link(e.source, e.target)
    im.run()
    modules = im.get_modules(depth_level=1)    # node -> top-level module
    return [modules[v] for v in range(g.vcount())]
