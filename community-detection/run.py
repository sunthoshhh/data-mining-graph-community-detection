"""
Run experiments for one algorithm.

  python run.py louvain smoke      # 10-second check that detect() works (do this first)
  python run.py louvain main       # all 9 graphs x 5 seeds  -> results/raw/main_louvain.csv
  python run.py louvain sweep      # parameter sensitivity   -> results/raw/sweep_louvain.csv
  python run.py louvain ablation   # component on/off tests  -> results/raw/ablation_louvain.csv
  python run.py louvain all        # everything above

Algorithms: louvain, leiden, infomap, sbm, spectral
Options:
  --skip-large      leave out the 50,000-node graph while testing
  --timeout=900     seconds before a single run is stopped and recorded as a timeout
"""
import sys

from harness import core

EXPERIMENTS = ["smoke", "main", "sweep", "ablation", "all"]


def cli():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2 or args[1] not in EXPERIMENTS:
        print(__doc__)
        sys.exit(1)
    algo, exp = args
    for a in sys.argv[1:]:
        if a.startswith("--timeout="):
            core.TIMEOUT_S = int(a.split("=")[1])
    graphs = [g for g in core.ALL_GRAPHS if not ("--skip-large" in sys.argv and g == "scale_n50000")]

    if exp in ("smoke", "all"):
        if not core.smoke(algo) and exp == "all":
            sys.exit("Fix detect() before running the full experiments.")
    if exp in ("main", "all"):
        print(f"\n== main: {algo} ==")
        core.main(algo, graphs)
    if exp in ("sweep", "all"):
        print(f"\n== sweep: {algo} ==")
        core.sweep(algo)
    if exp in ("ablation", "all"):
        print(f"\n== ablation: {algo} ==")
        core.ablation(algo)


if __name__ == "__main__":
    cli()
