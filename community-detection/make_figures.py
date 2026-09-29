"""
Build every figure and summary table from results/raw/*.csv.
Run after everyone has pushed their CSVs:   python make_figures.py

Every figure uses the same colour and marker per algorithm, so they can go straight into the report.
Figures that need data nobody has produced yet are skipped with a message.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from harness import core

RAW = os.path.join(core.RESULTS, "raw")
FIG = os.path.join(core.ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# fixed identity per algorithm (colour-blind checked palette + distinct markers)
STYLE = {
    "Louvain":           dict(color="#2a78d6", marker="o"),
    "Leiden":            dict(color="#eb6834", marker="s"),
    "Infomap":           dict(color="#1baf7a", marker="^"),
    "Bayesian SBM":      dict(color="#eda100", marker="D"),
    "Spectral":          dict(color="#e87ba4", marker="v"),
}
KEYS = [("louvain", "Louvain"), ("leiden", "Leiden"), ("infomap", "Infomap"),
        ("sbm", "Bayesian SBM"), ("spectral", "Spectral")]
PARAM_LABEL = {"resolution": "Resolution γ", "markov_time": "Markov time", "k_factor": "K ÷ true K"}
ORDER = list(STYLE)
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 6,
})


def read(prefix):
    files = glob.glob(os.path.join(RAW, f"{prefix}_*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df[df["status"] == "ok"] if "status" in df else df


def algos(df):
    return [a for a in ORDER if a in set(df["algorithm"])] + sorted(set(df["algorithm"]) - set(ORDER))


def style(a):
    return STYLE.get(a, dict(color=MUTED, marker="x"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/" + name)


def email_mu():
    p = os.path.join(core.DATA, "email", "summary.csv")
    return float(pd.read_csv(p)["mu_actual"][0]) if os.path.exists(p) else None


# --------------------------------------------------------------- figures ---

def fig1_accuracy_vs_mu(main):
    d = main[main.experiment == "mixing"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3), sharey=True)
    for ax, metric in zip(axes, ["nmi", "ari"]):
        for a in algos(d):
            s = d[d.algorithm == a].groupby("mu_target")[metric].agg(["mean", "std"])
            ax.errorbar(s.index, s["mean"], yerr=s["std"], label=a, capsize=3, **style(a))
        ax.set(xlabel="Mixing μ (higher = harder)", ylabel=metric.upper(), ylim=(-0.05, 1.05),
               title=f"{metric.upper()} vs difficulty")
    axes[0].legend(loc="lower left")
    save(fig, "fig1_accuracy_vs_mu.png")


def fig2_runtime_vs_n(main):
    d = main[(main.graph.str.startswith("mix_mu0.4")) | (main.experiment == "scale")]
    if d.n.nunique() < 2:
        return print("  skip fig2: needs the scale graphs")
    fig, ax = plt.subplots(figsize=(4, 3))
    for a in algos(d):
        s = d[d.algorithm == a].groupby("n")["runtime_s"].mean()
        ax.plot(s.index, s.values, label=a, **style(a))
    ax.set(xscale="log", yscale="log", xlabel="Nodes n (μ = 0.4)", ylabel="Runtime (s)",
           title="Runtime vs graph size")
    ax.legend()
    save(fig, "fig2_runtime_vs_n.png")


def fig3_stability(stab):
    graphs = [g for g in ["mix_mu0.2_s1", "mix_mu0.4_s1", "mix_mu0.6_s1", "email"] if g in set(stab.graph)]
    fig, ax = plt.subplots(figsize=(6, 3))
    al = algos(stab)
    w = 0.8 / len(al)
    for i, a in enumerate(al):
        s = stab[stab.algorithm == a].set_index("graph").reindex(graphs)
        xs = [j + (i - (len(al) - 1) / 2) * w for j in range(len(graphs))]
        ax.bar(xs, s.between_run_nmi_mean, width=w * 0.9, label=a, color=style(a)["color"])
    ax.set_xticks(range(len(graphs)), [g.replace("mix_", "").replace("_s1", "") for g in graphs])
    ax.set(ylabel="NMI between runs", ylim=(0, 1.05),
           title="Stability: do different seeds give the same answer?")
    ax.legend(ncol=2, loc="lower left")
    save(fig, "fig3_stability.png")


def fig4_modularity_gap(main):
    d = main[main.experiment == "mixing"].copy()
    d["gap"] = d.modularity - d.modularity_truth
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    for a in algos(d):
        s = d[d.algorithm == a].groupby("mu_target")["gap"].mean()
        ax.plot(s.index, s.values, label=a, **style(a))
    ax.set(xlabel="Mixing μ", ylabel="Q(found) − Q(true)",
           title="Modularity of found vs true partition")
    ax.text(0.02, 0.97, "above 0: algorithm's answer scores higher\nthan the truth on its own objective",
            transform=ax.transAxes, va="top", fontsize=7, color=MUTED)
    ax.legend(loc="lower left")
    save(fig, "fig4_modularity_gap.png")


def fig5_sweep(sweep):
    """One row per tunable parameter, one column per graph. 5a = accuracy, 5b = over/under-splitting."""
    graphs = [g for g in core.SWEEP_GRAPHS if g in set(sweep.graph)]
    params = [p for p in PARAM_LABEL if p in set(sweep.sweep_param)]
    for suffix, ylabel in [("a_nmi", "NMI"), ("b_split", "Communities found ÷ true")]:
        fig, axes = plt.subplots(len(params), len(graphs), figsize=(3.6 * len(graphs), 2.4 * len(params)),
                                 squeeze=False)
        for i, pname in enumerate(params):
            for j, g in enumerate(graphs):
                ax = axes[i, j]
                d = sweep[(sweep.graph == g) & (sweep.sweep_param == pname)]
                for a in algos(d):
                    s_ = d[d.algorithm == a].groupby("sweep_value").agg(
                        nmi=("nmi", "mean"), found=("n_found", "mean"), true=("n_true", "first"))
                    y = s_.nmi if suffix == "a_nmi" else s_.found / s_.true
                    ax.plot(s_.index, y, label=a, **style(a))
                ax.axvline(1.0, color=MUTED, lw=1, ls=":")       # the default setting
                if suffix == "a_nmi":
                    ax.set(ylim=(0, 1.05))
                else:
                    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
                    ax.set(yscale="log")
                ax.set(xlabel=PARAM_LABEL[pname], ylabel=ylabel, title=g if i == 0 else "")
                if j == 0:
                    ax.legend(fontsize=7)
        save(fig, f"fig5{suffix}_parameter_sweep.png")


def fig6_email_bridge(main):
    mu = email_mu()
    em = main[main.graph == "email"]
    if mu is None or em.empty:
        return print("  skip fig6: needs email results")
    d = main[main.experiment == "mixing"]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for a in algos(d):
        s = d[d.algorithm == a].groupby("mu_target")["nmi"].mean()
        ax.plot(s.index, s.values, label=a, alpha=0.9, **style(a))
        e = em[em.algorithm == a]["nmi"].mean()
        ax.scatter([mu], [e], s=90, facecolor="white", edgecolor=style(a)["color"], linewidth=2,
                   marker=style(a)["marker"], zorder=5)
    ax.axvline(mu, color=MUTED, ls=":", lw=1)
    ax.text(mu, 1.02, f" email-Eu-core μ = {mu:.2f}", fontsize=7, color=MUTED)
    ax.set(xlabel="Mixing μ", ylabel="NMI", ylim=(0, 1.08),
           title="Does LFR predict the real network?")
    ax.legend(loc="lower left", fontsize=7)
    save(fig, "fig6_email_bridge.png")


def fig7_ablation(abl):
    al = [a for a in algos(abl) if abl[abl.algorithm == a].variant.nunique() > 1]
    if not al:
        return
    fig, axes = plt.subplots(1, len(al), figsize=(3.6 * len(al), 3), squeeze=False, sharey=True)
    for ax, a in zip(axes[0], al):
        d = abl[abl.algorithm == a]
        piv = d.groupby(["graph", "variant"])["nmi"].mean().unstack()
        piv = piv.reindex([g for g in core.ABLATION_GRAPHS if g in piv.index])
        shades = ["#184f95", "#2a78d6", "#6da7ec", "#b7d3f6"]  # one hue, dark to light steps
        piv.plot.bar(ax=ax, color=shades[: piv.shape[1]], width=0.75, rot=0)
        ax.set_xticklabels([g.replace("mix_", "").replace("_s1", "") for g in piv.index])
        ax.set(title=a, xlabel="", ylabel="NMI", ylim=(0, 1.05))
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    save(fig, "fig7_ablation.png")


def fig8_disconnected(main):
    d = main[main.graph.isin(["mix_mu0.2_s1", "mix_mu0.4_s1", "mix_mu0.6_s1", "scale_n50000", "email"])]
    piv = d.groupby(["graph", "algorithm"])["frac_disconnected"].mean().unstack()
    piv = piv[[a for a in ORDER if a in piv.columns]]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    piv.plot.bar(ax=ax, color=[style(a)["color"] for a in piv.columns], width=0.8, rot=0)
    ax.set(ylabel="Share of communities\nthat are disconnected", xlabel="",
           title="Badly connected communities (Leiden's refinement should give 0)")
    save(fig, "fig8_disconnected.png")


def fig9_recovery_by_size(graph="mix_mu0.4_s1"):
    """Failure analysis: which TRUE communities get recovered, by size (seed 0 partitions)."""
    g, truth = core.load(graph)
    true_groups = {}
    for v, c in enumerate(truth):
        true_groups.setdefault(c, set()).add(v)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    any_data = False
    for key, a in KEYS:
        p = os.path.join(core.RESULTS, "partitions", key, f"{graph}_seed0.txt")
        if not os.path.exists(p):
            continue
        pred = pd.read_csv(p, sep=" ", header=None)[1].tolist()
        found = {}
        for v, c in enumerate(pred):
            found.setdefault(c, set()).add(v)
        sizes, best = [], []
        for comm in true_groups.values():
            cands = {pred[v] for v in comm}
            best.append(max(len(comm & found[c]) / len(comm | found[c]) for c in cands))
            sizes.append(len(comm))
        ax.scatter(sizes, best, label=a, alpha=0.8, s=22, color=style(a)["color"], marker=style(a)["marker"])
        any_data = True
    if not any_data:
        return plt.close(fig)
    ax.set(xlabel="True community size", ylabel="Best Jaccard match", ylim=(-0.05, 1.05),
           title=f"Which communities are recovered? ({graph})")
    ax.legend(fontsize=7)
    save(fig, f"fig9_recovery_by_size_{graph}.png")


def fig10_email_confusion():
    """Failure analysis on email: true departments (rows) vs found communities (columns)."""
    if not os.path.exists(os.path.join(core.DATA, "email", "labels.txt")):
        return
    _, truth = core.load("email")
    keys = [(k, a) for k, a in KEYS if os.path.exists(os.path.join(core.RESULTS, "partitions", k, "email_seed0.txt"))]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.4), squeeze=False)
    dept_order = pd.Series(truth).value_counts().index
    for ax, (k, a) in zip(axes[0], keys):
        pred = pd.read_csv(os.path.join(core.RESULTS, "partitions", k, "email_seed0.txt"), sep=" ", header=None)[1]
        tab = pd.crosstab(pd.Series(truth, name="dept"), pred.rename("found")).reindex(dept_order)
        tab = tab[tab.sum().sort_values(ascending=False).index[:25]]          # 25 largest found
        tab = tab.div(tab.sum(axis=1), axis=0)                                # row share
        ax.imshow(tab.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set(title=a, xlabel="Found community (largest 25)", ylabel="True department (largest first)",
               xticks=[], yticks=[])
    save(fig, "fig10_email_confusion.png")


PARTS = [("part1_louvain_leiden", ["Louvain", "Leiden"]), ("part2_infomap", ["Infomap"]),
         ("part3_sbm", ["Bayesian SBM"]), ("part4_spectral", ["Spectral"])]


def per_part_figures(main, sweep, abl):
    """One figure per report part (section 5): (a) this method vs the others on LFR,
    (b) its parameter sweep, (c) its ablation. Missing panels are left out."""
    for fname, names in PARTS:
        if not set(names) & set(main.algorithm):
            continue
        has_sweep = sweep is not None and set(names) & set(sweep.algorithm)
        has_abl = abl is not None and any(abl[abl.algorithm == a].variant.nunique() > 1 for a in names)
        panels = 1 + bool(has_sweep) + (sum(abl[abl.algorithm == a].variant.nunique() > 1 for a in names) if has_abl else 0)
        fig, axes = plt.subplots(1, panels, figsize=(3.6 * panels, 3), squeeze=False)
        axes = list(axes[0])

        ax = axes.pop(0)                                   # (a) context: this method highlighted
        d = main[main.experiment == "mixing"]
        for a in algos(d):
            s_ = d[d.algorithm == a].groupby("mu_target")["nmi"].mean()
            if a in names:
                ax.plot(s_.index, s_.values, label=a, zorder=3, **style(a))
            else:
                ax.plot(s_.index, s_.values, color="#c9c8c0", lw=1.5, zorder=1)
        ax.set(xlabel="Mixing μ", ylabel="NMI", ylim=(-0.05, 1.05), title="vs other methods (grey)")
        ax.legend(fontsize=7, loc="lower left")

        if has_sweep:                                      # (b) parameter sweep, solid = LFR, dashed = email
            ax = axes.pop(0)
            for a in names:
                for g, ls in [("mix_mu0.4_s1", "-"), ("email", "--")]:
                    x = sweep[(sweep.algorithm == a) & (sweep.graph == g)]
                    if x.empty:
                        continue
                    s_ = x.groupby("sweep_value")["nmi"].mean()
                    st = style(a)
                    ax.plot(s_.index, s_.values, ls=ls, color=st["color"], marker=st["marker"],
                            label=f"{a}, {'LFR μ=0.4' if g != 'email' else 'email'}")
            pname = sweep[sweep.algorithm.isin(names)].sweep_param.iloc[0]
            ax.axvline(1.0, color=MUTED, lw=1, ls=":")
            ax.set(xlabel=PARAM_LABEL.get(pname, pname), ylabel="NMI", ylim=(0, 1.05), title="Parameter sweep")
            ax.legend(fontsize=6)

        if has_abl:                                        # (c) ablation, one panel per method
            shades = ["#184f95", "#2a78d6", "#6da7ec", "#b7d3f6"]
            for a in names:
                x = abl[abl.algorithm == a]
                if x.variant.nunique() < 2:
                    continue
                ax = axes.pop(0)
                piv = x.groupby(["graph", "variant"])["nmi"].mean().unstack()
                piv = piv.reindex([g for g in core.ABLATION_GRAPHS if g in piv.index])
                piv.plot.bar(ax=ax, color=shades[: piv.shape[1]], width=0.75, rot=0)
                ax.set_xticklabels([g.replace("mix_", "").replace("_s1", "") for g in piv.index])
                ax.set(title=f"Ablation: {a}", xlabel="", ylabel="NMI", ylim=(0, 1.05))
                ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
        save(fig, f"{fname}.png")


# ---------------------------------------------------------------- tables ---

def tables(main, stab):
    t = main.groupby(["graph", "algorithm"]).agg(
        NMI=("nmi", "mean"), NMI_sd=("nmi", "std"), ARI=("ari", "mean"),
        modularity=("modularity", "mean"), found=("n_found", "mean"), true=("n_true", "first"),
        disconnected=("frac_disconnected", "mean"), runtime_s=("runtime_s", "mean"),
    ).round(3)
    if stab is not None:
        t = t.join(stab.set_index(["graph", "algorithm"])[["between_run_nmi_mean"]].round(3))
    t.to_csv(os.path.join(core.RESULTS, "summary_table.csv"))
    print("  wrote results/summary_table.csv")
    failed = glob.glob(os.path.join(RAW, "main_*.csv"))
    allrows = pd.concat([pd.read_csv(f) for f in failed])
    bad = allrows[allrows.status != "ok"]
    if len(bad):
        bad[["algorithm", "graph", "seed", "status"]].to_csv(os.path.join(core.RESULTS, "failures.csv"), index=False)
        print(f"  wrote results/failures.csv ({len(bad)} failed runs, report these!)")


def main():
    main_df, stab, sweep, abl = read("main"), read("stability"), read("sweep"), read("ablation")
    if main_df is None:
        return print("No results yet. Run: python run.py <algorithm> main")
    fig1_accuracy_vs_mu(main_df)
    fig2_runtime_vs_n(main_df)
    if stab is not None:
        fig3_stability(stab)
    fig4_modularity_gap(main_df)
    if sweep is not None:
        fig5_sweep(sweep)
    fig6_email_bridge(main_df)
    if abl is not None:
        fig7_ablation(abl)
    fig8_disconnected(main_df)
    fig9_recovery_by_size("mix_mu0.4_s1")
    if "email" in set(main_df.graph):
        fig9_recovery_by_size("email")
    fig10_email_confusion()
    per_part_figures(main_df, sweep, abl)
    tables(main_df, stab)


if __name__ == "__main__":
    main()
