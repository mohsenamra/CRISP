import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
os.makedirs("figures", exist_ok=True)
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8986", "axes.linewidth": 0.8, "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#52514e"
B = json.load(open("results/summary_bench_fastest.json")); S = json.load(open("results/summary_ssoa_v2.json"))
probs = [("setcover", "Set covering", B["setcover"]), ("cauction", "Comb. auction", B["cauction"]),
         ("facility", "Facility location", B["facility"]), ("indset", "Independent set", B["indset"]), ("ssoa", "Supplier selection", S)]
# ---------------- Paper Figure 3: risk-speed trade-off
fig, axs = plt.subplots(1, 5, figsize=(7.4, 2.5))
for ax, (key, title, d) in zip(axs, probs):
    cur = [c for c in d["curve"] if c["frac_fixed"] > 0 and c["fallback"] <= 0.2]
    x = np.array([100 * c["mean_gap"] for c in cur]); y = np.array([c["speedup_median"] for c in cur])
    ax.plot(x, y, color=GRAY, lw=1.6, zorder=1); ax.scatter(x, y, s=10, color=GRAY, zorder=2, linewidths=0)
    for lam in (0.9, 0.95, 0.99):
        c = next(c for c in d["curve"] if abs(c["lam"] - lam) < 1e-9)
        ax.scatter(100 * c["mean_gap"], c["speedup_median"], s=34, marker="s", color=ORANGE, edgecolors="white", linewidths=1.2, zorder=4)
    for a in (0.005, 0.01, 0.02):
        r = d["results"][f"CRC_meangap|n100|a{a}"]
        ax.scatter(100 * r["realized_risk"], r["speedup_median"], s=40, color=BLUE, edgecolors="white", linewidths=1.2, zorder=5)
    ax.axvline(1.0, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=0)
    ax.set_yscale("log"); ax.set_title(title, fontsize=8.5, color=INK, pad=4)
    from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}")); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Mean gap (%)"); ax.set_xlim(left=-0.1)
axs[0].set_ylabel("Median speed-up (×, log)")
h = [plt.Line2D([], [], color=GRAY, marker="o", ms=3.5, lw=1.6, label="All thresholds λ"),
     plt.Line2D([], [], color=ORANGE, marker="s", ls="", ms=6, label="Hand-set λ = 0.90 / 0.95 / 0.99"),
     plt.Line2D([], [], color=BLUE, marker="o", ls="", ms=6, label="CRISP-CRC, target 0.5 / 1 / 2%")]
fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig("figures/fig2_tradeoff.pdf"); fig.savefig("figures/fig2_tradeoff.png", dpi=200)
# ---------------- Paper Figure 4: share of calibration sets above target, tuned vs CRISP-LTT
PN = json.load(open("results/paper_numbers.json"))
labels = ["Set\ncovering", "Comb.\nauction", "Facility\nlocation", "Indep.\nset", "Supplier\nselection"]
tun = [100 * PN[k]["tuned_vs_crisp|LTT_P(gap>1%)|a0.1"]["tuned"]["above"] for k, *_ in probs]
cri = [100 * PN[k]["tuned_vs_crisp|LTT_P(gap>1%)|a0.1"]["crisp"]["above"] for k, *_ in probs]
fig, ax = plt.subplots(figsize=(4.6, 2.4)); xs = np.arange(5); w = 0.36
b1 = ax.bar(xs - w / 2 - 0.01, tun, w, color=ORANGE, label="Tuned on calibration set (no correction)")
b2 = ax.bar(xs + w / 2 + 0.01, cri, w, color=BLUE, label="CRISP-LTT")
for bars in (b1, b2):
    for b in bars: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.axhline(10, color=INK2, lw=0.8, ls=(0, (3, 3))); ax.text(4.75, 10.8, "δ = 10%", ha="right", fontsize=7.5, color=INK2)
ax.set_xticks(xs); ax.set_xticklabels(labels); ax.set_ylabel("Calibration sets above target (%)"); ax.grid(axis="x", visible=False)
ax.set_ylim(0, 45); ax.legend(frameon=False, fontsize=7.5, loc="upper right")
fig.tight_layout(); fig.savefig("figures/fig3_tuned.pdf"); fig.savefig("figures/fig3_tuned.png", dpi=200)
