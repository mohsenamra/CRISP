"""Figure: positioning map of the related work (how the level is chosen x what is guaranteed)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
os.makedirs("figures", exist_ok=True)
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.6})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d2"
STREAM = {"ML for MILP": "#2a78d6", "OR reduction": "#eb6834", "Unit commitment": "#1baf7a", "Conformal / distribution-free": "#8a5cd1"}
XL = ["Hand-set", "Tuned or\nsearched", "Learned or\nstructural", "Calibrated\n(conformal)"]
YL = ["None", "Indirect or\npartial bound", "Via a parallel\nfull solve", "Finite-sample, on\nanother quantity", "Finite-sample, on\nthe optimality gap"]
W = [  # (label, stream, x, y)
    ("Ding et al. 2020", "ML for MILP", 0, 0), ("Han et al. 2023", "ML for MILP", 0, 0), ("Liu et al. 2025", "ML for MILP", 0, 0),
    ("Sun et al. 2021", "OR reduction", 0, 0), ("Spieckermann et al. 2025", "OR reduction", 0, 0), ("Xu et al. 2025", "OR reduction", 0, 0), ("Yaakoubi & Dimitrakopoulos 2025", "OR reduction", 0, 0),
        ("Xavier et al. 2021", "Unit commitment", 0, 0), ("Gao et al. 2023", "Unit commitment", 0, 0), ("Ramesh & Li 2024a,b", "Unit commitment", 0, 0),
    ("Lu et al. 2025", "Unit commitment", 0, 0), ("He et al. 2026", "Unit commitment", 0, 0), ("Yang et al. 2025", "Unit commitment", 0, 0), ("Li et al. 2026", "ML for MILP", 0, 0),
    ("Yoon 2022", "ML for MILP", 1, 0), ("Larsen et al. 2024", "OR reduction", 1, 0), ("Fritz et al. 2026", "Unit commitment", 1, 0),
    ("Nair et al. 2020", "ML for MILP", 2, 0), ("Sonnerat et al. 2021", "ML for MILP", 2, 0), ("Huang et al. 2023", "ML for MILP", 2, 0),
    ("Qu et al. 2026", "OR reduction", 2, 0), ("Wang et al. 2026", "Unit commitment", 2, 0),
    ("Yoon et al. 2023", "ML for MILP", 1, 1), ("Pourahmadi & Kazempour 2025", "Unit commitment", 0, 1),
    ("Bertsimas & Stellato 2022", "ML for MILP", 2, 1), ("Clarke & Stellato 2025", "Conformal / distribution-free", 2, 1), ("Le et al. 2026", "ML for MILP", 2, 1), ("Chen et al. 2025", "ML for MILP", 1, 1),
    ("Qu & Yang 2024", "Unit commitment", 0, 2), ("Gao et al. 2025", "Unit commitment", 1, 2),
    ("Johnstone & Cox 2021", "Conformal / distribution-free", 3, 3), ("Patel et al. 2024", "Conformal / distribution-free", 3, 3),
    ("Ovalle et al. 2025", "Conformal / distribution-free", 3, 3), ("Yeh et al. 2025", "Conformal / distribution-free", 3, 3), ("Li et al. 2025", "Conformal / distribution-free", 3, 3),
    ("Clarke & Stellato 2026 (early stopping)", "Conformal / distribution-free", 3, 4), ("Anzum 2026 (early stopping)", "Conformal / distribution-free", 3, 4),
]
CWS = [5.4, 2.4, 2.8, 4.4]; CHS = [1.55, 0.85, 0.6, 1.05, 1.15]
XS = [sum(CWS[:i]) for i in range(5)]; YS = [sum(CHS[:j]) for j in range(6)]
fig, ax = plt.subplots(figsize=(10.4, 6.6))
for i in range(4):
    for j in range(5):
        ax.add_patch(Rectangle((XS[i], YS[j]), CWS[i], CHS[j], facecolor="#fbfbf9" if (i + j) % 2 else "#f4f4f1", edgecolor=GRID, lw=0.6))
ax.add_patch(Rectangle((XS[3], YS[4]), CWS[3], CHS[4], facecolor="#fff1d6", edgecolor="#d08a00", lw=1.8, zorder=2))
cells = {}
for lab, st, x, y in W: cells.setdefault((x, y), []).append((lab, st))
LINE = 0.205
for (x, y), items in cells.items():
    n = len(items); two = n > 7
    per_col = 7 if two else n
    for k, (lab, st) in enumerate(items):
        col, row = (k // per_col, k % per_col) if two else (0, k)
        x0 = XS[x] + 0.10 + col * (CWS[x] / 2)
        y0 = YS[y + 1] - 0.16 - row * LINE
        ax.add_patch(Rectangle((x0, y0 - 0.035), 0.065, 0.07, facecolor=STREAM[st], edgecolor="none", zorder=3))
        ax.text(x0 + 0.10, y0, lab, va="center", ha="left", fontsize=8.0, color=INK, zorder=3)
cx, cy = XS[3] + 0.12, YS[4] + 0.33
ax.add_patch(FancyBboxPatch((cx, cy - 0.22), CWS[3] - 0.24, 0.44, boxstyle="round,pad=0.01,rounding_size=0.06",
                            facecolor="#d08a00", edgecolor="none", zorder=4))
ax.text(cx + (CWS[3] - 0.24) / 2, cy, "CRISP (this paper)\ncalibrated level of variable fixing", ha="center", va="center", fontsize=8.6, linespacing=1.3,
        color="white", fontweight="bold", zorder=5)
ax.set_xlim(0, XS[4]); ax.set_ylim(0, YS[5])
ax.set_xticks([XS[i] + CWS[i] / 2 for i in range(4)]); ax.set_xticklabels(XL, fontsize=9.2, color=INK)
ax.set_yticks([YS[j] + CHS[j] / 2 for j in range(5)]); ax.set_yticklabels(YL, fontsize=9.2, color=INK)
ax.tick_params(length=0)
for s_ in ax.spines.values(): s_.set_visible(False)
ax.set_xlabel("How the reduction level (or the calibrated parameter) is chosen", fontsize=9.8, color=INK, labelpad=7)
ax.set_ylabel("Guarantee on the returned solution", fontsize=9.8, color=INK, labelpad=7)
h = [plt.Line2D([], [], marker="s", ls="", color=c, ms=5, label=k) for k, c in STREAM.items()]
fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, fontsize=9.2, bbox_to_anchor=(0.56, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("figures/fig_positioning.pdf"); fig.savefig("figures/fig_positioning.png", dpi=220)
print("ok")
