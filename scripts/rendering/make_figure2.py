#!/usr/bin/env python
"""Figure 2: what segmentation-first changed, and what it did not."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))
fig.patch.set_facecolor("white")

# --- panel 1: fixed window vs beat-sync, same vs diff ---------------------
ax = axes[0]
groups = ["FIXED window\n(probe 2)", "BEAT-SYNC\n(probe 3b)"]
same = [0.366, 0.131]
diff = [0.354, -0.007]
x = np.arange(2); w = 0.35
ax.bar(x - w/2, same, w, color="#2a9d8f", label="SAME choreography")
ax.bar(x + w/2, diff, w, color="#e9c46a", label="DIFF choreography")
ax.axhline(0, color="k", lw=1)
for i, (s, d) in enumerate(zip(same, diff)):
    ax.annotate("", xy=(i - w/2, s), xytext=(i + w/2, d),
                arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.6))
    ax.text(i, max(s, d) + 0.04, f"gap {s-d:+.3f}", ha="center",
            fontsize=10, color="crimson", fontweight="bold")
ax.text(1, -0.14, "p=0.004\nCI [+0.088,+0.194]", ha="center", fontsize=9,
        color="crimson")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("inter-dancer r"); ax.set_ylim(-0.22, 0.52)
ax.set_title("Segmentation UNCOVERED real signal\nthat fixed windows hid",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

# --- panel 2: structural boundaries -- the one real win -------------------
ax = axes[1]
vals = [0.331, 0.462, 0.500]
cols = ["#adb5bd", "#2a9d8f", "#1d3557"]
names = ["random\nnull", "pose→audio\nboundaries", "inter-dancer\nceiling"]
b = ax.bar(range(3), vals, color=cols, width=0.6)
for r, v in zip(b, vals):
    ax.text(r.get_x() + r.get_width()/2, v + 0.012, f"{v:.3f}",
            ha="center", fontweight="bold")
ax.annotate("77% of ceiling", xy=(1, 0.462), xytext=(1.55, 0.30),
            arrowprops=dict(arrowstyle="->", lw=1.6, color="crimson"),
            fontsize=11, color="crimson", fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(names)
ax.set_ylabel("boundary F1 (±0.5s)"); ax.set_ylim(0, 0.60)
ax.set_title("Pose DOES carry musical structure\n(the only genuine win)",
             fontweight="bold")
ax.grid(alpha=0.3, axis="y")

# --- panel 3: but the magnitude kills the application ---------------------
ax = axes[2]
names = ["ceiling $R^2$\n(perfect model)", "audio envelope\nbaseline $R^2$",
         "audio→KE ridge\nheld-out track"]
vals = [0.0171, 0.0025, -0.0089]
cols = ["#1d3557", "#adb5bd", "#e63946"]
b = ax.barh(range(3), vals, color=cols, height=0.55)
for i, v in enumerate(vals):
    ax.text(v + (0.0008 if v >= 0 else -0.0008), i, f"{v:+.4f}",
            va="center", ha="left" if v >= 0 else "right", fontweight="bold")
ax.axvline(0, color="k", lw=1)
ax.set_yticks(range(3)); ax.set_yticklabels(names, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("$R^2$"); ax.set_xlim(-0.016, 0.026)
ax.set_title("…but a PERFECT model explains only 1.7%\n→ cannot drive a light cue",
             fontweight="bold")
ax.grid(alpha=0.3, axis="x")

fig.suptitle("Probe 3b/4 — segmentation-first (per multimodal-ai lec.06) : "
             "signal is REAL and significant, but ~50× too weak to be useful",
             fontsize=13, fontweight="bold")
fig.tight_layout()
Path("/root/lightsem/runs/figures").mkdir(parents=True, exist_ok=True)
fig.savefig("/root/lightsem/runs/figures/verdict2.png", dpi=140, bbox_inches="tight")
print("saved /root/lightsem/runs/figures/verdict2.png")
