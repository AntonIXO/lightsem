#!/usr/bin/env python
"""Figure: the timescale gradient + the decisive same-vs-diff choreography control."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

scales = [0.1, 0.5, 2.0, 8.0]
x = np.arange(len(scales))
labels = ["0.1s\n(strobe)", "0.5s\n(beat)", "2.0s\n(bar)", "8.0s\n(phrase)"]

r_rms   = [0.031, 0.080, 0.261, 0.653]
r2_audio= [0.0019, 0.0050, -0.0254, -0.0694]
r2_ceil = [0.0006, 0.0022, 0.0174, 0.2449]
same    = [0.313, 0.366, 0.466, 0.812]
same_sd = [0.154, 0.222, 0.360, 0.280]
diff    = [0.290, 0.354, 0.464, 0.842]
diff_sd = [0.138, 0.208, 0.362, 0.224]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("white")

ax = axes[0]
ax.plot(x, r_rms, "o-", lw=2.5, ms=9, color="#e63946", label="r(KE, RMS envelope)")
ax.plot(x, np.sqrt(np.clip(r2_ceil, 0, None)), "s--", lw=2.5, ms=9,
        color="#457b9d", label="inter-dancer ceiling r")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_xlabel("smoothing timescale")
ax.set_ylabel("Pearson r"); ax.set_ylim(-0.05, 1.0)
ax.axhspan(-0.05, 0.15, color="grey", alpha=0.15)
ax.text(0.05, 0.02, "noise floor", fontsize=9, color="grey")
ax.set_title("M1/M3: pose only tracks audio\nat PHRASE scale", fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
w = 0.38
ax.bar(x - w/2, r2_audio, w, color="#e63946", label="audio→KE Ridge $R^2$ (held-out track)")
ax.bar(x + w/2, r2_ceil, w, color="#457b9d", label="inter-dancer ceiling $R^2$")
ax.axhline(0, color="k", lw=1)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("$R^2$")
ax.set_title("M2: audio→pose does NOT generalise\nto a new track (negative at 2–8s)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

ax = axes[2]
ax.bar(x - w/2, same, w, yerr=same_sd, capsize=4, color="#2a9d8f",
       label="SAME choreography (n=115)")
ax.bar(x + w/2, diff, w, yerr=diff_sd, capsize=4, color="#e9c46a",
       label="DIFFERENT choreography (n=250)")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("max cross-corr r (±1s lag)")
ax.set_title("A2 DECISIVE: same choreo agrees no better\nthan different choreo → no signal",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y"); ax.set_ylim(0, 1.15)

fig.suptitle("Pose as intermediate representation for audio→light: ruled out  "
             "(AIST++, 1408 seqs / 60 tracks, CPU probes)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("/root/lightsem/verdict.png", dpi=140, bbox_inches="tight")
print("saved /root/lightsem/verdict.png")
