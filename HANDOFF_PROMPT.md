# LightSem prototype — handoff brief

Paste everything below the line into a fresh Prime Agent chat.

---

You are taking over an active research-engineering project called **LightSem**, on an Arch Linux
box with an RTX 5070, running as root. Home dir of the project is `/root/lightsem`. You are
expected to work **autonomously and end-to-end**: build real artifacts, run them, and report
what real execution returned. Never fabricate output.

## 0. What LightSem is

Goal: **semantic audio → stage lighting**. Move beyond the industry's L2 (FFT threshold triggers
— strobe on bass, as in Resolume / SoundSwitch / MADRIX) to **L3: structure-aware, temporally
extended audio→light mapping**, where the model understands build-up / drop / breakdown /
phrase structure and maps it into a continuous light-scene space.

The intended contribution niche is **EDM / club lighting with anticipatory control** — light
that *leads* a build-up rather than reacting to it. See §4 for why that specific niche.

## 1. Environment — exact, with pitfalls that will bite you

- Arch Linux, kernel 7.1.5-1-cachyos, **root**. GPU: NVIDIA RTX 5070, 12227 MiB, driver 610.43.03.
- Project venv: `/root/lightsem/.venv` (Python 3.12).
- Installed and verified: `librosa 0.11.0`, `numpy 2.4.6`, `scipy`, `scikit-learn 1.9.0`,
  `mediapipe 1.0.0`, `cv2 5.0.0` (opencv-python-headless), `yt-dlp 2026.07.04`, `matplotlib`,
  `pandas`, `pyarrow`, `moderngl 5.12.0`, `glfw`, `pyrr`, `imageio 2.37.4`, `imageio-ffmpeg`,
  `av 18.0.0`, `PIL 12.3.0`. System `ffmpeg n8.1.2`.
- Host python is 3.14.6 with **no pip module**, and `pip` maps to a mismatched 3.11. PEP 668 is
  enforced. **Use the venv or `uv` — never system pip.**

### PITFALL 1 — never invoke the venv binary directly
Calling `/root/lightsem/.venv/bin/python script.py` from the shell tool fails with
`ValueError: embedded null byte` (a lifecycle guard tries to read the binary as a script).

**Always** do it this way — write a `.sh` wrapper and run `bash wrapper.sh`:
```bash
#!/usr/bin/env bash
cd /root/lightsem
export PATH="/root/lightsem/.venv/bin:$PATH"
exec python your_script.py
```
Existing wrappers follow this pattern: `run_probe1.sh` … `run_probe4.sh`, `run_fig.sh`.

### PITFALL 2 — long jobs
Use background execution with completion notification, then poll. A blocking `wait` is clamped
to 300 s. Background bash also emits harmless locale noise
(`не удаётся задать группу процесса терминала`, `tcsetattr: ... ioctl`) — ignore it.
**Verify job completion by `stat`/mtime on output files, not by notification arrival** —
notifications arrive in delayed batches and stale ones have already caused confusion here.

### PITFALL 3 — `web_extract` is blocked for github.com and google.github.io
It returns "Blocked: URL targets a private or internal network address". Use `curl -sL` instead;
that works fine.

### Headless GPU rendering — VERIFIED WORKING
```python
ctx = moderngl.create_context(standalone=True, backend="egl")
# -> NVIDIA GeForce RTX 5070/PCIe/SSE2, GL 3.3.0
```
Confirmed rendering to an offscreen FBO and reading back correct pixels
(`/root/lightsem/check_gl.py`). `/dev/dri/renderD128` is world-accessible. **You do not need an
X/Wayland display to render.**

### Desktop session (only if you want to show something live)
There is a GNOME session belonging to another user: `kirill`, uid 1000, seat0, tty2,
`gnome-shell` pid 917, `Xwayland` pid 1365, socket `/run/user/1000/wayland-0`, plus X sockets
`/tmp/.X11-unix/X0` and `X1`. To put a window on that desktop as root:
```bash
XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 <cmd>   # wayland
DISPLAY=:0 XAUTHORITY=/home/kirill/.Xauthority <cmd>             # X11 fallback
```
This is optional. **Prefer headless render → MP4** as the primary deliverable; it is
reproducible and reviewable. Live desktop display is a nice-to-have demo.

## 2. Data already on disk — do not re-download

**AIST Dance DB / AIST++**, obtained legally under its research licence:

| Path | Contents |
|---|---|
| `/root/lightsem/data/kp3d/keypoints3d/*.pkl` | **2817 files, 884 MB** — 3D keypoints |
| `/root/lightsem/data/wav/wav/*.wav` | **60 tracks, 656 MB** — the music |

Filename schema: `gLH_sBM_cAll_d16_mLH1_ch05.pkl`
= genre / setting / camera / **dancer** / **music id** / **choreography id**.
This structure is what makes inter-dancer and same-choreography controls possible.

Constants used throughout: `KP_FPS = 60.0`, `SR = 22050`, `HOP = 512`.

Working URLs (the `storage.googleapis.com/aist_plusplus_public/...` ones return **403** — use these):
- `https://github.com/google/aistplusplus_dataset/releases/download/v1.0/keypoints3d.zip`
  (also `cameras.zip`, `keypoints2d.zip`, `motions.zip`)
- `https://aistdancedb.ongaaccel.jp/v1.0.0/audio/wav.zip`
- Video is reachable too — verified HTTP 206 on a range request to
  `https://aistdancedb.ongaaccel.jp/v1.0.0/video/10M/gBR_sBM_c01_d04_mBR0_ch01.mp4`
  (the `60M/` path and `refined_10M_all_video_url.csv` both 404 — use `10M/`).

Data quirks already discovered:
- Keypoints contain **NaN frames**. Drop sequences with >10 % bad frames, else linearly
  interpolate per joint per coordinate.
- Filename regex must accept `cAll`, not just `c\d+`:
  `r"(g\w+?)_(s\w+?)_(c[\w\d]+?)_(d\d+)_(m\w+?)_(ch\d+)"`
- **Pose clips are short (median 8.7 s, range 7.1–48.0 s) while audio tracks run 30–51 s.**
  Any audio analysis must be clipped to each sequence's own duration. Ignoring this once
  produced `n = 0` usable pairs after a "≥20 beats" filter — a silent, total data loss.

### On YouTube / Just Dance — READ THIS BEFORE SCRAPING
The original project sketch proposed scraping Just Dance gameplay from YouTube → pose estimation
→ derive DMX from pose. **Do not build the corpus that way.** Two independent reasons:

1. **Legal.** It violates YouTube ToS and sits in unresolved choreography + footage copyright.
2. **It is unnecessary.** AIST gives the same thing — the same dancers under the same music,
   multi-view, with ground-truth 3D keypoints — under a research licence. Already downloaded.

A cookie-based downloader was attempted here for completeness and **is currently blocked**:
Brave profile `/home/kirill/.config/BraveSoftware/Brave-Browser/Default/Cookies` holds 961
cookies (25 for youtube.com, including `SID` / `SAPISID` / `LOGIN_INFO`, so the user is logged
in), but `Local State` has `os_crypt = {"portal"}` with **no `encrypted_key`** — Brave is using
the GNOME keyring via the XDG secrets portal. Result:
`Extracted 0 cookies from brave (961 could not be decrypted)`. Decryption would require an
unlocked keyring in `kirill`'s session bus. Public videos download fine without cookies anyway.
Scripts: `ingest/test_cookies.sh`, `ingest/test_cookies2.sh`. **Treat this as a closed dead end
unless you are explicitly told otherwise.**

## 3. What has ALREADY been empirically ruled out — do not redo this

Four probes were run on the AIST data to test the sketch's central mechanism: *can dancer
kinetic energy serve as free dense supervision for DMX?* Scripts and logs are in
`/root/lightsem` (`probe1_pose_vs_audio.py`, `probe2_artifact_controls.py`,
`probe3b_segment_first.py`, `probe4_significance.py`, plus `*_log.txt`).

`KE(t) = Σ_j ||dP_j/dt||²` on root-relative 3D keypoints.

**Probe 1** (263 sequences) — r(KE, RMS envelope) by smoothing scale:
`0.031 @0.1s · 0.080 @0.5s · 0.261 @2s · 0.653 @8s`.
Ridge from a 28-dim audio stack → KE, **GroupKFold by music**:
R² = `0.002 / 0.005 / −0.025 / −0.069`. Does not generalise to a held-out track at all.
Inter-dancer ceiling r: `0.024 @0.1s → 0.495 @8s`.

**Probe 2** — killed three artifact explanations (fixed sync lag via ±1 s cross-correlation;
same- vs different-choreography; amplitude-vs-timing via jerk-peak event F1).

**Probe 3b** — corrected a real methodological error (probes 1–2 used fixed windows on a
continuous modality; the correct rule is *segment before you align*). Beat-synchronous
inter-dancer KE: SAME choreography `+0.131 ± 0.320` (n=136) vs DIFF `−0.007 ± 0.277`.
Structural-boundary F1 pose-vs-audio `0.462` against a random null of `0.352`.

**Probe 4** — cluster-aware significance. Permutation within track, 10 000 iters:
gap `+0.1377`, null `+0.0001 ± 0.0489`, **p = 0.0041**. Bootstrap over 12 tracks:
95 % CI `[+0.0881, +0.1942]`, Cohen's d `+0.476`.

**Verdict, recorded as a formal rule-out:** choreography-specific information at beat scale
*does* exist and is statistically significant — but its **shared variance is R² ≈ 1.7 %**,
against an RMS baseline of 0.25 % and a held-out-track ridge R² that is *negative*. So:

> Dancer kinetic energy is **not** a viable dense beat-resolution DMX supervision signal.
> The exploitable signal lives at **structural scale**, and at that scale the plain audio
> envelope already carries it — the pose detour is redundant.

**Design consequence you must carry forward:** do not map waveform → DMX. Map
**structure `{build-up, drop, breakdown, phrase}` → scene**. Structure is obtainable directly
from audio: no dancer, no pose estimation, no legal problem.

## 4. Prior art — full review at `/root/lightsem/RESEARCH.md`

A Perplexity deep-research pass (web + scholar) was run, pre-seeded with everything above so it
could not merely restate known facts. **Read `/root/lightsem/RESEARCH.md` first.** Summary of
what it established:

**Exists:**
- **Skip-BART** — arXiv `2506.01482`, **accepted ICLR 2026**. BART audio→light hue+value with a
  skip connection across the music/light frame grid. Human eval: p < 0.001 vs rule-based,
  **p = 0.72 vs human lighting engineers** (indistinguishable). Code
  `github.com/RS2002/Skip-BART`, weights `huggingface.co/RS2002/Skip-BART`.
- **SeqLight** — arXiv `2605.03660`, multi-light HSV via imitation learning.
- **RPMC-L2** — `huggingface.co/datasets/RS2002/RPMC-L2`, CC-BY-4.0, ~699 clips. The *only*
  public audio↔lighting corpus in existence.
- Structure/beat SOTA: All-In-One (Kim & Nam 2023), Beat This! (ISMIR 2024), BeatNet (ISMIR
  2021), zero-latency PLP beat tracker (TISMIR 2024).
- **EDMFormer + EDM-98** (arXiv `2603.08759`) — 98 annotated EDM tracks targeting **drops and
  buildups**; source corpus `disco-eth/edm-cue` (4710 songs).
- **BlenderDMX** (`github.com/open-stage/blender-dmx`) — GPLv3, Python-scriptable,
  Art-Net/sACN, Eevee realtime. The one credible RL/eval environment.
- OLA, OSC2DMX — transport only.

**Does not exist (these are the openings):**
- Any **EDM/club lighting** dataset or model. RPMC-L2 is rock/metal-skewed and its authors say so.
- Any **anticipatory** lighting work. Skip-BART is non-causal only in the offline sense
  (bidirectional BART sees the whole track); its authors name real-time/online control as open.
- Any **real-time ML DMX** system. Not a single published audio→DMX latency figure.
- Any audio↔lighting dataset besides RPMC-L2. Note GDTF/MVR are *fixture and rig-geometry*
  formats, **not** behavioural corpora — a common misreading. No public show-file corpus exists.
- Any **objective metric** for light↔music synchrony. No FID analog, no Beat Alignment Score
  analog. Everything reduces to human preference plus RMSE on HSV.
- Any published use of **dance motion as a lighting control signal** — VJ practice only, no
  papers, no code. (We tested the naive version; see §3. The gap being open is not proof it is
  fertile.)

**Claiming AI without evidence:** SoundSwitch Autoscript (really beatgrid + phrase heuristics
over presets), MaestroDMX, AI LIGHTSHOW, AULIOS, Y-Link.

## 5. YOUR TASK

Build a working prototype, in this order. Each stage must actually run.

### Stage A — data & parsing layer
`lightsem/data/` package:
- Loader for AIST++ keypoints (NaN policy from §2) and paired audio, aligned on the
  music-id/choreography-id schema, clipped to per-sequence duration.
- Cache to parquet. Emit a manifest with per-sequence duration, dancer, music id, choreo id,
  NaN fraction, and usable/rejected status with reason.
- A CLI entry point and a smoke test proving it loads a real sequence end to end.
- Keep the yt-dlp ingest module as a **general** downloader (arbitrary URL + optional cookie
  file), documented as not the corpus path. Do not run a scraping campaign.

### Stage B — audio structure & anticipation
`lightsem/audio/`:
- Beat / downbeat / tempo tracking, onset and RMS envelopes, spectral features.
- **Structural segmentation** — boundaries plus segment labels approximating
  `{intro, build-up, drop, breakdown, outro}`. Start with librosa (novelty + recurrence matrix,
  laplacian segmentation); it is a legitimate baseline. Note All-In-One / EDMFormer as the
  upgrade path.
- **Anticipation signal — this is the novel bit.** Compute a build-up tension curve and a
  *predicted time-to-next-drop*. The lighting cue must be emittable **before** the drop lands.
  Make the lookahead horizon an explicit parameter and report it in the output. Be honest in
  your report about which part is genuinely predictive versus which is offline lookahead over a
  fully-known track — do not blur that line.

### Stage C — DMX mapping layer
`lightsem/dmx/`:
- A fixture model: dimmer, strobe, RGB/HSV colour, pan/tilt, zoom/beam angle, gobo.
- A rig definition — a small club rig, e.g. 4–8 moving heads on a truss + 2–4 wash/blinder +
  strobe + haze.
- Map **structure → scene**, per §3, *not* waveform → DMX. Scenes are named, parameterised
  looks; transitions are driven by segment boundaries and the anticipation curve, with
  beat-synchronous micro-timing on top.
- Keep the **micro/latency layer deterministic** (beatgrid-driven strobe and hit accents). The
  learned layer belongs at macro/meso scale. This split is the one point the original sketch
  got right.
- Output a standard DMX universe frame stream (512 channels @ ≥30 fps) plus a readable JSON/CSV
  cue list. Optionally emit Art-Net so it can drive real hardware or BlenderDMX later.

### Stage D — stage simulator and visualisation
`lightsem/sim/` — a real-time-capable renderer, **moderngl + standalone EGL** (verified working,
§1). Do not depend on Blender for the prototype; note BlenderDMX as the future
photoreal/RL path.

Requirements:
- 3D stage: floor, back wall, truss, fixture positions.
- **Volumetric beams with haze** — this is what makes a lighting sim legible. Raymarch a cone
  per fixture with distance falloff and a haze scattering term. This is the visual core; give
  it real effort.
- Moving heads with pan/tilt actually articulating, plus beam-angle/zoom.
- **A projector fixture** — textured/gobo projection onto stage geometry, as distinct from a
  beam light.
- Consumes the Stage C DMX stream, renders synced to the audio, muxes with ffmpeg to an MP4.
- A live mode is welcome; if you do one, target the `kirill` GNOME session per §1. The MP4 is
  the required deliverable, live mode is optional.

### Stage E — end-to-end
One command: pick an AIST track (or any WAV) → features → structure → DMX → rendered MP4 with
audio. Produce at least one real rendered video and report its actual path, duration, resolution
and file size.

## 6. Definition of done

- Every stage runs; you have exercised it, not merely written it.
- A real MP4 exists on disk with real audio and real light animation, and you state its verified
  properties.
- A short honest report: what works, what is baseline-quality, what is stubbed, and what the
  anticipation layer genuinely does versus merely appears to do.
- Anything non-obvious you discovered gets written into the repo, not just said in chat.

## 7. Working rules

- **Empirical and skeptical.** Verify claims against real data before building on them. Negative
  results are useful outcomes — report them plainly, do not bury them.
- **Never fabricate output.** If something is blocked, say so and try an alternative. A reported
  blocker beats an invented result every time.
- Price any metric you report against a trivial baseline. A bare correlation is not a result.
- Do not silently work around a missing capability — name it.
- The prior rule-out in §3 is binding: **do not rebuild the pose→DMX supervision path.** If you
  believe you have new evidence that overturns it, state the evidence explicitly and say what
  you are retracting.
