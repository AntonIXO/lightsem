#!/usr/bin/env python
"""H-H, step 1: Beat Saber lightshow maps -> the Stage 9 target format (PLAN.md 9.1 / 9.2).

Per usable map (manifest `usable`, the difficulty with the most lighting events):

  cache/beatsaver_light/<song8>=<id>.npz   group-level light at 40 fps on the map's exact BPM grid:
        inten (G,T) mean intensity over the group's (virtual) lights, rgb (G,T,3), on (G,T),
        n_active (T), events: structured (t_s, group, kind, color) for every authored cue,
        fx: ring spin / ring zoom / laser-speed event trains, groups: names, meta json
  cache/beatsaver_visual/<song8>=<id>.npz  the operator-head format (exp-016 recipe, unchanged
        scripts): x = causal 10 Hz audio features, y = 6-d scene vector rendered from the group
        state (hue cos/sin, sat, value, value p90, hue concentration), plus ev_any (10 Hz, any
        authored cue in the bin) and ev_group (G, T10)
  cache/beatsaver_audio{,_offline}/<song8>=<id>.npz   the 92-channel bus at 40 fps (extract_bus)
  beatsaver/wav/<id>.wav                   44.1 kHz mono decode of the mapper's .egg

The v2 lanes (0 back lasers, 1 ring lights, 2/3 left/right lasers, 4 centre, 6/7/10/11 extras)
and the v3 event-box groups are both rendered through one small virtual-light engine: every
cue addresses a subset of a group's lights (Chroma _lightID, v3 box filters), with the game's
on / flash / fade / off / transition semantics, and the group envelope is the mean over lights.
Colours come from the difficulty's env colour overrides or the defaults (left red, right blue),
Chroma _color when present.  The song key groups different maps of the same song for splits.

  bash run.sh scripts/data/build_beatsaver.py [--workers 10] [--limit N]
"""
from __future__ import annotations

import argparse, colorsys, hashlib, io, json, re, subprocess, sys, time, zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/lightsem")
_argv, sys.argv = sys.argv, sys.argv[:1]
from scripts.data.extract_rpmc_causal import features_from_mel, SR, HOP, FPS_OUT  # noqa: E402
sys.argv = _argv
from lightsem.audio.online import causal_resample, ema, extract_bus  # noqa: E402
from lightsem.motion.io import save_bus  # noqa: E402
from scripts.diagnostics.probe_event_target import event_train  # noqa: E402

ROOT = Path("/mnt/lightsem-data/beatsaver")
CACHE = Path("/mnt/lightsem-data/cache")
FPS = 40.0
N_VIRT = 16                     # virtual lights per v3 group / per v2 lane without lightIDs
LIGHT_LANES = {0: "back_lasers", 1: "ring_lights", 2: "left_lasers", 3: "right_lasers", 4: "center_lights",
               6: "extra_left", 7: "extra_right", 10: "extra_3", 11: "extra_4"}
DEFAULT_LEFT, DEFAULT_RIGHT, WHITE = (1.0, 0.0, 0.0), (0.0, 0.45, 1.0), (1.0, 1.0, 1.0)
FLASH_TAU, FADE_TAU, FLASH_PEAK = 0.25, 0.55, 1.3


def song_key(v: dict) -> str:
    s = re.sub(r"[^a-z0-9]", "", f"{v.get('song') or ''}{v.get('artist') or ''}".lower())
    return hashlib.sha1(s.encode()).hexdigest()[:8]


def rgb_of(c, default):
    if isinstance(c, dict):
        return (float(c.get("r", 0)), float(c.get("g", 0)), float(c.get("b", 0)))
    if isinstance(c, (list, tuple)) and len(c) >= 3:
        return tuple(float(x) for x in c[:3])
    return default


# ------------------------------------------------------------------ cue extraction
class Cues:
    """cues[group] = list of (t_s, light_mask(np.bool_ N), mode, brightness, rgb)"""
    def __init__(self):
        self.groups: dict[str, list] = {}
        self.nlights: dict[str, int] = {}
        self.events: list[tuple] = []          # (t_s, group, kind, color_idx)
        self.fx: dict[str, list[float]] = {"ring_spin": [], "ring_zoom": [], "laser_speed": [], "boost": []}

    def add(self, g, t, mask, mode, b, rgb, cidx):
        self.groups.setdefault(g, []).append((t, mask, mode, b, rgb))
        if mode != "extend":
            self.events.append((t, g, mode, cidx))


def parse_v2(events, beat2t, colors, cues: Cues):
    # first pass: lightIDs per lane
    ids: dict[int, list] = {}
    for e in events:
        ty = e.get("_type")
        if ty in LIGHT_LANES:
            cd = e.get("_customData") or {}
            lid = cd.get("_lightID")
            if lid is not None:
                for x in (lid if isinstance(lid, list) else [lid]):
                    if isinstance(x, (int, float)):
                        ids.setdefault(ty, [])
                        if x not in ids[ty] and len(ids[ty]) < 64:
                            ids[ty].append(x)
    for ty, name in LIGHT_LANES.items():
        n = max(len(ids.get(ty, [])), 1)
        cues.nlights[name] = n if ids.get(ty) else N_VIRT
    boost = False
    for e in sorted(events, key=lambda e: e.get("_time", 0)):
        ty, val = e.get("_type"), e.get("_value", 0)
        if not isinstance(ty, (int, float)) or not isinstance(val, (int, float)):
            continue
        t = beat2t(float(e.get("_time", 0)))
        if ty == 5:
            boost = bool(val); cues.fx["boost"].append(t); continue
        if ty == 8:
            cues.fx["ring_spin"].append(t); continue
        if ty == 9:
            cues.fx["ring_zoom"].append(t); continue
        if ty in (12, 13):
            cues.fx["laser_speed"].append(t); continue
        if ty not in LIGHT_LANES:
            continue
        name = LIGHT_LANES[ty]
        cd = e.get("_customData") or {}
        n = cues.nlights[name]
        lid = cd.get("_lightID")
        if lid is not None and ids.get(ty):
            mask = np.zeros(n, bool)
            for x in (lid if isinstance(lid, list) else [lid]):
                if x in ids[ty]:
                    mask[ids[ty].index(x)] = True
        else:
            mask = np.ones(n, bool)
        val = int(val)
        if val == 0:
            mode, cidx = "off", -1
        else:
            cidx = 1 if val <= 4 else 0 if val <= 8 else 2
            k = (val - 1) % 4
            mode = ("on", "flash", "fade", "transition")[k]
        rgb = rgb_of(cd.get("_color"), None) if cd.get("_color") is not None else None
        if rgb is None:
            rgb = WHITE if cidx == 2 else (colors["boost"][cidx] if boost else colors["main"][cidx]) if cidx >= 0 else (0, 0, 0)
        b = float(e.get("_floatValue", 1.0)) if isinstance(e.get("_floatValue", 1.0), (int, float)) else 1.0
        cues.add(name, t, mask, mode, b if mode != "off" else 0.0, rgb, cidx)


def parse_v3(d, beat2t, colors, cues: Cues):
    # v3 basic events keep the v2 lane semantics (with float brightness)
    ev2 = [{"_time": e.get("b", 0), "_type": e.get("et"), "_value": e.get("i", 0),
            "_floatValue": e.get("f", 1.0), "_customData": e.get("customData")} for e in d.get("basicBeatmapEvents") or []]
    parse_v2(ev2, beat2t, colors, cues)
    for grp in d.get("lightColorEventBoxGroups") or []:
        g = f"g{int(grp.get('g', 0))}"
        cues.nlights[g] = N_VIRT
        t0 = float(grp.get("b", 0.0))
        for box in grp.get("e") or []:
            f = box.get("f") or {}
            idx = np.arange(N_VIRT)
            ft, p, tt = int(f.get("f", 1)), int(f.get("p", 1) or 1), int(f.get("t", 0))
            if ft == 1:                                    # division: section tt of p
                p = max(p, 1); mask = (idx * p // N_VIRT) == min(tt, p - 1)
            else:                                          # step and offset: p offset, tt step
                mask = (idx >= p) & (((idx - p) % max(tt, 1)) == 0)
            if f.get("r"):
                mask = mask[::-1].copy()
            sel = np.flatnonzero(mask); nsel = max(len(sel), 1)
            w, dtype = float(box.get("w", 0.0) or 0.0), int(box.get("d", 1) or 1)
            offs = np.zeros(N_VIRT)
            if w > 0:
                k = np.zeros(N_VIRT); k[sel] = np.arange(len(sel))
                offs = (w * k / max(nsel - 1, 1)) if dtype == 1 else np.minimum(w * k, 4.0)
            r, rtype = float(box.get("r", 0.0) or 0.0), int(box.get("t", 1) or 1)
            boffs = np.zeros(N_VIRT)
            if r != 0:
                k = np.zeros(N_VIRT); k[sel] = np.arange(len(sel))
                boffs = (r * k / max(nsel - 1, 1)) if rtype == 1 else r * k
            for j, ev in enumerate(box.get("e") or []):
                tb = t0 + float(ev.get("b", 0.0))
                trans = int(ev.get("i", 0)); cidx = int(ev.get("c", 0)); s = float(ev.get("s", 1.0))
                mode = ("on", "transition", "extend")[min(trans, 2)]
                rgb = WHITE if cidx == 2 else colors["main"][min(cidx, 1)]
                strobe = float(ev.get("f", 0) or 0)
                # distribution: one cue per distinct offset (grouped) keeps the cue count sane
                if w > 0 and j == 0 or (w > 0 and box.get("b")):
                    for o in np.unique(offs[sel]):
                        m = np.zeros(N_VIRT, bool); m[sel[offs[sel] == o]] = True
                        cues.add(g, beat2t(tb + o), m, mode, s, rgb, cidx)
                else:
                    m = np.zeros(N_VIRT, bool); m[sel] = True
                    cues.add(g, beat2t(tb), m, mode, s, rgb, cidx)
                if strobe > 0:
                    cues.fx.setdefault("strobe", []).append(beat2t(tb))
    for grp in d.get("lightRotationEventBoxGroups") or []:
        t0 = float(grp.get("b", 0.0))
        for box in grp.get("e") or []:
            for ev in box.get("l") or []:
                cues.fx.setdefault("rotation", []).append(beat2t(t0 + float(ev.get("b", 0.0))))


# ------------------------------------------------------------------ render
def render_group(cl: list, n: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    """(T,) mean intensity and (T,3) intensity-weighted rgb of n virtual lights."""
    inten = np.zeros((n, T)); rgb = np.zeros((n, T, 3))
    cur_b = np.zeros(n); cur_rgb = np.zeros((n, 3)); cur_t = np.zeros(n)   # per light state at last cue
    cur_mode = np.array(["off"] * n, dtype=object)
    tgrid = np.arange(T) / FPS
    cl = sorted(cl, key=lambda c: c[0])
    # per light: list of segment boundaries -> render with a sweep per cue (cues are sparse per light)
    last_idx = np.zeros(n, int)
    def seg(i, a, b, mode, bri, col, t_cue, prev_val):
        if b <= a:
            return
        tt = tgrid[a:b] - t_cue
        if mode == "off":
            v = np.zeros(b - a)
        elif mode == "on" or mode == "extend":
            v = np.full(b - a, bri)
        elif mode == "flash":
            v = bri * (1.0 + (FLASH_PEAK - 1.0) * np.exp(-tt / FLASH_TAU))
        elif mode == "fade":
            v = bri * FLASH_PEAK * np.exp(-tt / FADE_TAU)
        else:  # transition: reached bri at t_cue; the ramp was written on the previous segment
            v = np.full(b - a, bri)
        inten[i, a:b] = v; rgb[i, a:b] = col
    for (t, mask, mode, bri, col) in cl:
        k = min(max(int(round(t * FPS)), 0), T)
        for i in np.flatnonzero(mask):
            a = last_idx[i]
            if mode == "transition" and k > a:
                # linear ramp from the light's value at a to bri at k (game 'transition' semantics)
                v0 = inten[i, a - 1] if a > 0 else 0.0
                inten[i, a:k] = np.linspace(v0, bri, k - a, endpoint=False)
                rgb[i, a:k] = 0.5 * (cur_rgb[i] + np.asarray(col))
            else:
                seg(i, a, k, cur_mode[i], cur_b[i], cur_rgb[i], cur_t[i], None)
            last_idx[i] = k
            cur_mode[i] = "on" if mode == "transition" else ("on" if mode == "extend" else mode)
            cur_b[i] = bri if mode != "extend" else cur_b[i]
            cur_rgb[i] = col if mode != "extend" else cur_rgb[i]
            cur_t[i] = t
    for i in range(n):
        seg(i, last_idx[i], T, cur_mode[i], cur_b[i], cur_rgb[i], cur_t[i], None)
    m = inten.mean(0)
    w = inten / np.maximum(inten.sum(0, keepdims=True), 1e-9)
    c = (w[:, :, None] * rgb).sum(0)
    return m, c


def scene_vector(inten: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """(G,T),(G,T,3) -> (T,6) in RPMC-L2's reduced format (hue cos/sin, sat, value, value p90, hue conc)."""
    G, T = inten.shape
    val = inten.mean(0)
    p90 = np.percentile(inten, 90, axis=0)
    hue = np.zeros((G, T)); sat = np.zeros((G, T))
    for g in range(G):
        for t in range(T):
            r, gg, b = np.clip(rgb[g, t] / max(rgb[g, t].max(), 1e-9), 0, 1) if rgb[g, t].max() > 0 else (0, 0, 0)
            h, s, _ = colorsys.rgb_to_hsv(r, gg, b)
            hue[g, t], sat[g, t] = h, s
    w = inten / np.maximum(inten.sum(0, keepdims=True), 1e-9)
    hc = (w * np.cos(2 * np.pi * hue)).sum(0); hs = (w * np.sin(2 * np.pi * hue)).sum(0)
    conc = np.sqrt(hc ** 2 + hs ** 2)
    s = (w * sat).sum(0)
    return np.stack([hc, hs, s, val, p90, conc], 1).astype(np.float32)


def scene_vector_fast(inten, rgb):
    """vectorised hue/sat via numpy (same maths as colorsys)."""
    G, T = inten.shape
    mx = rgb.max(-1); mn = rgb.min(-1)
    d = mx - mn
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = np.zeros((G, T))
    with np.errstate(divide="ignore", invalid="ignore"):
        hr = ((g - b) / d) % 6; hg = (b - r) / d + 2; hb = (r - g) / d + 4
        hue = np.where(mx == r, hr, np.where(mx == g, hg, hb)) / 6.0
        hue = np.where(d > 1e-9, hue, 0.0)
        sat = np.where(mx > 1e-9, d / np.maximum(mx, 1e-9), 0.0)
    val = inten.mean(0); p90 = np.percentile(inten, 90, axis=0)
    w = inten / np.maximum(inten.sum(0, keepdims=True), 1e-9)
    hc = (w * np.cos(2 * np.pi * hue)).sum(0); hs = (w * np.sin(2 * np.pi * hue)).sum(0)
    return np.stack([hc, hs, (w * sat).sum(0), val, p90, np.sqrt(hc ** 2 + hs ** 2)], 1).astype(np.float32)


# ------------------------------------------------------------------ per map
def one(job):
    mid, v = job
    key = f"{song_key(v)}={mid}"
    # Workers are independent processes; create every output directory here
    # rather than relying on a parent-side mkdir race.
    for sub in ("beatsaver_light", "beatsaver_audio", "beatsaver_audio_offline", "beatsaver_visual"):
        (CACHE / sub).mkdir(parents=True, exist_ok=True)
    out_l = CACHE / "beatsaver_light" / f"{key}.npz"
    if out_l.exists() and (CACHE / "beatsaver_visual" / f"{key}.npz").exists() and \
            (CACHE / "beatsaver_audio_offline" / f"{key}.npz").exists():
        return mid, "exists", 0.0
    t0 = time.time()
    try:
        z = zipfile.ZipFile(ROOT / "maps" / f"{mid}.zip")
        names = z.namelist()
        info = json.loads(z.read(next(n for n in names if n.lower() == "info.dat")))
        bpm = float(info.get("_beatsPerMinute") or (info.get("audio") or {}).get("bpm"))
        offset = float(info.get("_songTimeOffset") or 0.0)
        dn = max(v["diffs"], key=lambda n: v["diffs"][n]["basic_events"] + v["diffs"][n]["color_events"])
        d = json.loads(z.read(dn))
        # env colours for this difficulty
        colors = {"main": [DEFAULT_LEFT, DEFAULT_RIGHT], "boost": [DEFAULT_LEFT, DEFAULT_RIGHT]}
        for st in info.get("_difficultyBeatmapSets") or []:
            for dm in st.get("_difficultyBeatmaps") or []:
                if dm.get("_beatmapFilename") == dn:
                    cd = dm.get("_customData") or {}
                    colors["main"] = [rgb_of(cd.get("_envColorLeft"), DEFAULT_LEFT), rgb_of(cd.get("_envColorRight"), DEFAULT_RIGHT)]
                    colors["boost"] = [rgb_of(cd.get("_envColorLeftBoost"), colors["main"][0]), rgb_of(cd.get("_envColorRightBoost"), colors["main"][1])]
        # beat -> seconds (v3 bpmEvents honoured by the game; v2 custom BPM changes are editor-only)
        bpm_ev = sorted([(float(e["b"]), float(e["m"])) for e in (d.get("bpmEvents") or []) if e.get("m")], key=lambda x: x[0])
        if bpm_ev and bpm_ev[0][0] > 0:
            bpm_ev = [(0.0, bpm)] + bpm_ev
        if not bpm_ev:
            bpm_ev = [(0.0, bpm)]
        seg_b = np.array([b for b, _ in bpm_ev]); seg_m = np.array([m for _, m in bpm_ev])
        seg_t = np.concatenate([[0.0], np.cumsum(np.diff(seg_b) * 60.0 / seg_m[:-1])])
        def beat2t(b):
            i = max(int(np.searchsorted(seg_b, b, side="right") - 1), 0)
            return offset + seg_t[i] + (b - seg_b[i]) * 60.0 / seg_m[i]
        cues = Cues()
        if "_events" in d:
            parse_v2(d["_events"], beat2t, colors, cues)
            schema = "v2"
        else:
            parse_v3(d, beat2t, colors, cues)
            schema = "v3"
        # audio
        wav = ROOT / "wav" / f"{mid}.wav"
        if not wav.exists():
            aud = next(n for n in names if n.lower().endswith((".egg", ".ogg", ".wav", ".mp3")))
            raw = z.read(aud)
            r = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-vn", "-ac", "1", "-ar", "44100", "-y", str(wav)],
                               input=raw, capture_output=True)
            if r.returncode != 0:
                return mid, f"ffmpeg: {r.stderr[:100]!r}", time.time() - t0
        import soundfile as sf
        dur = float(sf.info(str(wav)).duration)
        T = int(round(dur * FPS))
        groups = [g for g in cues.groups if len(cues.groups[g]) > 0]
        if not groups:
            return mid, "no light groups", time.time() - t0
        inten = np.zeros((len(groups), T), np.float32); rgb = np.zeros((len(groups), T, 3), np.float32)
        for gi, g in enumerate(groups):
            m, c = render_group(cues.groups[g], cues.nlights[g], T)
            inten[gi], rgb[gi] = m, c
        on = inten > 0.05
        n_active = on.sum(0).astype(np.int16)
        ev = np.array([(t, groups.index(g), k, c) for (t, g, k, c) in cues.events if g in groups],
                      dtype=[("t_s", "f4"), ("group", "i2"), ("kind", "U10"), ("color", "i1")])
        fx = {k: np.array(sorted(vv), np.float32) for k, vv in cues.fx.items()}
        meta = {"id": mid, "song_key": song_key(v), "schema": schema, "bpm": bpm, "bpm_events": len(bpm_ev),
                "offset_s": offset, "environment": v.get("environment"), "diff": dn, "n_cues": int(len(ev)),
                "duration_s": dur, "name": v.get("name"), "uploader": v.get("uploader"),
                "n_lights": {g: int(cues.nlights[g]) for g in groups}, "chroma": int(max(x["chroma_events"] for x in v["diffs"].values()) > 0)}
        np.savez_compressed(out_l, t=np.arange(T, dtype=np.float32) / FPS, groups=np.array(groups), inten=inten, rgb=rgb,
                            on=on, n_active=n_active, events=ev, meta=json.dumps(meta),
                            **{f"fx_{k}": vv for k, vv in fx.items()})
        # the 92-ch buses at 40 fps (causal and offline)
        for causal, sub in ((True, "beatsaver_audio"), (False, "beatsaver_audio_offline")):
            p = CACHE / sub / f"{key}.npz"
            if not p.exists():
                save_bus(extract_bus(str(wav), causal=causal), p)
        # the operator-head format at 10 Hz (exp-016 recipe: same features as build_justdance_visual)
        p = CACHE / "beatsaver_visual" / f"{key}.npz"
        if not p.exists():
            import librosa
            au, _ = librosa.load(str(wav), sr=SR)
            mel = librosa.feature.melspectrogram(y=au, sr=SR, n_fft=2048, hop_length=HOP, n_mels=128)
            mel_db = librosa.power_to_db(mel).astype(np.float32)
            ch, grid = features_from_mel(mel_db)
            fps_in = SR / HOP
            dur10 = min(dur, mel_db.shape[1] / fps_in)
            n_out = int(round(dur10 * FPS_OUT)); t_out = np.arange(n_out) / FPS_OUT
            t_in = np.arange(mel_db.shape[1]) / fps_in
            cols, fnames = [], []
            for name, vv in ch.items():
                vv = np.nan_to_num(vv, nan=0.0, posinf=0.0, neginf=0.0)
                for tau in (0.2, 1.0, 4.0):
                    cols.append(causal_resample(ema(vv, fps_in, tau), t_in, t_out)); fnames.append(f"{name}_ema{tau}")
            per = grid["period_s"]; ph = np.mod((t_out - grid["phase_s"]) / per, 1.0)
            cols += [np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph), (1 - ph) ** 2.5, np.full(n_out, per)]
            fnames += ["beat_sin", "beat_cos", "beat_accent", "beat_period_s"]
            X = np.stack(cols, 1).astype(np.float32)
            y6 = scene_vector_fast(inten, rgb)
            t40 = np.arange(T) / FPS
            Y = np.stack([np.interp(t_out, t40, y6[:, k]) for k in range(6)], 1).astype(np.float32)
            # authored-cue trains at 10 Hz: any cue in the bin, per group
            ev_group = np.zeros((len(groups), n_out), np.float32)
            for e in ev:
                k = int(e["t_s"] * FPS_OUT)
                if 0 <= k < n_out:
                    ev_group[e["group"], k] = 1.0
            ev_any = (ev_group.sum(0) > 0).astype(np.float32)
            ev_scene, _ = event_train(Y)
            np.savez_compressed(p, x=X, y=Y, names=np.array(fnames), bpm=np.float32(grid["bpm"]), map_bpm=np.float32(bpm),
                                ev_any=ev_any, ev_group=ev_group, ev_scene=ev_scene, n_active=np.interp(t_out, t40, n_active).astype(np.float32),
                                meta=json.dumps({"duration_s": dur10, "n_groups": len(groups), "grid_bpm": float(grid["bpm"]), "map_bpm": bpm}))
        return mid, f"ok {schema} {len(groups)}g {len(ev)}cues {dur:.0f}s", time.time() - t0
    except Exception as e:  # noqa: BLE001
        import traceback
        return mid, f"{type(e).__name__}: {e} | {traceback.format_exc().splitlines()[-3]}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="comma-separated map ids (debug)")
    a = ap.parse_args()
    man = json.loads((ROOT / "manifest.json").read_text())
    jobs = [(k, v) for k, v in sorted(man.items()) if v.get("usable")]
    if a.ids:
        jobs = [j for j in jobs if j[0] in a.ids.split(",")]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} maps, {a.workers} workers", flush=True)
    t0 = time.time(); n_ok = 0
    with Pool(a.workers) as pool:
        for i, (mid, msg, dt) in enumerate(pool.imap_unordered(one, jobs)):
            n_ok += msg.startswith("ok") or msg == "exists"
            print(f"[{i+1}/{len(jobs)}] {mid} {msg} {dt:.0f}s", flush=True)
    print(f"done {n_ok}/{len(jobs)} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
