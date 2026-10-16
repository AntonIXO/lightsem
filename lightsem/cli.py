"""LightSem CLI — one command from audio to a rendered, audio-synced MP4.

    bash run_torch.sh -m lightsem.cli render --wav path/to.wav --out show.mp4
    bash run_torch.sh -m lightsem.cli render --wav path/to.wav --out show.mp4 --d4max --d6
    bash run_torch.sh -m lightsem.cli dance --wav path/to.wav --out runs/body
    bash run.sh -m lightsem.cli analyse --wav path/to.wav
    bash run.sh -m lightsem.cli manifest

The historical rule-driven renderer is explicitly named legacy-render.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .audio import CAUSAL, ORACLE, compute, evaluate_anticipation, extract, segment
from .data import AistDataset, write_manifest
from .dmx import club_rig, render_stream
from .sim import RenderConfig, StageRenderer

WAV_DIR = Path("/root/lightsem/data/wav/wav")
OUT_DIR = Path("/root/lightsem/runs/cli")


def resolve_wav(args) -> Path:
    if args.wav:
        p = Path(args.wav)
        if not p.exists():
            raise SystemExit(f"no such wav: {p}")
        return p
    if args.aist:
        p = WAV_DIR / f"{args.aist}.wav"
        if not p.exists():
            raise SystemExit(f"no AIST track {args.aist} at {p}")
        return p
    raise SystemExit("provide --wav or --aist")


def do_analyse(args) -> int:
    wav = resolve_wav(args)
    feats = extract(path=str(wav), duration_s=args.duration)
    segs = segment(feats)
    ev = evaluate_anticipation(feats, segs, lookahead_s=args.lookahead)

    print(f"track     : {wav.name}")
    print(f"duration  : {feats.duration_s:.2f}s   tempo {feats.tempo:.1f} BPM")
    print(f"beats     : {feats.beats.size}   downbeats {feats.downbeats.size}")
    print(f"\nsegments ({len(segs)}):")
    for s in segs:
        print(f"  {s.start_s:7.2f}-{s.end_s:7.2f}s  {s.label:10s} "
              f"energy={s.energy:.3f} slope={s.energy_slope:+.3f}")
    print(f"\nanticipation (mode=causal, lookahead={args.lookahead}s):")
    for k in ("n_drops", "n_cues", "recall", "precision", "mean_lead_s",
              "baseline_recall_mean"):
        print(f"  {k:22s} {ev[k]}")
    return 0


def do_manifest(args) -> int:
    ds = AistDataset()
    metas = ds.build_manifest(limit=args.limit)
    paths = write_manifest(metas)
    print(json.loads(paths["summary"].read_text()))
    print(f"manifest: {paths['parquet']}")
    return 0


def probe_media(path: Path) -> dict:
    """ffprobe the produced file so reported properties are MEASURED."""
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {"error": r.stderr[:400]}
    return json.loads(r.stdout)


def do_render(args) -> int:
    t_start = time.time()
    wav = resolve_wav(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / f"{wav.stem}_show.mp4"

    print(f"[1/5] audio features   : {wav.name}")
    feats = extract(path=str(wav), duration_s=args.duration)
    print(f"      {feats.duration_s:.2f}s  {feats.tempo:.1f} BPM  "
          f"{feats.beats.size} beats  {feats.n_frames} frames")

    print("[2/5] structure")
    segs = segment(feats)
    for s in segs:
        print(f"      {s.start_s:7.2f}-{s.end_s:7.2f}s  {s.label}")

    print(f"[3/5] anticipation (mode={args.mode}, lookahead={args.lookahead}s)")
    antic = compute(feats, segs, mode=args.mode, lookahead_s=args.lookahead)
    ev = evaluate_anticipation(feats, segs, lookahead_s=args.lookahead)
    print(f"      cues={ev['n_cues']} drops={ev['n_drops']} "
          f"recall={ev['recall']} precision={ev['precision']} "
          f"mean_lead={ev['mean_lead_s']}")

    print(f"[4/5] DMX stream @ {args.dmx_fps} fps")
    rig = club_rig()
    stream = render_stream(rig, feats, segs, antic, fps=args.dmx_fps)
    print(f"      {stream.frames.shape} uint8  {stream.duration_s:.2f}s")
    cue_json = out.with_suffix(".cues.json")
    stream.save_cue_list(cue_json, out.with_suffix(".cues.csv"))
    stream.save_npy(out.with_suffix(".universe.npy"))

    print(f"[5/5] render {args.width}x{args.height} @ {args.fps} fps")
    r = StageRenderer(rig, RenderConfig(
        width=args.width, height=args.height, fps=args.fps, haze=args.haze))
    print(f"      GL: {r.renderer_name}")
    r.render_video(stream.frames, stream.fps, out,
                   audio_path=wav, duration_s=feats.duration_s,
                   progress_every=args.progress_every)

    elapsed = time.time() - t_start
    info = probe_media(out)
    size = out.stat().st_size
    print(f"\nWROTE {out}")
    print(f"  size      : {size/1e6:.2f} MB ({size} bytes)")
    if "streams" in info:
        for s in info["streams"]:
            if s["codec_type"] == "video":
                print(f"  video     : {s['codec_name']} {s['width']}x{s['height']} "
                      f"{s.get('r_frame_rate')} fps  {s.get('nb_frames')} frames")
            elif s["codec_type"] == "audio":
                print(f"  audio     : {s['codec_name']} {s.get('sample_rate')}Hz "
                      f"{s.get('channels')}ch")
        print(f"  duration  : {float(info['format']['duration']):.2f}s")
    print(f"  cue list  : {cue_json} ({len(stream.cues)} cues)")
    print(f"  wall time : {elapsed:.1f}s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lightsem")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--wav")
        p.add_argument("--aist", help="AIST music id, e.g. mBR0")
        p.add_argument("--duration", type=float, default=None)
        p.add_argument("--lookahead", type=float, default=2.0)

    pa = sub.add_parser("analyse", help="features + structure + anticipation")
    common(pa)
    pa.set_defaults(func=do_analyse)

    pm = sub.add_parser("manifest", help="scan AIST corpus -> parquet manifest")
    pm.add_argument("--limit", type=int, default=None)
    pm.set_defaults(func=do_manifest)

    from .dance.cli import add_options, do_dance, do_render as do_native_render
    pd = sub.add_parser("dance", help="fixed native C + MuQ -> body; optional D4-max/D6")
    add_options(pd)
    pd.add_argument("--out", type=Path, help="New run directory containing body.npz, plan.npz and provenance")
    pd.set_defaults(func=do_dance)

    pn = sub.add_parser("render", help="fixed native C + MuQ -> actual DMX + lit3D MP4")
    add_options(pn)
    pn.add_argument("--out", type=Path)
    pn.add_argument("--start", type=float, default=0., help="Render start in source seconds; generate/analyze complete song first")
    pn.add_argument("--duration", type=float, default=None, help="Render-only crop in seconds; original audio speed")
    pn.add_argument("--colors", type=Path, help="Own-song P2 color cache with authenticated REPORT.json")
    pn.add_argument("--ssl-light", action="store_true",
                    help="Use the opt-in SSL latent light adapter on top of native-C mapper-01 DMX")
    pn.add_argument("--ssl-light-checkpoint", type=Path,
                    help="Override the authenticated SSL light checkpoint")
    for name in ("beams", "lasers", "bulbs"):
        pn.add_argument("--" + name, type=int, default=None)
    pn.set_defaults(func=do_native_render)

    pr = sub.add_parser("legacy-render", help="historical rule-driven renderer, for reproducing old studies")
    common(pr)
    pr.add_argument("--out")
    pr.add_argument("--mode", choices=[CAUSAL, ORACLE], default=CAUSAL)
    pr.add_argument("--width", type=int, default=1920)
    pr.add_argument("--height", type=int, default=1080)
    pr.add_argument("--fps", type=float, default=30.0)
    pr.add_argument("--dmx-fps", type=float, default=40.0)
    pr.add_argument("--haze", type=float, default=0.115)
    pr.add_argument("--progress-every", type=int, default=60)
    pr.set_defaults(func=do_render)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
