#!/usr/bin/env python3
"""Train the standalone canonical v2 direct-head repair once, seed 0."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from lightsem.dance.pipeline import accelerator_lock
from lightsem.experiments.audio_light_foundation.data import make_fixture_metadata
from lightsem.experiments.audio_light_foundation.model import AudioToLightConfig
from lightsem.experiments.foundation_audio_light.direct_model import AudioOnlyDirectLightModel, DirectLightConfig
from lightsem.experiments.foundation_audio_light.safety_projection import SafetyConfig
from lightsem.experiments.foundation_audio_light.peft import trainable_parameter_report
from scripts.experiments.run_foundation_v3_audio_only import CONTRACT, batch, load_row, loss_parts, audit_output, sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-angle", type=float, default=0.4)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing artifact root: {args.out}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contract = json.loads(CONTRACT.read_text())
    rows = [load_row(row) for row in contract["rows"]]
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["wav_sha256"], []).append(row)
    group_keys = np.array(sorted(groups))
    group_keys = group_keys[np.random.default_rng(args.seed).permutation(len(group_keys))]
    split = max(1, int(0.8 * len(group_keys)))
    train_groups = set(group_keys[:split])
    train_rows = [row for row in rows if row["wav_sha256"] in train_groups]
    val_rows = [row for row in rows if row["wav_sha256"] not in train_groups]
    train_windows = [
        (index, start)
        for index, row in enumerate(train_rows)
        for start in range(0, max(1, len(row["audio"]) - args.window + 1), args.window)
        if start + args.window <= len(row["audio"])
    ]
    fixtures = torch.from_numpy(make_fixture_metadata()).float().to(device)
    foundation_config = AudioToLightConfig(
        audio_dim=1116, width=96, heads=4, local_window=32, latent_slots=8,
        adapter_kind="hierarchical", adapter_bottleneck=32, adapter_heads=4,
        adapter_local_window=32,
    )
    direct_config = DirectLightConfig(
        foundation=foundation_config,
        safety=SafetyConfig(dt=0.1, max_angle=args.max_angle, max_velocity=0.18,
                            max_acceleration=0.5, max_jerk=1.5),
    )
    model = AudioOnlyDirectLightModel(direct_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    history = []
    started = time.time()
    with accelerator_lock():
        model.train()
        for step in range(args.steps):
            batch_indices = [train_windows[(step * args.batch_size + j) % len(train_windows)] for j in range(args.batch_size)]
            target = batch(train_rows, batch_indices, args.window, device)
            timestamps = torch.arange(args.window, device=device, dtype=torch.float32)[None].expand(args.batch_size, -1) / 10.0
            output = model(target["audio"], fixtures[None].expand(args.batch_size, -1, -1), timestamps=timestamps,
                           valid=target["valid"], duration_seconds=args.window / 10.0)
            parts = loss_parts(output, target)
            optimizer.zero_grad(set_to_none=True)
            parts["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 0 or (step + 1) % max(1, args.steps // 4) == 0:
                history.append({"step": step + 1, **{key: float(value.detach().cpu()) for key, value in parts.items()}})
        model.eval()
        val_audits = []
        predictions = {}
        with torch.no_grad():
            for row in val_rows[:4]:
                n = min(len(row["audio"]), 512)
                audio = torch.from_numpy(row["audio"][:n][None]).float().to(device)
                valid = torch.ones((1, n), dtype=torch.bool, device=device)
                timestamps = torch.arange(n, device=device, dtype=torch.float32)[None] / 10.0
                output = model(audio, fixtures[None], timestamps=timestamps, valid=valid, duration_seconds=n / 10.0)
                val_audits.append({"id": row["id"], **audit_output(output, 10.0)})
                predictions[row["id"]] = {key: value.detach().cpu().numpy() for key, value in output.items()
                                            if key in {"aim", "rgb", "brightness", "event", "audio_latent", "anchor"}}
            if len(val_rows) >= 2:
                n = min(len(val_rows[0]["audio"]), len(val_rows[1]["audio"]), 512)
                a0 = torch.from_numpy(val_rows[0]["audio"][:n][None]).float().to(device)
                a1 = torch.from_numpy(val_rows[1]["audio"][:n][None]).float().to(device)
                valid = torch.ones((1, n), dtype=torch.bool, device=device)
                timestamps = torch.arange(n, device=device, dtype=torch.float32)[None] / 10.0
                own = model(a0, fixtures[None], timestamps=timestamps, valid=valid, duration_seconds=n / 10.0)
                repeat = model(a0, fixtures[None], timestamps=timestamps, valid=valid, duration_seconds=n / 10.0)
                swapped = model(a1, fixtures[None], timestamps=timestamps, valid=valid, duration_seconds=n / 10.0)
                deterministic = max(float((own[key] - repeat[key]).abs().max().cpu()) for key in ("aim", "rgb", "brightness", "event"))
                audio_swap = {key: float((own[key] - swapped[key]).abs().mean().cpu()) for key in ("aim", "rgb", "brightness", "event")}
            else:
                deterministic, audio_swap = None, {}
    args.out.mkdir(parents=True, exist_ok=False)
    for sid, values in predictions.items():
        np.savez_compressed(args.out / f"PREDICTIONS_{sid}.npz", time=np.arange(len(values["aim"][0])) / 10.0, **values)
    checkpoint = args.out / "model.pt"
    torch.save({"state_dict": model.state_dict(), "config": model.config_dict(), "seed": args.seed,
                "input_contract": "canonical MuQ1024 layer9 + Base92 92 = 1116; audio-only inference"}, checkpoint)
    parameterized_raw = all(item["raw_safety_pass"] for item in val_audits)
    output_gate = all(item["output_collapse_pass"] for item in val_audits)
    status = "diagnostic_candidate" if parameterized_raw and output_gate else "rejected_raw_safety" if not parameterized_raw else "output_collapsed"
    receipt = {
        "schema": "lightsem.foundation_v2_repair.v1",
        "protocol_id": "foundation-v2-repair-direct-bounded-20261020-v1",
        "status": status,
        "promotion": "forbidden_until_raw_fd_safety_own_song_mmca_decoded_dmx_and_human_gates",
        "seed": args.seed, "steps": args.steps, "device": str(device),
        "train_groups": len(train_groups), "validation_groups": len(group_keys) - len(train_groups),
        "train_rows": len(train_rows), "validation_rows": len(val_rows),
        "input_contract": "canonical MuQ1024 layer9 + Base92 92 = 1116; no zero padding",
        "body_inference_input": False, "muq_finetuned": False,
        "architecture": "direct_audio_only_bounded_head",
        "trainable_parameters": trainable_parameter_report(model),
        "config": model.config_dict(),
        "history": history, "validation_audits": val_audits,
        "deterministic_max_abs_diff": deterministic, "audio_swap_mean_abs_delta": audio_swap,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "muq_receipt": str(Path("/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_v2_muq_20261016_v3/COMPLETE_V2.json")),
        "muq_receipt_sha256": sha(Path("/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_v2_muq_20261016_v3/COMPLETE_V2.json")),
        "runtime_seconds": time.time() - started,
        "promotion_reason": "parameterized component safety is not a substitute for the fresh unfiltered finite-difference raw audit",
    }
    (args.out / "TRAINING.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.out.resolve()), "status": status,
                      "validation": val_audits, "audio_swap": audio_swap}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
