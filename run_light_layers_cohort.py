"""CPU-only, source-completeness qualification. Never opens TEST raw inputs.

The output is a CANDIDATE cohort, not authorization to extract or train.
Historical VAL is exposed development data, never fresh confirmation.
No target, confidence, mask, identity, cut, or model-score selection is done.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

CAMPAIGN_ROOT = Path("/mnt/lightsem-data/experiments/lightsem/lighting-cue-model-20260922t094347z")
MANIFEST_PATH = Path("/mnt/lightsem-data/experiments/lightsem/stage11_20260908/data/manifest.json")
QUOTAS = {"train": 100, "val": 40}
DURATION_S = 64.0
PADDING_S = .2
MAX_GAP_S = .2
RAW_KEYS = ("source_path", "pose_path", "wav_path", "audio_path")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def canonical_order_key(campaign_id, canonical_id):
    """Exact UTF-8 concatenation, with no separators or hidden RNG state."""
    return hashlib.sha256((campaign_id + "cohort-v1" + canonical_id).encode("utf-8")).hexdigest()


def centered_window(active_start_s, active_end_s, source_duration_s):
    values = [float(v) for v in (active_start_s, active_end_s, source_duration_s)]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite_duration_or_active_span")
    lo, hi, duration = values
    if duration < DURATION_S:
        raise ValueError("source_duration_below_64s")
    if lo < 0 or hi < lo or hi > duration + .1:
        raise ValueError("invalid_active_span")
    if hi - lo < DURATION_S:
        raise ValueError("active_span_below_64s")
    center_start = (Decimal(str(active_start_s)) + Decimal(str(active_end_s))
                    - Decimal(str(DURATION_S))) / 2
    start = float(center_start.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    if start < 0 or start + DURATION_S > duration + 1e-9:
        raise ValueError("rounded_window_outside_source")
    return start, DURATION_S


def validate_split(recordings, split):
    """Fail closed on split disagreement before opening any source files."""
    indexed, canonical_partitions = {}, defaultdict(set)
    for row in recordings:
        rid = row["recording_id"]
        if rid in indexed:
            raise ValueError("duplicate recording_id: " + rid)
        indexed[rid] = row
        registered = split["recordings"].get(rid)
        if registered != {"partition": row["partition"], "canonical_id": row["canonical_id"]}:
            raise ValueError("historical split disagreement: " + rid)
        if row["partition"] not in ("train", "val", "test"):
            raise ValueError("unknown historical partition: " + rid)
        canonical_partitions[row["canonical_id"]].add(row["partition"])
        for coach in row.get("coaches", []):
            if coach.get("partition", row["partition"]) != row["partition"]:
                raise ValueError("coach split disagreement: " + rid)
    if any(len(parts) != 1 for parts in canonical_partitions.values()):
        raise ValueError("canonical family crosses historical partitions")
    for partition in ("val", "test"):
        if partition in split:
            expected = set(split[partition])
            observed = {cid for cid, parts in canonical_partitions.items() if partition in parts}
            if expected != observed:
                raise ValueError("historical canonical split disagreement: " + partition)


class SourceAccess:
    """A read allowlist tied to historical metadata; TEST wins on path aliases."""
    def __init__(self, recordings, split):
        validate_split(recordings, split)
        self.rows = {row["recording_id"]: row for row in recordings}
        self.test_paths = {str(Path(row[key]).resolve())
                           for row in recordings if row["partition"] == "test"
                           for key in RAW_KEYS if row.get(key)}
        self.access_counts = Counter()
        self.accessed_recordings = defaultdict(set)

    def authorize(self, row, key):
        registered = self.rows.get(row.get("recording_id"))
        if (registered is None or row.get("partition") not in QUOTAS
                or registered["partition"] not in QUOTAS
                or row["partition"] != registered["partition"]
                or not registered.get("eligible_core", False)):
            raise PermissionError("TEST or noneligible source read forbidden")
        if key not in RAW_KEYS or row.get(key) != registered.get(key) or not row.get(key):
            raise PermissionError("source path outside registered allowlist")
        path = Path(row[key])
        if str(path.resolve()) in self.test_paths:
            raise PermissionError("TEST source path alias forbidden")
        return path

    def open(self, row, key):
        path = self.authorize(row, key)
        stream = path.open("rb")
        self.access_counts[row["partition"] + ":" + key] += 1
        self.accessed_recordings[row["partition"]].add(row["recording_id"])
        return stream

    def readable(self, row, key):
        with self.open(row, key) as stream:
            if not stream.read(1):
                raise ValueError("empty_file")
            stat = os.fstat(stream.fileno())
        return {"path": row[key], "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def digest(self, row, key):
        digest = hashlib.sha256()
        with self.open(row, key) as stream:
            for block in iter(lambda: stream.read(4 * 1024**2), b""):
                digest.update(block)
        return digest.hexdigest()


def qualify_pose_arrays(times, boxes, coach_indices, start_s, duration_s=DURATION_S):
    """Only timestamp coverage and raw finite positive-area registered boxes.

    Check all pose frames in the padded interval, including the immediately
    bracketing frames needed to cover its edges. No interpolation or filling.
    Confidence/keypoint arrays are deliberately not inputs to this function.
    """
    reasons, detail = [], {}
    times = np.asarray(times, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim == 2:
        boxes = boxes[:, None, :]
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        return ["pose_times_not_finite_strictly_increasing"], detail
    if boxes.ndim != 3 or boxes.shape[-1] != 4 or len(boxes) != len(times):
        return ["pose_box_shape_invalid"], detail
    if (not coach_indices or len(set(coach_indices)) != len(coach_indices)
            or any(type(i) is not int or i < 0 or i >= boxes.shape[1] for i in coach_indices)):
        return ["registered_coach_indices_invalid"], detail
    lo, hi = start_s - PADDING_S, start_s + duration_s + PADDING_S
    extra_indices = sorted(set(range(boxes.shape[1])) - set(coach_indices))
    detail.update(pose_frames_total=len(times), pose_coaches=boxes.shape[1],
                  registered_coach_indices=coach_indices,
                  unregistered_pose_slot_indices=extra_indices,
                  unregistered_pose_slot_count=len(extra_indices), checked_interval_s=[lo, hi],
                  pose_first_s=float(times[0]), pose_last_s=float(times[-1]))
    if times[0] > lo + 1e-9 or times[-1] < hi - 1e-9:
        return ["pose_times_do_not_cover_padded_window"], detail
    first = max(0, int(np.searchsorted(times, lo + 1e-9, side="right")) - 1)
    last = min(len(times) - 1, int(np.searchsorted(times, hi - 1e-9, side="left")))
    selected_times = times[first:last + 1]
    max_gap = float(np.max(np.diff(selected_times)))
    detail.update(first_checked_pose_index=first, last_checked_pose_index=last,
                  checked_pose_frames=len(selected_times), max_gap_s=max_gap)
    if max_gap > MAX_GAP_S + 1e-9:
        reasons.append("pose_gap_above_0.2s")
    selected_boxes = boxes[first:last + 1][:, coach_indices, :]
    valid = np.isfinite(selected_boxes).all(axis=2) & (selected_boxes[:, :, 2:] > selected_boxes[:, :, :2]).all(axis=2)
    invalid_each = (~valid).sum(axis=0)
    detail["invalid_box_frames_per_registered_coach"] = {
        str(coach): int(count) for coach, count in zip(coach_indices, invalid_each)}
    detail["invalid_box_frames_any_registered_coach"] = int(np.count_nonzero(~valid.all(axis=1)))
    # Supplemental policy is fixed before source reads and reported separately.
    # Presence means ALL registered boxes jointly valid, not mean coach coverage.
    for label, begin, end in (("window64s", start_s, start_s + duration_s),
                              ("center8s", start_s + duration_s/2 - 4,
                               start_s + duration_s/2 + 4)):
        inside = (selected_times >= begin - 1e-9) & (selected_times < end - 1e-9)
        frame_valid = valid[inside].all(axis=1)
        detail[label + "_presence"] = {
            "interval_s": [begin, end], "interval_convention": "half-open",
            "pose_frames": int(len(frame_valid)), "all_registered_valid_frames": int(frame_valid.sum()),
            "fraction": float(frame_valid.mean()) if len(frame_valid) else 0.0}
    if not valid.all():
        reasons.append("registered_coach_box_missing_nonfinite_or_nonpositive")
    return reasons, detail


def coverage90_qualification(strict_result):
    """Separate supplemental policy. Never changes the strict qualification."""
    reasons = [reason for reason in strict_result["reasons"]
               if reason != "registered_coach_box_missing_nonfinite_or_nonpositive"]
    detail = strict_result["detail"]
    for label in ("window64s", "center8s"):
        presence = detail.get("pose", {}).get(label + "_presence", {})
        if presence.get("fraction", 0.) < .90:
            reasons.append(label + "_registered_coach_presence_below_0.90")
    return {"qualified": not reasons, "reasons": reasons, "detail": detail,
            "strict_qualified": strict_result["qualified"],
            "policy": "supplemental_coverage90_joint_registered_box_presence"}


def qualify_record(row, access):
    # Guard before stat/open/window handling. Even malformed TEST rows cannot read.
    registered = access.rows.get(row.get("recording_id"))
    if (registered is None or registered["partition"] not in QUOTAS
            or row.get("partition") != registered["partition"]
            or not registered.get("eligible_core", False)):
        raise PermissionError("TEST or noneligible source read forbidden")
    detail, reasons = {"readable_files": {}}, []
    try:
        start, duration = centered_window(row["active_start_s"], row["active_end_s"], row["duration_s"])
        detail.update(start_s=start, duration_s=duration)
    except (KeyError, ValueError, TypeError) as error:
        return {"qualified": False, "reasons": [str(error)], "detail": detail}
    for key in RAW_KEYS:
        try:
            detail["readable_files"][key] = access.readable(row, key)
        except (OSError, ValueError, PermissionError) as error:
            reasons.append("unreadable_" + key)
            detail.setdefault("read_errors", {})[key] = str(error)
    if "pose_path" in detail["readable_files"]:
        try:
            with access.open(row, "pose_path") as stream, np.load(stream, allow_pickle=False) as pose:
                # Do not load confidence, keypoints, target features or predictions.
                times, boxes = pose["t"], pose["box"]
                meta = json.loads(str(pose["meta"]))
            coach_indices = [coach["coach_index"] for coach in row["coaches"]]
            pose_reasons, pose_detail = qualify_pose_arrays(times, boxes, coach_indices, start, duration)
            reasons.extend(pose_reasons)
            detail["pose"] = pose_detail
            width, height = int(meta["width"]), int(meta["height"])
            detail.update(original_width=width, original_height=height)
            if min(width, height) <= 0:
                reasons.append("nonpositive_pose_geometry")
            expected = row.get("pose_provenance", {})
            if (width, height) != (expected.get("width"), expected.get("height")):
                reasons.append("pose_geometry_differs_from_manifest")
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            reasons.append("unreadable_or_malformed_pose_arrays")
            detail["pose_error"] = str(error)
    return {"qualified": not reasons, "reasons": reasons, "detail": detail}


def choose_groups(recordings, split, campaign_id, qualify, quotas=None):
    """Pure deterministic ordering/selection, with an injected source-only check.

    Qualify all eligible TRAIN/VAL candidates even after a quota is filled, so
    exclusions are complete. Select the first qualifying recording per family
    and the first quota families. Never replace a row using model outcomes.
    """
    quotas = QUOTAS if quotas is None else quotas
    validate_split(recordings, split)
    groups = defaultdict(list)
    for row in recordings:
        groups[row["canonical_id"]].append(row)
    selected, decisions = [], []
    counts = Counter()
    for cid in sorted(groups, key=lambda cid: (canonical_order_key(campaign_id, cid), cid)):
        family_selected = False
        for row in sorted(groups[cid], key=lambda row: row["recording_id"]):
            decision = {"recording_id": row["recording_id"], "canonical_id": cid,
                        "partition": row["partition"], "canonical_order_sha256": canonical_order_key(campaign_id, cid),
                        "source_metadata": row}
            if row["partition"] == "test":
                decision.update(status="not_opened_test", reasons=["TEST_raw_inputs_forbidden"])
            elif not row.get("eligible_core", False):
                decision.update(status="metadata_ineligible", reasons=["eligible_core_false"])
            else:
                result = qualify(row)
                decision["qualification"] = result
                if not result["qualified"]:
                    decision.update(status="source_excluded", reasons=result["reasons"])
                elif family_selected:
                    decision.update(status="qualified_not_selected_same_canonical", reasons=["one_recording_per_canonical"])
                elif counts[row["partition"]] >= quotas[row["partition"]]:
                    decision.update(status="qualified_not_selected_quota", reasons=["fixed_partition_quota_reached"])
                    family_selected = True
                else:
                    counts[row["partition"]] += 1
                    family_selected = True
                    decision.update(status="selected_candidate", reasons=[])
                    selected.append((row, result))
            decisions.append(decision)
    return selected, decisions


def hash_record(row, qualification, access, artifact_hashes, campaign_root):
    pose_hash = access.digest(row, "pose_path")
    if pose_hash != row["pose_sha256"]:
        raise ValueError("selected pose hash mismatch; stop without replacement: " + row["recording_id"])
    verified = {}
    for key in ("wav_path", "audio_path"):
        actual = access.digest(row, key)
        expected = artifact_hashes.get(row[key])
        if expected is not None and actual != expected:
            raise ValueError("selected audio hash mismatch; stop without replacement: " + row["recording_id"])
        verified[key] = actual
    expected_source_hash = row["source_sha256"]
    if len(expected_source_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_source_hash):
        raise ValueError("missing/malformed expected source hash")
    partition = row["partition"]
    rhythm = (campaign_root / "rhythm" / "features" if partition == "train"
              else campaign_root / "rhythm_dev_features") / (row["recording_id"] + ".npz")
    detail = qualification["detail"]
    result = {"recording_id": row["recording_id"], "canonical_id": row["canonical_id"],
              "partition": partition, "historical_partition": partition,
              "partition_role": "historical_TRAIN" if partition == "train" else "historical_VAL_exposed_development",
              "source_path": row["source_path"], "source_sha256": expected_source_hash,
              "source_hash_verification": "expected_from_stage11_manifest; verify_deferred_to_extractor",
              "pose_path": row["pose_path"], "pose_sha256": pose_hash,
              "pose_hash_verification": "verified_during_source_qualification",
              "audio_path": row["wav_path"], "audio_sha256": verified["wav_path"],
              "audio_bus_path": row["audio_path"], "audio_bus_sha256": verified["audio_path"],
              "audio_hash_verification": "WAV_and_bus_verified_during_source_qualification",
              "legacy_visual_path": row["visual_path"], "rhythm_path": str(rhythm),
              "rhythm_status": "existing_TRAIN_features_not_read" if partition == "train" else "planned_new_development_features_not_extracted",
              "coaches": row["coaches"], "start_s": detail["start_s"], "duration_s": detail["duration_s"],
              "original_width": detail["original_width"], "original_height": detail["original_height"],
              "source_duration_s": row["duration_s"],
              "active_start_s": row["active_start_s"], "active_end_s": row["active_end_s"],
              "source_file_metadata": detail["readable_files"],
              "source_qualification": detail["pose"]}
    result["source_metadata_sha256"] = object_sha256(row)
    result["input_metadata_sha256"] = object_sha256(result)
    return result


def write_json_exclusive(path, value):
    """Atomic no-clobber publication, including a completed-file hash in caller."""
    partial = path.with_name(path.name + ".partial")
    with partial.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(partial, path)
    partial.unlink()


def protocol(campaign_id):
    return {"campaign_id": campaign_id, "cohort_version": "cohort-v1", "quotas": QUOTAS,
            "canonical_order": "SHA256(UTF8(campaign_id + 'cohort-v1' + canonical_id)); ascending hexdigest",
            "recording_order": "ascending lexicographic recording_id within canonical_id",
            "one_recording_per_canonical": True,
            "eligibility": "historical TRAIN/VAL eligible_core metadata only; preserve original membership",
            "window": {"duration_s": DURATION_S, "location": "center of registered active span",
                       "start_rounding": "global 0.1 second grid; Decimal ROUND_HALF_UP"},
            "qualification": {"padding_s": PADDING_S, "max_pose_gap_s": MAX_GAP_S,
                              "all_registered_coaches_every_pose_frame": True,
                              "registered_coaches": "explicit row.coaches[].coach_index; nonempty, unique, in range",
                              "unregistered_pose_slots": "reported but not source-validity gates; no independent track-quality claim",
                              "padded_interval_edges": "include immediately bracketing pose frames; timestamps cover both edges",
                              "boxes": "raw finite XYXY with strictly positive width and height; no fill or interpolation",
                              "paths": "source video, pose, raw WAV, audio bus must be readable/nonempty",
                              "duration": "source duration and active span at least 64s; rounded window inside source",
                              "geometry": "positive pose dimensions matching manifest metadata"},
            "prohibited_selection_inputs": ["confidence", "targets", "model scores", "target histograms", "TEST raw inputs"],
            "scope": "source-box completeness only; NOT independent mask, identity or cut accuracy",
            "VAL_role": "historical_VAL_exposed_development; never fresh confirmation",
            "replacements": "none based on model/target results; selected hash failure stops without replacement",
            "authorization": "candidate_pending_parent_preregistration; NOT frozen for extraction"}


def coverage90_protocol(campaign_id):
    rules = protocol(campaign_id)
    rules["policy"] = "supplemental_coverage90_joint_registered_box_presence"
    rules["qualification"]["all_registered_coaches_every_pose_frame"] = False
    rules["qualification"]["presence_threshold"] = .90
    rules["qualification"]["presence_measure"] = "fraction of raw pose frames with ALL registered boxes jointly finite and positive-area"
    rules["qualification"]["presence_intervals"] = "BOTH half-open full64s and centered8s, independently >=0.90"
    rules["scope"] = "supplemental source-box coverage policy; NOT independent mask, identity or cut accuracy"
    rules["required_extractor_policy"] = "invalidate every missing-registered-box frame; no imputed masks or labels"
    rules["strict_cohort"] = "retained separately and not silently replaced"
    rules["scientific_gate"] = "original >=80% eligible historical DEV canonical groups remains unchanged; not assessed here"
    return rules


def build_cohort(manifest_path, split_path, campaign_root):
    started = time.monotonic()
    campaign_root = Path(campaign_root)
    paths = {"cohort": campaign_root / "cohort_v1.json",
             "report": campaign_root / "cohort_v1_qualification_report.json",
             "protocol": campaign_root / "cohort_v1_selection_protocol.json",
             "supplemental_cohort": campaign_root / "cohort_v1_coverage90_candidate.json",
             "supplemental_report": campaign_root / "cohort_v1_coverage90_qualification_report.json",
             "supplemental_protocol": campaign_root / "cohort_v1_coverage90_selection_protocol.json"}
    for path in paths.values():
        if path.exists() or path.with_name(path.name + ".partial").exists():
            raise FileExistsError("refuse to overwrite cohort artifacts: " + str(path))
    manifest = json.loads(Path(manifest_path).read_text())
    split = json.loads(Path(split_path).read_text())
    access = SourceAccess(manifest["recordings"], split)
    rules = protocol(campaign_root.name)
    supplemental_rules = coverage90_protocol(campaign_root.name)
    dependency_hashes = {str(manifest_path): sha256_file(manifest_path),
                         str(split_path): sha256_file(split_path),
                         str(Path(__file__).resolve()): sha256_file(__file__)}
    campaign_root.mkdir(parents=True, exist_ok=True)
    # Publish BOTH fixed policies before the first raw source qualification read.
    for key, policy in (("protocol", rules), ("supplemental_protocol", supplemental_rules)):
        write_json_exclusive(paths[key], {"status": "selection_rules_fixed_before_source_screening",
                                         "protocol": policy, "dependency_hashes": dependency_hashes})
    progress, qualifications = Counter(), {}

    def qualify(row):
        result = qualify_record(row, access)
        qualifications[row["recording_id"]] = result
        progress[row["partition"]] += 1
        if sum(progress.values()) % 50 == 0:
            print(json.dumps({"source_candidates_checked": dict(progress), "elapsed_s": round(time.monotonic()-started, 2)}), flush=True)
        return result

    selected, decisions = choose_groups(manifest["recordings"], split, campaign_root.name, qualify)
    supplemental_selected, supplemental_decisions = choose_groups(
        manifest["recordings"], split, campaign_root.name,
        lambda row: coverage90_qualification(qualifications[row["recording_id"]]))
    print(json.dumps({"source_screening_complete": True,
                      "strict_selected_before_hash_verification": len(selected),
                      "supplemental_selected_before_hash_verification": len(supplemental_selected)}), flush=True)
    # Hash the union once. Any integrity failure stops; it does not trigger replacements.
    verified_rows = {}
    for row, qualification in selected + supplemental_selected:
        if row["recording_id"] not in verified_rows:
            verified_rows[row["recording_id"]] = hash_record(row, qualification, access, manifest["artifact_hashes"], campaign_root)
    summaries = {}
    for prefix, chosen, candidate_decisions, policy in (
            ("", selected, decisions, rules),
            ("supplemental_", supplemental_selected, supplemental_decisions, supplemental_rules)):
        rows = [verified_rows[row["recording_id"]] for row, _ in chosen]
        counts = Counter(row["partition"] for row in rows)
        qualified_families = {part: len({decision["canonical_id"] for decision in candidate_decisions
                                       if decision["partition"] == part and decision.get("qualification", {}).get("qualified")})
                              for part in QUOTAS}
        summary = {part: {"requested": quota, "selected": counts[part],
                          "qualified_canonical_families": qualified_families[part],
                          "shortfall": max(0, quota-counts[part]),
                          "sufficient": counts[part] >= quota}
                   for part, quota in QUOTAS.items()}
        summaries[prefix + "summary"] = summary
        cohort_path = paths[prefix + "cohort"]
        report_path = paths[prefix + "report"]
        protocol_path = paths[prefix + "protocol"]
        cohort = {"schema_version": 1, "campaign_id": campaign_root.name,
                  "status": "candidate_pending_parent_preregistration", "frozen_for_extraction": False,
                  "scope": policy["scope"], "protocol": policy, "summary": summary,
                  "dependency_hashes": dependency_hashes,
                  "selection_protocol_path": str(protocol_path),
                  "selection_protocol_sha256": sha256_file(protocol_path), "inputs": rows}
        write_json_exclusive(cohort_path, cohort)
        report = {"schema_version": 1, "campaign_id": campaign_root.name,
                  "status": "candidate_pending_parent_preregistration", "frozen_for_extraction": False,
                  "summary": summary, "scope": policy["scope"], "protocol": policy,
                  "cohort_path": str(cohort_path), "cohort_sha256": sha256_file(cohort_path),
                  "dependency_hashes": dependency_hashes,
                  "recording_decision_counts": dict(Counter(decision["status"] for decision in candidate_decisions)),
                  "source_exclusion_reason_counts": dict(Counter(reason for decision in candidate_decisions
                      if decision["status"] == "source_excluded" for reason in decision["reasons"])),
                  "raw_open_counts": dict(access.access_counts),
                  "raw_access_audit_scope": "shared strict and supplemental source screening plus selected hash-verification union",
                  "raw_accessed_recordings": {part: sorted(rids) for part, rids in access.accessed_recordings.items()},
                  "TEST_raw_open_count": sum(count for key, count in access.access_counts.items() if key.startswith("test:")),
                  "decisions": candidate_decisions, "elapsed_s": time.monotonic()-started}
        write_json_exclusive(report_path, report)
    return {**summaries, "files": {key: str(path) for key, path in paths.items()},
            "status": "candidate_pending_parent_preregistration", "TEST_raw_open_count": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--campaign-root", type=Path, default=CAMPAIGN_ROOT)
    args = parser.parse_args(argv)
    split = args.split_manifest or args.manifest.with_name("split_manifest.json")
    result = build_cohort(args.manifest, split, args.campaign_root)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
