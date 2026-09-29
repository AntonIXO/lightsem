#!/usr/bin/env python3
"""Write LightSem findings into the LIGHTSEM vault over MCP stdio.

The vibescience-lightsem tool namespace only appears after a gateway restart,
so this talks to the server directly with VIBESCIENCE_VAULT pinned, which
guarantees records land in /root/lightsem-vault and never in the optiHealth
vault.

PITFALL that cost a debugging cycle: feeding all requests via
subprocess.run(input=...) closes stdin immediately, and the server exits on
EOF after replying to `initialize` only -- the tools/call responses are never
produced. stdin must be held OPEN and each response read back incrementally.
"""
import json
import os
import subprocess
import sys

SERVER = "/root/vibescience-mcp/.venv/bin/vibescience-mcp"
VAULT = "/root/lightsem-vault"
PROBLEM = "audio-to-light-semantic-mapping-20260807"


class Client:
    def __init__(self):
        env = dict(os.environ)
        env["VIBESCIENCE_VAULT"] = VAULT
        env["VIBESCIENCE_SERVER_NAME"] = "vibescience-lightsem"
        self.p = subprocess.Popen(
            [SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
        self._id = 0

    def _send(self, obj):
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()

    def _read(self):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("server closed stdout")
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)

    def initialize(self):
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                               "clientInfo": {"name": "writer", "version": "1"}}})
        r = self._read()
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return r["result"]["serverInfo"]

    def call(self, name, args):
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": "tools/call",
                    "params": {"name": name, "arguments": args}})
        msg = self._read()
        txt = msg.get("result", {}).get("content", [{}])[0].get("text", "")
        try:
            d = json.loads(txt)
            if isinstance(d.get("result"), str):
                d = json.loads(d["result"])
            return d.get("data", d)
        except Exception:
            return txt

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


RULE_OUT_CORPUS = {
    "problem_id": PROBLEM,
    "claim": (
        "AIST Dance DB audio is a usable corpus for validating EDM structure "
        "detection and anticipatory (build-up -> drop) lighting control. "
        "ELIMINATED. AIST tracks are short, loudness-normalised dance loops "
        "whose energy dynamic range is ~5x too small to contain the "
        "build-up/drop structure the anticipation layer exists to detect. "
        "Also eliminated as the SOLE explanation of the weak first result: "
        "'the segmentation/anticipation detector is broken' -- a synthetic "
        "control proves the detector localises real drops to <1 s."
    ),
    "method": (
        "Two-hypothesis discrimination with a SYNTHETIC CONTROL SIGNAL. The "
        "first anticipation run on 5 AIST tracks gave recall 0.000 on 3/5, and "
        "on the one track scoring 0.800 the rate-matched random baseline was "
        "0.625 (i.e. no skill). Two competing explanations were separated "
        "rather than assumed: H1 detector broken; H2 corpus has no structure.\n\n"
        "scripts/diagnostics/probe_control_synthetic.py synthesises EDM-shaped audio (kick on the "
        "beat, rising riser + amplitude ramp through an 8 s build, wide bright "
        "drop section, quiet breakdown) with drops at KNOWN times 16.0 s and "
        "36.0 s, and runs the IDENTICAL librosa laplacian-segmentation + "
        "labelling + anticipation pipeline over it. Energy dynamic range (RMS "
        "p90/p10, std/mean) is measured on the synthetic track and on 10 real "
        "AIST tracks for direct comparison.\n\n"
        "scripts/diagnostics/probe_anticipation_baselines.py then prices the anticipation claim "
        "against THREE nulls rather than a bare number: (1) rate-matched "
        "uniform-random cues, 500 draws; (2) a metronome cue process at the "
        "same rate, 20 phases; (3) a NULL TRACK of steady-state audio with no "
        "drops, to expose a trigger-happy detector. Scoring uses precision and "
        "nearest-cue lead time, because recall saturates for every method once "
        "the credit window is wide."
    ),
    "evidence": (
        "=== ENERGY DYNAMIC RANGE (the decisive contrast) ===\n"
        "  synthetic EDM control : RMS p90/p10 = 18.27   std/mean = 0.726\n"
        "  AIST median of 10     : RMS p90/p10 =  3.76   std/mean ~ 0.45\n"
        "  per-track AIST p90/p10: mBR0 6.45, mBR1 4.16, mBR2 3.45, mBR3 4.06,\n"
        "                          mBR4 4.79, mBR5 5.42, mHO0 2.95, mHO1 2.68,\n"
        "                          mHO2 2.86, mHO3 3.24\n"
        "AIST segment energies sit inside a 0.475-0.518 band on a 0-1 "
        "normalised scale -- essentially no build-up/drop contour exists.\n\n"
        "=== H1 REFUTED: the detector works on audio that HAS structure ===\n"
        "After fixing a labelling defect (below), drop localisation error "
        "against the known synthetic drops is 0.44 s and 0.57 s.\n"
        "Anticipation on the synthetic track, scored honestly:\n"
        "  REAL      precision 1.000   nearest-lead 2.78 s +- 0.08 s\n"
        "  RANDOM    precision 0.495 +- 0.208   nearest-lead 4.36 s +- 2.48 s\n"
        "  METRONOME precision 0.500 +- 0.106   nearest-lead 3.37 s +- 1.66 s\n"
        "  precision z vs random = +2.42\n"
        "The lead-time STD is the sharpest discriminator: 0.08 s real vs 2.48 s "
        "random. It is not merely hitting the window, it is hitting a "
        "consistent phase of the build.\n\n"
        "=== NULL TRACK CONTROL (not trigger-happy) ===\n"
        "On steady-state drop-free audio the cue rate collapses 0.146 Hz -> "
        "0.021 Hz. RESIDUAL WEAKNESS, NOT HIDDEN: the labeller still tags 3 "
        "spurious 'drop' segments on that audio, so boundary detection is solid "
        "while the functional labeller is the weak link.\n\n"
        "=== WHY RECALL WAS THE WRONG HEADLINE ===\n"
        "At 7 cues / 48 s with a 12 s credit window, rate-matched random cues "
        "reach 0.816 recall vs the real detector's 0.875 -- a margin near zero. "
        "Recall saturates and cannot separate methods; precision and lead-time "
        "concentration can, and do (z = +2.42).\n\n"
        "=== THREE REAL BUGS THE CONTROL EXPOSED (none visible from AIST) ===\n"
        "1. librosa.util.sync emits one MORE column than beat frames, so a "
        "cluster-change index could equal len(beat_times) -> IndexError crash.\n"
        "2. QUANTILE labelling mislabels loud segments: a 0.613-energy segment "
        "was labelled 'breakdown' while the track max was 0.620, because it sat "
        "below the 0.66 quantile. Localisation error was 3.23 s / 6.66 s. Fixed "
        "by labelling RELATIVE TO THE TRACK'S OWN ENERGY RANGE plus an explicit "
        "drop-follows-build-up rule -> 0.44 s / 0.57 s.\n"
        "3. THE METRIC PUNISHED BETTER ANTICIPATION (see the companion "
        "rule-out).\n\n"
        "=== CONSEQUENCE ===\n"
        "AIST stays correct for pose/choreography work and is fine as a "
        "RENDERING input, but structure-detection and anticipation claims must "
        "be validated on EDM-98 / EDM-CUE (arXiv 2603.08759, HF "
        "disco-eth/edm-cue), NOT yet downloaded. Any anticipation number quoted "
        "on AIST audio is uninformative about the mechanism."
    ),
}

RULE_OUT_METRIC = {
    "problem_id": PROBLEM,
    "claim": (
        "A single scalar recall against a fixed credit window is an adequate "
        "metric for anticipatory lighting control. ELIMINATED on two "
        "independent grounds: (a) a window tied to the emission lookahead "
        "actively PENALISES earlier, better anticipation; (b) at realistic cue "
        "rates a rate-matched random process scores ~0.82 recall, so the metric "
        "cannot separate a working detector from noise."
    ),
    "method": (
        "scripts/diagnostics/probe_cue_trace.py dumped actual cue onset times against true drop "
        "times at two lookahead settings instead of trusting the aggregate. "
        "scripts/diagnostics/probe_anticipation_baselines.py then re-scored with precision and "
        "nearest-cue lead time against rate-matched random (500 draws), a "
        "metronome process (20 phases), and a drop-free null track."
    ),
    "evidence": (
        "TRACE at lookahead 2.0 s, true drops 16.0 s / 36.0 s:\n"
        "  cue onsets 9.71, 12.68, 13.14, 30.02, 32.37, 32.86, 33.30\n"
        "  leads to drop 16.0: 6.29, 3.32, 2.86\n"
        "  leads to drop 36.0: 5.98, 3.63, 3.14, 2.70\n"
        "Every drop HAD qualifying cues. The 3.0 s window (lookahead_s + "
        "tolerance_s) scored the 6.29 s lead as a MISS, so recall printed 0.000 "
        "for a system cueing 2.7-6.5 s ahead of every drop. The metric rewarded "
        "LATER warning -- backwards for a system whose purpose is leading the "
        "build-up.\n\n"
        "SATURATION: at 7 cues / 48 s with a 12 s window, random recall = 0.816 "
        "+- 0.192 vs real 0.875. Unusable as a discriminator.\n\n"
        "WHAT REPLACED IT (and does discriminate):\n"
        "  precision    real 1.000 vs random 0.495 +- 0.208  (z = +2.42)\n"
        "  nearest lead real 2.78 s +- 0.08 s vs random 4.36 s +- 2.48 s\n"
        "  null-track cue rate 0.146 Hz -> 0.021 Hz\n"
        "FIX: separate max_lead_s (how early still counts as anticipatory, "
        "default 12 s) from lookahead_s (when the rig fires). Recall 0.000 -> "
        "0.875 with no change to the detector itself.\n\n"
        "GENERAL LESSON: a metric that disagrees with a visibly working system "
        "is a suspect metric. The per-cue trace settled it; the aggregate never "
        "would have."
    ),
}


def main():
    c = Client()
    info = c.initialize()
    print("connected to:", info)
    assert info["name"] == "vibescience-lightsem", "wrong server!"

    for label, payload in (("corpus", RULE_OUT_CORPUS), ("metric", RULE_OUT_METRIC)):
        out = c.call("rule_out", payload)
        rid = out.get("rule_out_id") if isinstance(out, dict) else None
        print(f"  rule_out[{label}] -> {rid or str(out)[:200]}")

    listing = c.call("list_rule_outs", {"problem_id": PROBLEM})
    if isinstance(listing, list):
        print(f"\nvault {VAULT} now holds {len(listing)} rule-outs:")
        for r in listing:
            print("  -", str(r.get("rule_out_id"))[:76])
    c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
