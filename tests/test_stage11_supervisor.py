"""CPU execution doubles for persistent campaign supervision; never launch CUDA."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from lightsem.experiments.stage11 import stage11_supervisor as supervisor
from lightsem.experiments.stage11.stage11_common import (
    artifact_manifest, authenticated_json, digest, json_digest, read_authenticated, read_json,
)


class CampaignFixture:
    def __init__(self, root, training_seconds=1500.0, extension=True):
        self.root = Path(root)
        self.training_seconds = training_seconds
        self.commands = []
        self.protocol = {
            "training_seeds": [0, 1, 2, 3, 4], "extension_seeds": [5] if extension else [],
            "generation_seeds": [101, 102, 103, 104], "max_wall_hours": 24,
            "arms": {arm: {"family": "motion" if arm in supervisor.DESCRIPTOR_ARMS else "scene",
                           "steps": 2, "batch": 4, "frames": 8, "validation_steps": [2]}
                     for arm in supervisor.ALL_ARMS},
        }
        authenticated_json(self.root / "protocol.json", self.protocol)
        source = self.root / "source"
        source.mkdir()
        (source / "frozen.py").write_text("# pinned test source\n")
        files = {"frozen.py": digest(source / "frozen.py")}
        authenticated_json(self.root / "frozen_sources.json", {
            "snapshot": str(source), "files": files, "source_sha256": json_digest(files),
        })
        self.data = {family: [{"canonical_id": split + "-song", "partition": split, "key": family + "-" + split}
                              for split in ("train", "val", "test")]
                     for family in ("motion", "scene")}
        authenticated_json(self.root / "training_data.json", self.data)
        authenticated_json(self.root / "normalizers.json", {"test": "pinned"})
        self.pinned = supervisor._pin_identity(self.root)

    def train_report(self, arm, seed):
        job = supervisor.Job("train", arm, seed)
        directory = supervisor.report_path(self.root, job).parent
        directory.mkdir(parents=True, exist_ok=True)
        identity = {**self.pinned, "arm": arm, "seed": seed, "config": self.protocol["arms"][arm],
                    "initial_encoder_sha256": "identical-by-seed"}
        for name in ("best.pt", "latest.pt", "training.jsonl"):
            (directory / name).write_bytes(f"{arm}:{seed}:{name}".encode())
        (directory / "identity.json").write_text(json.dumps(identity))
        authenticated_json(directory / "validation_step_2.json", {
            "identity": identity, "step": 2, "rows": [{"canonical_id": "val-song"}], "selection_score": 0.0,
        })
        report = {
            "status": "completed", "identity": identity, "steps": 2, "precision": "fp32",
            "nonfinite": False, "oom": False, "training_seconds": self.training_seconds,
            "wall_seconds": self.training_seconds,
            "artifacts": artifact_manifest([directory / name for name in
                                            ("best.pt", "latest.pt", "identity.json", "training.jsonl")]),
        }
        authenticated_json(directory / "report.json", report)
        return report

    def evaluation_report(self, arm, seed, split):
        job = supervisor.Job("evaluate", arm, seed, split)
        directory = supervisor.report_path(self.root, job).parent
        directory.mkdir(parents=True, exist_ok=True)
        family = self.protocol["arms"][arm]["family"]
        training = supervisor.report_path(self.root, supervisor.Job("train", arm, seed))
        generation_seeds = [101, 102, 103, 104] if family == "scene" else []
        rows, captures = [], []
        for gs in generation_seeds or [None]:
            capture = directory / f"capture_{gs}.npz"
            capture.write_bytes(f"{arm}:{seed}:{split}:{gs}".encode())
            captures.append(capture)
            row = {"key": family + "-" + split, "canonical_id": split + "-song",
                   "capture_path": str(capture), "capture_sha256": digest(capture)}
            if gs is not None:
                row["generation_seed"] = gs
            rows.append(row)
        report = {
            "status": "completed",
            "identity": {**self.pinned, "arm": arm, "seed": seed, "split": split, "family": family,
                         "config": self.protocol["arms"][arm], "generation_seeds": generation_seeds,
                         "integration_steps": 8, "controls": True, "donor_controls": arm == "S_temporal",
                         "training_report_sha256": digest(training), "checkpoint_sha256": digest(training.parent / "best.pt")},
            "expected_canonical_ids": [split + "-song"], "canonical_count": 1, "row_count": len(rows),
            "rows": rows, "artifacts": artifact_manifest(captures),
        }
        authenticated_json(directory / "report.json", report)
        return report

    def aggregate_report(self, split):
        if split == "test":
            unlock = read_authenticated(self.root / "test_unlock.json")
            jobs = [supervisor.Job("evaluate", arm, seed, split) for arm, info in unlock["arms"].items()
                    for seed in info["training_seeds"]]
        else:
            jobs = [supervisor.Job("evaluate", job.arm, job.seed, split)
                    for job in supervisor.training_queue(self.protocol)]
            if read_authenticated(self.root / "extension_decision.json")["execute"]:
                jobs += [supervisor.Job("evaluate", arm, 5, split) for arm in supervisor.SCENE_ARMS]
        report = {"status": "complete", "identity": self.pinned, "gates": {"H11_D": {"passed": True}},
                  "eligible_families": ["motion"], "missing_runs": [],
                  "expected_runs": [{"arm": job.arm, "seed": job.seed} for job in jobs],
                  "arm_seeds": {arm: sorted(job.seed for job in jobs if job.arm == arm)
                                for arm in dict.fromkeys(job.arm for job in jobs)},
                  "input_reports": {str(supervisor.report_path(self.root, job)): digest(supervisor.report_path(self.root, job))
                                    for job in jobs}}
        authenticated_json(supervisor.report_path(self.root, supervisor.Job("aggregate", split=split)), report)
        return report

    def render_report(self, split):
        file = self.root / "results" / f"fake_materials_{split}.json"
        file.write_text('{"preference_status":"pending human ratings"}')
        report = {"status": "complete", "protocol_sha256": self.pinned["protocol_sha256"],
                  "artifacts": artifact_manifest([file])}
        authenticated_json(supervisor.report_path(self.root, supervisor.Job("render", split=split)), report)

    def runner(self, command, **kwargs):
        self.commands.append(command)
        self.assert_command_environment(command, kwargs)
        kwargs["on_start"](123456)
        kwargs["on_heartbeat"]()
        operation = command[5]
        value = lambda flag: command[command.index(flag) + 1]
        if operation == "train":
            self.train_report(value("--arm"), int(value("--seed")))
        elif operation == "evaluate":
            self.evaluation_report(value("--arm"), int(value("--seed")), value("--split"))
        elif operation == "aggregate":
            self.aggregate_report(value("--split"))
        elif operation == "render":
            self.render_report(value("--split"))
        return 0

    def assert_command_environment(self, command, kwargs):
        assert command[:5] == ["python", "-m", "lightsem.experiments.stage11", "--root", str(self.root)]
        assert kwargs["cwd"] == self.root / "source"
        assert kwargs["env"]["PYTHONPATH"] == str(self.root / "source")
        assert kwargs["env"]["PATH"].split(os.pathsep)[0] == str(supervisor.TORCH_ENV / "bin")

    def unlock_motion(self):
        unlock = {"schema_version": 1, "protocol_sha256": self.pinned["protocol_sha256"],
                  "validation_aggregate_sha256": digest(self.root / "results" / "aggregate_val.json"),
                  "eligible_families": ["motion"], "reviewed_by": "primary-agent", "arms": {}}
        for arm in supervisor.DESCRIPTOR_ARMS:
            unlock["arms"][arm] = {"config_sha256": json_digest(self.protocol["arms"][arm]),
                                   "training_seeds": [0, 1, 2, 3, 4],
                                   "checkpoints": {str(seed): digest(self.root / "runs" / arm / f"seed_{seed}" / "best.pt")
                                                   for seed in range(5)}}
        authenticated_json(self.root / "test_unlock.json", unlock)
        return unlock


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(supervisor, "check_gpu_idle"))
        self.stack.enter_context(patch.object(supervisor, "require_space", return_value=100.0))
        self.tmp = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.fixture = CampaignFixture(self.tmp)
        self.root = self.fixture.root

    def run_core(self):
        return supervisor.supervise(self.root, runner=self.fixture.runner)

    def test_queue_rotates_matched_arms_and_never_invents_commands(self):
        jobs = supervisor.training_queue(self.fixture.protocol)
        self.assertEqual(len(jobs), 25)
        self.assertEqual([(j.arm, j.seed) for j in jobs[:4]], [("D_old", 0), ("D_full", 0), ("D_full", 1), ("D_old", 1)])
        self.assertEqual([j.arm for j in jobs[10:16]], ["S_pointwise", "S_temporal", "S_swapped", "S_temporal", "S_swapped", "S_pointwise"])
        with self.assertRaises(ValueError):
            supervisor.command_for_job(self.root, supervisor.Job("shell", "D_old", 0))

    def test_core_stops_for_primary_review_even_when_gate_passes(self):
        result = self.run_core()
        self.assertEqual(result["status"], "awaiting_primary_review")
        self.assertEqual(len(self.fixture.commands), 51)
        self.assertTrue(result["minimum_gpu_training_duration_met"])
        self.assertEqual(result["training_seconds"], 37500)
        self.assertFalse((self.root / "test_unlock.json").exists())
        self.assertFalse((self.root / "results" / "test").exists())
        self.assertTrue((self.root / "campaign_queue_core.json.sha256.json").exists())
        self.assertEqual(read_json(self.root / "campaign_status.json")["child_pid"], None)

    def test_extension_is_one_complete_predeclared_triplet_without_time_padding(self):
        self.fixture.training_seconds = 1000
        result = self.run_core()
        self.assertTrue(result["extension_decision"]["execute"])
        trained = [command for command in self.fixture.commands if command[5] == "train"]
        self.assertEqual(len(trained), 28)
        self.assertEqual([command[command.index("--arm") + 1] for command in trained[-3:]], list(supervisor.SCENE_ARMS))
        self.assertFalse(result["minimum_gpu_training_duration_met"])
        self.assertEqual(result["training_seconds"], 28000)

    def test_completed_runs_need_explicit_resume_and_are_reauthenticated(self):
        self.run_core()
        with self.assertRaisesRegex(RuntimeError, "resume-supervisor"):
            self.run_core()
        self.fixture.commands.clear()
        result = supervisor.supervise(self.root, resume_supervisor=True, runner=self.fixture.runner)
        self.assertEqual(result["status"], "awaiting_primary_review")
        self.assertEqual(self.fixture.commands, [])
        best = self.root / "runs" / "D_old" / "seed_0" / "best.pt"
        best.write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "Artifact drift"):
            supervisor.supervise(self.root, resume_supervisor=True, runner=self.fixture.runner)
        self.assertEqual(read_json(self.root / "campaign_status.json")["status"], "failed")

    def test_nonzero_child_exit_stops_immediately_without_retry(self):
        calls = []

        def failed(command, **kwargs):
            calls.append(command)
            kwargs["on_start"](234567)
            return 17

        with self.assertRaisesRegex(RuntimeError, "automatic retry is disabled"):
            supervisor.supervise(self.root, runner=failed)
        self.assertEqual(len(calls), 1)
        status = read_json(self.root / "campaign_status.json")
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["last_returncode"], 17)
        self.assertIsNone(status["child_pid"])

    def test_checkpoint_recovery_requires_exact_explicit_authorization(self):
        run = self.root / "runs" / "D_old" / "seed_0"

        def interrupted(command, **kwargs):
            run.mkdir(parents=True)
            (run / "latest.pt").write_bytes(b"interrupted-checkpoint")
            raise InterruptedError("external interruption")

        with self.assertRaises(InterruptedError):
            supervisor.supervise(self.root, runner=interrupted)
        with self.assertRaisesRegex(RuntimeError, "missing exact"):
            supervisor.supervise(self.root, resume_supervisor=True, runner=self.fixture.runner)
        authenticated_json(self.root / "recovery_authorizations.json", {
            "protocol_sha256": self.fixture.pinned["protocol_sha256"],
            "authorizations": [{"arm": "D_old", "seed": 0, "reason": "external interruption before final",
                                "no_observed_final": True, "checkpoint_sha256": digest(run / "latest.pt")}],
        })
        result = supervisor.supervise(self.root, resume_supervisor=True, runner=self.fixture.runner)
        self.assertEqual(result["status"], "awaiting_primary_review")
        self.assertIn("--resume", self.fixture.commands[0])
        self.assertTrue(all("--resume" not in command for command in self.fixture.commands[1:]))

    def test_source_drift_during_campaign_prevents_next_child(self):
        def mutate_after_first(command, **kwargs):
            result = self.fixture.runner(command, **kwargs)
            (self.root / "source" / "frozen.py").write_text("# changed source\n")
            return result

        with self.assertRaisesRegex(RuntimeError, "Frozen source drift"):
            supervisor.supervise(self.root, runner=mutate_after_first)
        self.assertEqual(len(self.fixture.commands), 1)
        self.assertEqual(read_json(self.root / "campaign_status.json")["status"], "failed")

    def test_expired_global_deadline_is_persisted_and_never_reset(self):
        self.fixture.protocol["campaign_started_at"] = time.time() - 25 * 3600
        authenticated_json(self.root / "protocol.json", self.fixture.protocol)
        self.fixture.pinned = supervisor._pin_identity(self.root)
        with self.assertRaises(supervisor.CampaignDeadlineExceeded):
            self.run_core()
        first = read_json(self.root / "campaign_status.json")
        self.assertEqual(first["status"], "deadline_exceeded")
        with self.assertRaises(supervisor.CampaignDeadlineExceeded):
            supervisor.supervise(self.root, resume_supervisor=True, runner=self.fixture.runner)
        self.assertEqual(read_json(self.root / "campaign_status.json")["deadline"], first["deadline"])
        self.assertEqual(self.fixture.commands, [])

    def test_test_phase_requires_authentication_and_keeps_all_five_control_seeds(self):
        self.run_core()
        with self.assertRaises(FileNotFoundError):
            supervisor.supervise(self.root, phase="test", runner=self.fixture.runner)
        self.fixture.unlock_motion()
        self.fixture.commands.clear()
        result = supervisor.resume_after_review(self.root, runner=self.fixture.runner)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.fixture.commands), 12)
        self.assertEqual({command[command.index("--arm") + 1] for command in self.fixture.commands if "--arm" in command}, set(supervisor.DESCRIPTOR_ARMS))
        self.assertTrue((self.root / "results" / "render_test.json").is_file())

    def test_unlock_cannot_omit_baseline_arm_or_change_checkpoint(self):
        self.run_core()
        unlock = self.fixture.unlock_motion()
        del unlock["arms"]["D_old"]
        authenticated_json(self.root / "test_unlock.json", unlock)
        with self.assertRaisesRegex(RuntimeError, "every control arm"):
            supervisor.validate_test_unlock(self.root, self.fixture.protocol, self.fixture.pinned)
        unlock = self.fixture.unlock_motion()
        unlock["arms"]["D_full"]["checkpoints"]["0"] = "wrong"
        authenticated_json(self.root / "test_unlock.json", unlock)
        with self.assertRaisesRegex(RuntimeError, "checkpoint mismatch"):
            supervisor.validate_test_unlock(self.root, self.fixture.protocol, self.fixture.pinned)

    def test_diagnostic_render_is_an_explicit_phase_and_keeps_test_sealed(self):
        self.run_core()
        self.fixture.commands.clear()
        result = supervisor.supervise(self.root, phase="diagnostic_render", runner=self.fixture.runner)
        self.assertEqual(result["status"], "completed_diagnostics")
        self.assertEqual(len(self.fixture.commands), 1)
        self.assertEqual(self.fixture.commands[0][-2:], ["--split", "val"])
        self.assertFalse((self.root / "results" / "test").exists())

    def test_predeclared_completed_extension_cannot_be_dropped_from_test(self):
        self.fixture.training_seconds = 1000
        self.run_core()
        path = self.root / "results" / "aggregate_val.json"
        aggregate = read_authenticated(path)
        aggregate["eligible_families"] = ["motion", "scene"]
        aggregate["gates"].update({"H11_T": {"passed": True}, "H11_A": {"passed": True}})
        authenticated_json(path, aggregate)
        unlock = self.fixture.unlock_motion()
        unlock["eligible_families"] = ["motion", "scene"]
        for arm in supervisor.SCENE_ARMS:
            unlock["arms"][arm] = {"config_sha256": json_digest(self.fixture.protocol["arms"][arm]),
                                   "training_seeds": list(range(6)),
                                   "checkpoints": {str(seed): digest(self.root / "runs" / arm / f"seed_{seed}" / "best.pt")
                                                   for seed in range(6)}}
        authenticated_json(self.root / "test_unlock.json", unlock)
        _, jobs = supervisor.validate_test_unlock(self.root, self.fixture.protocol, self.fixture.pinned)
        self.assertEqual(len(jobs), 28)
        unlock["arms"]["S_temporal"]["training_seeds"] = list(range(5))
        authenticated_json(self.root / "test_unlock.json", unlock)
        with self.assertRaisesRegex(RuntimeError, "every validated predeclared seed"):
            supervisor.validate_test_unlock(self.root, self.fixture.protocol, self.fixture.pinned)

    def test_missing_generation_seed_cannot_be_hidden_by_a_smaller_row_count(self):
        self.fixture.train_report("S_temporal", 0)
        report = self.fixture.evaluation_report("S_temporal", 0, "val")
        report["rows"].pop()
        report["row_count"] = len(report["rows"])
        job = supervisor.Job("evaluate", "S_temporal", 0, "val")
        authenticated_json(supervisor.report_path(self.root, job), report)
        with self.assertRaisesRegex(RuntimeError, "recording/generation-seed matrix"):
            supervisor.verify_completion(self.root, job, self.fixture.protocol, self.fixture.pinned)

    def test_gpu_contention_stops_before_launch(self):
        with patch.object(supervisor, "check_gpu_idle", side_effect=RuntimeError("unexpected accelerator process")):
            with self.assertRaisesRegex(RuntimeError, "unexpected accelerator"):
                self.run_core()
        self.assertEqual(self.fixture.commands, [])
        self.assertEqual(read_json(self.root / "campaign_status.json")["status"], "failed")

    def test_supervisor_lock_excludes_second_owner(self):
        with supervisor._supervisor_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "another Stage11 supervisor"):
                supervisor.supervise(self.root, runner=self.fixture.runner)

    def test_successful_exit_without_authenticated_artifacts_is_not_completion(self):
        with self.assertRaises(FileNotFoundError):
            supervisor.supervise(self.root, runner=lambda *args, **kwargs: 0)
        state = read_json(self.root / "campaign_status.json")
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["completed"], {})

    def test_real_cpu_subprocess_streams_and_enforces_deadline(self):
        pids = []
        with patch("builtins.print"):
            code = supervisor.run_child(
                ["python", "-c", "print('cpu-only')"], cwd=self.root,
                env=dict(os.environ), log_path=self.root / "child.log", deadline=time.time() + 10,
                on_start=pids.append, on_heartbeat=lambda: None,
            )
        self.assertEqual(code, 0)
        self.assertIn("cpu-only", (self.root / "child.log").read_text())
        with self.assertRaises(supervisor.CampaignDeadlineExceeded):
            supervisor.run_child(
                ["python", "-c", "import time; time.sleep(5)"], cwd=self.root,
                env=dict(os.environ), log_path=self.root / "deadline.log", deadline=time.time() + 0.08,
                on_start=pids.append, on_heartbeat=lambda: None,
            )
        with self.assertRaises(ProcessLookupError):
            os.kill(pids[-1], 0)


if __name__ == "__main__":
    unittest.main()
