import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.training.fit_bradley_terry import analyse_question, correctness, grouped_folds, load_inputs


def fixture():
    meta = {"knobs": ["level"], "defaults": {"level": 1.0}, "pairs": []}
    rows, groups = [], {}
    for i in range(20):
        pid = f"p{i}"
        a, b = (2.0, 0.5) if i % 2 else (0.5, 2.0)
        meta["pairs"].append({"id": pid, "music_id": f"recording{i // 2}",
                              "song_key": f"song{i // 2}",
                              "phi": {"A": {"level": a}, "B": {"level": b}},
                              "foil_overbright": None})
        answer = "a" if a > b else "b"
        rows.append({"id": pid, "rater": "r1", "fits_music": answer, "engineer": answer})
        groups[pid] = f"song:song{i // 2}"
    return meta, {p["id"]: p for p in meta["pairs"]}, rows, groups


class PreferenceTests(unittest.TestCase):
    def test_song_groups_do_not_cross_folds_and_cover_every_row(self):
        groups = np.repeat(["same", "second", "third", "fourth", "fifth"], [4, 1, 3, 2, 1])
        folds = grouped_folds(groups)
        self.assertEqual(sorted(np.concatenate(folds).tolist()), list(range(len(groups))))
        for held in folds:
            train = np.setdiff1d(np.arange(len(groups)), held)
            self.assertFalse(set(groups[train]) & set(groups[held]))

    def test_known_preference_generalises_and_sample_gate_stays_closed(self):
        meta, pairs, rows, groups = fixture()
        r = analyse_question(meta, pairs, rows, groups, "fits_music")
        self.assertEqual(r["heldout_accuracy"], 1.0)
        self.assertGreater(r["w"]["level"], 0)
        self.assertFalse(r["gate_G9.6-acc"])
        self.assertTrue(0.5 <= r["fitted_knobs"]["level"] <= 2)

    def test_foils_and_ties_cannot_affect_fit(self):
        meta, pairs, rows, groups = fixture()
        before = analyse_question(meta, pairs, rows, groups, "fits_music")
        for pid, foil, answer in [("foil", "A", "b"), ("tie", None, "=")]:
            pairs[pid] = {"phi": {"A": {"level": 1000}, "B": {"level": 0.001}},
                          "foil_overbright": foil}
            groups[pid] = "song:extra"
            rows.append({"id": pid, "rater": "r1", "fits_music": answer})
        after = analyse_question(meta, pairs, rows, groups, "fits_music")
        self.assertEqual(before["w"], after["w"])
        self.assertEqual(after["n_decided"], 20)
        self.assertEqual(after["n_foil_rows"], 1)
        self.assertEqual(after["foil_pass_rate"], 1)

    def test_indifference_has_chance_credit(self):
        np.testing.assert_array_equal(correctness(np.array([0.5, 0.5]), np.array([0, 1])), [0.5, 0.5])

    def test_repeated_rater_pair_rejected_but_second_rater_allowed(self):
        meta, _, rows, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pairs.json").write_text(json.dumps(meta))
            path = root / "ratings.jsonl"
            duplicate = copy.deepcopy(rows[0])
            path.write_text("\n".join(json.dumps(r) for r in rows + [duplicate]))
            with self.assertRaisesRegex(ValueError, "duplicate rater/pair"):
                load_inputs(path)
            duplicate["rater"] = "r2"
            path.write_text("\n".join(json.dumps(r) for r in rows + [duplicate]))
            self.assertEqual(len(load_inputs(path)[2]), 21)

    def test_invalid_answer_rejected(self):
        meta, _, rows, _ = fixture()
        rows[0]["engineer"] = "typo"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pairs.json").write_text(json.dumps(meta))
            path = root / "ratings.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, "invalid engineer answer"):
                load_inputs(path)

    def test_explicit_song_key_cannot_override_canonical_mapping(self):
        meta, _, rows, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pairs.json").write_text(json.dumps(meta))
            path = root / "ratings.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows))
            song_map = root / "songs.json"
            song_map.write_text(json.dumps({"rows": [{"yt_id": "recording0", "song_key": "different"}]}))
            with self.assertRaisesRegex(ValueError, "conflicts with canonical"):
                load_inputs(path, song_map)

    def test_one_recording_cannot_have_two_explicit_song_keys(self):
        meta, _, rows, _ = fixture()
        meta["pairs"][1]["song_key"] = "different"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pairs.json").write_text(json.dumps(meta))
            path = root / "ratings.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, "conflicting song keys"):
                load_inputs(path)

    def test_defaults_and_foil_labels_are_validated(self):
        meta, _, rows, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "ratings.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows))
            for value in (0, -1, float("nan")):
                meta["defaults"]["level"] = value
                (root / "pairs.json").write_text(json.dumps(meta))
                with self.assertRaisesRegex(ValueError, "default log-knob"):
                    load_inputs(path)
            meta["defaults"]["level"] = 1
            meta["pairs"][0]["foil_overbright"] = "a"
            (root / "pairs.json").write_text(json.dumps(meta))
            self.assertEqual(load_inputs(path)[1]["p0"]["foil_overbright"], "A")
            meta["pairs"][0]["foil_overbright"] = "invalid"
            (root / "pairs.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "foil_overbright"):
                load_inputs(path)


if __name__ == "__main__":
    unittest.main()
