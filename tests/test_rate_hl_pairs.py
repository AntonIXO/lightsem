"""CPU-only rating CLI tests; no media player or real ratings are touched."""

import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import rate_hl_pairs


class RateHLPairsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pairs_path = self.root / "pairs.json"
        self.pairs_path.write_text(json.dumps({"pairs": [{"id": "pair_1"}]}))
        self.ratings_path = self.root / "ratings.jsonl"

    def run_session(self, answers, rater="anton"):
        with patch("builtins.input", side_effect=answers) as prompt, \
                patch("rate_hl_pairs.subprocess.run") as player, \
                contextlib.redirect_stdout(io.StringIO()):
            completed = rate_hl_pairs.rate_pairs(self.pairs_path, rater)
        return completed, prompt, player

    def read_ratings(self):
        if not self.ratings_path.exists():
            return []
        return [json.loads(line) for line in self.ratings_path.read_text().splitlines()]

    def test_second_rater_can_rate_existing_pair_without_changing_original(self):
        original = json.dumps({"id": "pair_1", "rater": "anton", "fits_music": "a",
                               "engineer": "b", "t": 123}) + "\n"
        self.ratings_path.write_text(original)

        completed, _, player = self.run_session(["B", "="], rater="second")

        self.assertTrue(completed)
        self.assertEqual(player.call_count, 2)
        self.assertTrue(self.ratings_path.read_text().startswith(original))
        ratings = self.read_ratings()
        self.assertEqual(len(ratings), 2)
        self.assertEqual(ratings[1]["rater"], "second")
        self.assertEqual(ratings[1]["fits_music"], "b")
        self.assertEqual(ratings[1]["engineer"], "=")

    def test_same_rater_resumes_without_duplicate_rating(self):
        self.run_session(["a", "b"])
        original = self.ratings_path.read_bytes()

        completed, prompt, player = self.run_session([])

        self.assertTrue(completed)
        prompt.assert_not_called()
        player.assert_not_called()
        self.assertEqual(self.ratings_path.read_bytes(), original)

    def test_invalid_answers_are_reprompted_at_both_questions(self):
        completed, prompt, player = self.run_session(["", "c", " A ", "anything", " B "])

        self.assertTrue(completed)
        self.assertEqual(prompt.call_count, 5)
        self.assertEqual(player.call_count, 2)
        rating, = self.read_ratings()
        self.assertEqual((rating["fits_music"], rating["engineer"]), ("a", "b"))

    def test_quit_at_either_question_does_not_write_partial_rating(self):
        for answers in (["q"], ["a", "q"], ["a", EOFError()], ["a", KeyboardInterrupt()]):
            with self.subTest(answers=answers):
                completed, _, _ = self.run_session(answers)
                self.assertFalse(completed)
                self.assertFalse(self.ratings_path.exists())

    def test_replay_at_either_question_restarts_both_answers(self):
        for answers in (["r", "b", "="], ["a", "r", "b", "="]):
            with self.subTest(answers=answers):
                completed, _, player = self.run_session(answers)
                self.assertTrue(completed)
                self.assertEqual(player.call_count, 4)
                rating, = self.read_ratings()
                self.assertEqual((rating["fits_music"], rating["engineer"]), ("b", "="))
                self.ratings_path.unlink()

    def test_import_does_not_parse_args_or_start_rating_session(self):
        with patch("argparse.ArgumentParser.parse_args") as parse_args, \
                patch("subprocess.run") as player, patch("builtins.input") as prompt:
            importlib.reload(rate_hl_pairs)
        parse_args.assert_not_called()
        player.assert_not_called()
        prompt.assert_not_called()

    def test_existing_cli_arguments_select_pairs_and_rater(self):
        with patch("rate_hl_pairs.rate_pairs") as rate:
            status = rate_hl_pairs.main(["--rater", "second", "--pairs", str(self.pairs_path)])
        self.assertEqual(status, 0)
        rate.assert_called_once_with(self.pairs_path, "second")


if __name__ == "__main__":
    unittest.main()
