"""Runtime selection tests; run with python -m unittest discover -s scripts."""

import contextlib
import io
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from depgraphpru import choose_thresholds


class RuntimeThresholdTests(unittest.TestCase):
    def select(self, rows, fraction=0.20):
        with contextlib.redirect_stdout(io.StringIO()):
            mean, std, target = choose_thresholds(rows, fraction)
        selected = [row for row in rows
                    if row["mean_score"] < mean and row["std_score"] < std]
        return selected, target

    def test_fraction_rounds_per_model(self):
        for size, expected in ((56, 11), (40, 8), (3, 1)):
            rows = [{"mean_score": 1, "std_score": i / 100} for i in range(size)]
            selected, target = self.select(rows)
            self.assertEqual(target, expected)
            self.assertEqual(selected, rows[:expected])

    def test_std_takes_priority_over_mean(self):
        rows = [{"mean_score": 100 - i, "std_score": i} for i in range(10)]
        selected, _ = self.select(rows)
        self.assertEqual(selected, rows[:2])

    def test_mean_resolves_std_boundary_ties(self):
        rows = [{"mean_score": i, "std_score": 1} for i in range(10)]
        selected, _ = self.select(rows)
        self.assertEqual(selected, rows[:2])

    def test_identical_pairs_choose_closest_attainable_count(self):
        rows = [{"mean_score": mean, "std_score": 1} for mean in (1, 2, 2, 3, 4)]
        selected, target = self.select(rows, 0.4)
        self.assertEqual(target, 2)
        self.assertEqual(selected, rows[:1])

    def test_all_zero_scores_cannot_be_split_by_thresholds(self):
        rows = [{"mean_score": 0, "std_score": 0}] * 5
        selected, _ = self.select(rows)
        self.assertEqual(len(selected), 5)

    def test_invalid_inputs(self):
        for rows, fraction in (([], .2), ([{"mean_score": 1, "std_score": 1}], 0),
                               ([{"mean_score": 1, "std_score": math.nan}], .2)):
            with self.assertRaises(ValueError):
                self.select(rows, fraction)


if __name__ == "__main__":
    unittest.main()
