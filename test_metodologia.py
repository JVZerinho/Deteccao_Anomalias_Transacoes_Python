"""Invariantes metodologicas e casos extremos. python -m unittest -v test_metodologia"""
import unittest
import numpy as np
import pandas as pd
from desafio import (Costs, FEATURES, audit_and_deduplicate, best_policy,
                     make_preprocessor, metrics, population_stability_index,
                     reference_cuts, risk_decile, split_data, threshold_table, to_score)


def fixture(n=200):
    rng = np.random.default_rng(73)
    data = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    data["Time"] = np.arange(n, dtype=float)
    data["Amount"] = rng.lognormal(2, 1, size=n)
    data["Class"] = np.tile([0, 0, 0, 1], n//4)
    return data


class MethodologyTests(unittest.TestCase):
    def test_exact_duplicates_do_not_cross_splits(self):
        raw = fixture()
        raw = pd.concat([raw, raw.iloc[:12]], ignore_index=True)
        data, report = audit_and_deduplicate(raw)
        self.assertEqual(report["duplicates_removed"], 12)
        parts = split_data(data, 42)
        self.assertEqual([len(parts[k]) for k in ("train", "validation", "test")], [120, 40, 40])
        ids = [set(parts[k].index) for k in parts]
        self.assertEqual(len(set.union(*ids)), 200)
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])

    def test_invalid_and_conflicting_data_are_rejected(self):
        for column, value in [("Amount", -1), ("Time", -1), ("V1", np.inf), ("V2", np.nan), ("Class", 2)]:
            with self.subTest(column=column):
                raw = fixture()
                raw.loc[0, column] = value
                with self.assertRaises(ValueError):
                    audit_and_deduplicate(raw)
        raw = fixture()
        copied = raw.iloc[[0]].copy()
        copied["Class"] = 1 - copied["Class"]
        with self.assertRaisesRegex(ValueError, "conflitantes"):
            audit_and_deduplicate(pd.concat([raw, copied], ignore_index=True))

    def test_holdout_extremes_cannot_change_fitted_scaler(self):
        train = fixture().iloc[:120][FEATURES]
        holdout = fixture().iloc[120:][FEATURES].copy()
        holdout[["Time", "Amount"]] = 1e12
        transformer = make_preprocessor().fit(train)
        before = transformer.named_transformers_["robust"].center_.copy()
        transformer.transform(holdout)
        np.testing.assert_array_equal(before, transformer.named_transformers_["robust"].center_)
        np.testing.assert_array_equal(before, np.median(train[["Time", "Amount"]], axis=0))

    def test_threshold_cost_matches_independent_bruteforce_with_ties(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        p = np.array([0.1, 0.5, 0.5, 0.8, 1.0, 1.0])
        costs = Costs(review=7, missed_fraud=91)
        table = threshold_table(y, p, costs)
        for row in table.itertuples():
            alert = p >= row.threshold
            fn = int(np.sum((y == 1) & ~alert))
            fp = int(np.sum((y == 0) & alert))
            self.assertEqual(row.simulated_cost, 7*int(alert.sum()) + 91*fn)
            self.assertEqual(row.fp, fp)
        chosen = best_policy(table)
        self.assertEqual(chosen["simulated_cost"], table.simulated_cost.min())

    def test_no_alert_policy_is_available_even_with_probability_one(self):
        y, p = np.array([0, 1]), np.array([1.0, 1.0])
        chosen = best_policy(threshold_table(y, p, Costs(review=1000, missed_fraud=1)))
        self.assertGreater(chosen["threshold"], 1)
        self.assertEqual(chosen["alerts"], 0)
        self.assertEqual(chosen["fn"], 1)

    def test_score_direction_and_range(self):
        np.testing.assert_array_equal(to_score([0, 0.5, 1]), [1000, 500, 0])
        for p in [[-0.1], [1.1], [np.nan]]:
            with self.assertRaises(ValueError):
                to_score(p)

    def test_psi_tied_scores_and_shifted_distribution(self):
        reference = np.full(100, 500.0)
        cuts = reference_cuts(reference)
        same = population_stability_index(reference, reference, cuts)
        shifted = population_stability_index(reference, np.zeros(100), cuts)
        self.assertAlmostEqual(same["value"], 0)
        self.assertGreater(shifted["value"], 0)
        self.assertTrue(np.isfinite(shifted["value"]))
        self.assertAlmostEqual(sum(shifted["actual_proportions"]), 1)
        self.assertEqual(len(shifted["actual_counts"]), 10)
        self.assertTrue(np.all(risk_decile([0, 500, 1000], cuts) >= 1))

    def test_hand_calculated_metrics(self):
        result = metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5, Costs(5, 500))
        for key in ["roc_auc", "gini", "ks", "precision", "recall", "average_precision"]:
            self.assertEqual(result[key], 1)
        self.assertEqual(result["simulated_cost"], 10)
        self.assertEqual((result["tn"], result["fp"], result["fn"], result["tp"]), (2, 0, 0, 2))


if __name__ == "__main__":
    unittest.main()
