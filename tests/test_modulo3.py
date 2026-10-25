import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
import shutil
import gzip

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analytics_data import load_bundle, simulate_validation
from sql.carregar_e_consultar import load_tables


def fixture():
    rows = []
    for split, offset in [("validation", 0), ("test", 100)]:
        for i, seconds in enumerate([0, 1, 1, 3599, 3600, 86399, 86400, 90000, 90001, 90002, 90003, 90004]):
            fraud = int(i in [2, 8])
            probability = 0.9 if i in [2, 5, 8] else 0.1
            rows.append({"split": split, "source_row_id": offset+i, "Time": seconds,
                         "Amount": float((i+1)*300), "Class": fraud,
                         "fraud_probability": probability, "score": 1000*(1-probability),
                         "risk_decile": 1 if probability > .5 else 10, "alert": int(probability >= .5)})
    return pd.DataFrame(rows)


class SqlTests(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.data = fixture()
        report = {"selection": {"threshold": .5}, "cost_assumptions": {"review": 5, "missed_fraud": 500}}
        self.sections = load_tables(self.connection, "sqlite", self.data, report, 1000)

    def tearDown(self):
        self.connection.close()

    def query(self, name):
        return pd.read_sql_query(self.sections[name], self.connection)

    def test_q1_value_time_boundaries_and_totals(self):
        result = self.query("q1_faixa_valor_hora")
        self.assertEqual(int(result.transactions.sum()), len(self.data))
        self.assertEqual(int(result.frauds.sum()), int(self.data.Class.sum()))
        self.assertEqual(result.fraud_amount_observed.sum(), self.data.loc[self.data.Class == 1, "Amount"].sum())
        self.assertEqual(set(result.relative_hour), {0, 1, 23, 24, 25})

    def test_q2_lag_ties_partition_and_rolling_window(self):
        result = self.query("q2_janelas")
        expected = self.data.sort_values(["split", "Time", "source_row_id"])
        for split in ["test", "validation"]:
            actual = result[result.split == split]
            raw = expected[expected.split == split]
            np.testing.assert_allclose(actual.moving_average_10_transactions, raw.Amount.rolling(10, min_periods=1).mean())
            np.testing.assert_allclose(actual.seconds_since_previous_transaction, raw.Time.diff(), equal_nan=True)
            np.testing.assert_array_equal(actual.cumulative_alerts, raw.alert.cumsum())

    def test_q3_high_value_false_positives(self):
        result = self.query("q3_falsos_positivos_alto_valor")
        self.assertEqual(set(result.transaction_id), {5, 105})
        self.assertTrue((result.total_high_value_false_positives == 1).all())

    def test_q4_relative_day_is_not_rounded(self):
        result = self.query("q4_coorte_temporal")
        self.assertEqual(set(result.relative_day), {0, 1})
        self.assertTrue((result[result.relative_day == 0].transactions == 6).all())
        self.assertTrue((result[result.relative_day == 1].cumulative_frauds == 2).all())

    def test_duplicate_load_refused(self):
        with self.assertRaises(ValueError):
            load_tables(self.connection, "sqlite", self.data, {}, 1000)


class ScenarioTests(unittest.TestCase):
    def test_compressed_predictions_preserve_original_hash_and_results(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            for name in ['metrics_report.json','score_deciles.csv','shap_importance.csv','shap_summary.png']:
                shutil.copy2(ROOT/'outputs'/name, target/name)
            raw = (ROOT/'outputs/predictions.csv').read_bytes()
            (target/'predictions.csv.gz').write_bytes(gzip.compress(raw, mtime=0))
            bundle = load_bundle(target)
            self.assertEqual(bundle.report['test']['tp'], 78)
            self.assertEqual(len(bundle.predictions), 113491)
            # A valid gzip with altered CSV must fail the original report's hash.
            (target/'predictions.csv.gz').write_bytes(gzip.compress(raw.replace(b'validation', b'validatioX',1),mtime=0))
            with self.assertRaisesRegex(ValueError,'Artefato'):
                load_bundle(target)

    def test_simulation_matches_frozen_report_on_real_validation(self):
        bundle = load_bundle(ROOT/"outputs")
        report = bundle.report
        validation = bundle.predictions[bundle.predictions.split == "validation"]
        result = simulate_validation(validation, report["selection"]["threshold"],
                                     report["cost_assumptions"]["review"], report["cost_assumptions"]["missed_fraud"])
        for key in ["n", "tp", "fp", "tn", "fn", "alerts", "simulated_cost"]:
            self.assertEqual(result[key], report["validation"][key])

    def test_test_sample_is_refused_in_simulator(self):
        with self.assertRaisesRegex(ValueError, "validacao"):
            simulate_validation(fixture(), .5, 5, 500)

    def test_cost_and_threshold_extremes(self):
        data = fixture().query("split=='validation'")
        none = simulate_validation(data, np.nextafter(1., np.inf), 5, 500)
        all_alerts = simulate_validation(data, 0, 5, 500)
        self.assertEqual(none["alerts"], 0)
        self.assertEqual(none["simulated_cost"], data.Class.sum()*500)
        self.assertEqual(all_alerts["alerts"], len(data))
        self.assertEqual(all_alerts["simulated_cost"], len(data)*5)
        with self.assertRaises(ValueError):
            simulate_validation(data, .5, -1, 500)

    def test_missing_artifacts_do_not_create_fake_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                load_bundle(Path(folder))


class DashboardTests(unittest.TestCase):
    def test_controls_change_only_validation_metrics(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT/"dashboard/app_dashboard.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        final_metrics = [(m.label, m.value) for m in app.metric[:4]]
        initial_cost = next(m.value for m in app.metric if m.label == "Custo simulado total")
        app.number_input(key="review").set_value(20.0).run()
        self.assertFalse(app.exception)
        self.assertEqual(final_metrics, [(m.label, m.value) for m in app.metric[:4]])
        self.assertNotEqual(initial_cost, next(m.value for m in app.metric if m.label == "Custo simulado total"))
        app.checkbox(key="use_frozen").set_value(False).run()
        app.slider(key="threshold").set_value(1.0).run()
        self.assertFalse(app.exception)
        self.assertEqual(final_metrics, [(m.label, m.value) for m in app.metric[:4]])
        app.checkbox(key="no_alerts").set_value(True).run()
        self.assertEqual(next(m.value for m in app.metric if m.label == "Alertas na validação"), "0")
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()

