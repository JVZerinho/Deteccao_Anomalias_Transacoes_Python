"""Reconciliacao independente com Pandas e comparacao SQLite/PostgreSQL."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analytics_data import load_bundle

bundle = load_bundle(ROOT/"outputs")
p = bundle.predictions.copy()
source = ROOT/"outputs/sql_results"
pg = ROOT/"outputs/postgres_results"
queries = ["q1_faixa_valor_hora", "q2_janelas", "q3_falsos_positivos_alto_valor", "q4_coorte_temporal"]
comparisons = {}
for name in queries:
    left = pd.read_csv(source/(name+".csv"), float_precision="round_trip")
    right = pd.read_csv(pg/(name+".csv"), float_precision="round_trip")
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False, atol=1e-8, rtol=1e-12)
    comparisons[name] = {"rows": len(left), "sqlite_postgres_values_match": True}

q1 = pd.read_csv(source/(queries[0]+".csv"))
p["relative_hour"] = (p.Time//3600).astype(int)
p["relative_day"] = (p.Time//86400).astype(int)
p["amount_band"] = np.select([p.Amount<10,p.Amount<100,p.Amount<1000],
                              ['01: abaixo de 10','02: de 10 a menos de 100','03: de 100 a menos de 1000'],
                              default='04: 1000 ou mais')
expected = p.groupby(["split", "relative_hour", "amount_band"]).agg(transactions=("Class","size"), frauds=("Class","sum")).reset_index()
pd.testing.assert_frame_equal(q1[expected.columns], expected, check_dtype=False)
assert np.isclose(q1.fraud_amount_observed.sum(), p.loc[p.Class==1,"Amount"].sum())
assert np.isclose(q1.simulated_missed_fraud_loss.sum(), ((p.Class==1)&(p.alert==0)).sum()*bundle.report['cost_assumptions']['missed_fraud'])
assert np.isclose(q1.simulated_review_cost.sum(), p.alert.sum()*bundle.report['cost_assumptions']['review'])

q2 = pd.read_csv(source/(queries[1]+".csv"))
sorted_p = p.sort_values(["split","Time","source_row_id"])
for split in ['test','validation']:
    actual = q2[q2.split==split]
    raw = sorted_p[sorted_p.split==split]
    np.testing.assert_array_equal(actual.transaction_id, raw.source_row_id)
    np.testing.assert_allclose(actual.moving_average_10_transactions, raw.Amount.rolling(10,min_periods=1).mean(), rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(actual.seconds_since_previous_transaction, raw.Time.diff(),equal_nan=True)
    np.testing.assert_array_equal(actual.cumulative_alerts, raw.alert.cumsum())

q3 = pd.read_csv(source/(queries[2]+".csv"))
expected_fp = p[(p.Class==0)&(p.alert==1)&(p.Amount>=1000)]
assert set(q3.transaction_id)==set(expected_fp.source_row_id)

q4 = pd.read_csv(source/(queries[3]+".csv"))
expected = p.groupby(['split','relative_day']).agg(transactions=('Class','size'),frauds=('Class','sum')).reset_index()
pd.testing.assert_frame_equal(q4[expected.columns], expected, check_dtype=False)
engine = json.loads((pg/'engine.json').read_text(encoding='utf-8'))
summary = {'status':'passed','loaded_transactions':len(p), 'queries':comparisons,
           'independent_pandas_reconciliation':True,'methodology_and_ui_tests':10,
           'postgres_verification':engine,
           'postgres_limit':'PostgreSQL WASM via PGlite; native PostgreSQL server/psycopg connection not exercised.',
           'high_value_false_positive_cutoff':1000,'high_value_false_positive_count':len(expected_fp),
           'test_model_retraining':False}
(ROOT/'VALIDACAO_MODULO_3.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
