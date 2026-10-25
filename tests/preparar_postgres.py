"""Exporta entradas verificadas para o teste PostgreSQL/PGlite, sem incluir a base original."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analytics_data import load_bundle

bundle = load_bundle(ROOT/"outputs")
destination = ROOT/"outputs"/"pg_inputs"
destination.mkdir(parents=True, exist_ok=True)
p = bundle.predictions
transactions = p[["source_row_id", "split", "Time", "Amount", "Class"]].copy()
transactions["Time"] = transactions["Time"].astype("int64")
transactions.to_csv(destination/"transacoes.csv", index=False)
p[["source_row_id", "fraud_probability", "score", "risk_decile", "alert"]].to_csv(destination/"scores_modelo.csv", index=False)
parameters = [1, bundle.report["selection"]["threshold"], 1000,
              bundle.report["cost_assumptions"]["review"], bundle.report["cost_assumptions"]["missed_fraud"]]
(destination/"parameters.json").write_text(json.dumps(parameters), encoding="utf-8")
print(f"Entradas verificadas exportadas: {len(p)} linhas.")
