"""Shared data contract for SQL and dashboard; never trains or tunes a model."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_ARTIFACTS = ["score_deciles.csv", "predictions.csv", "shap_importance.csv", "shap_summary.png"]


@dataclass
class Bundle:
    report: dict
    predictions: pd.DataFrame
    deciles: pd.DataFrame
    importance: pd.DataFrame
    output_dir: Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_bundle(output_dir: Path) -> Bundle:
    output_dir = Path(output_dir).resolve()
    report = json.loads((output_dir/"metrics_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "completed" or report.get("schema_version") != "2.0":
        raise ValueError("Relatorio do Modulo 2 ausente, incompleto ou de versao incompativel.")
    for name in REQUIRED_ARTIFACTS:
        file = output_dir/name
        expected = report["artifact_sha256"].get(name)
        if not file.is_file() or not expected or digest(file) != expected:
            raise ValueError(f"Artefato ausente ou de outra execucao: {name}")
    predictions = pd.read_csv(output_dir/"predictions.csv", float_precision="round_trip")
    deciles = pd.read_csv(output_dir/"score_deciles.csv", float_precision="round_trip")
    importance = pd.read_csv(output_dir/"shap_importance.csv", float_precision="round_trip")
    required = {"split", "source_row_id", "Time", "Amount", "Class", "fraud_probability", "score", "risk_decile", "alert"}
    if set(predictions.columns) != required or set(predictions["split"]) != {"validation", "test"}:
        raise ValueError("Esquema de previsoes invalido.")
    if predictions.isna().any().any() or not predictions["source_row_id"].is_unique:
        raise ValueError("Previsoes com nulos ou IDs duplicados.")
    if not np.isfinite(predictions.drop(columns="split").to_numpy()).all():
        raise ValueError("Previsoes com valores nao finitos.")
    if not predictions["Class"].isin([0, 1]).all() or not predictions["alert"].isin([0, 1]).all():
        raise ValueError("Rotulo/alerta nao binario.")
    if not predictions["fraud_probability"].between(0, 1).all():
        raise ValueError("Probabilidade fora de [0,1].")
    if (predictions[["Time", "Amount"]] < 0).any().any():
        raise ValueError("Tempo/valor negativo.")
    if not np.equal(predictions["Time"] % 1, 0).all():
        raise ValueError("Este contrato requer Time em segundos inteiros; nao ha arredondamento implicito.")
    np.testing.assert_allclose(predictions["score"], 1000*(1-predictions["fraud_probability"]), rtol=0, atol=1e-9)
    bands = np.searchsorted(report["score"]["cutpoints"], predictions["score"], side="right")+1
    np.testing.assert_array_equal(bands, predictions["risk_decile"])
    threshold = report["selection"]["threshold"]
    for split in ["validation", "test"]:
        part = predictions[predictions["split"] == split]
        reference = report[split]
        actual = frozen_counts(part, threshold)
        if any(actual[k] != reference[k] for k in ["n", "tp", "fp", "tn", "fn"]):
            raise ValueError(f"Contagens divergentes do JSON: {split}")
        np.testing.assert_array_equal((part.fraud_probability >= threshold).astype(int), part.alert)
        dec = deciles[deciles["split"] == split].sort_values("risk_decile")
        if list(dec.risk_decile) != list(range(1, 11)):
            raise ValueError("Faixas de score incompletas ou duplicadas.")
        grouped = part.groupby("risk_decile")["Class"].agg(["size", "sum"]).reindex(range(1, 11), fill_value=0)
        np.testing.assert_array_equal(grouped["size"], dec.transactions)
        np.testing.assert_array_equal(grouped["sum"], dec.frauds)
        np.testing.assert_allclose(dec.fraud_capture, dec.frauds/part.Class.sum(), rtol=0, atol=1e-12)
    top = deciles[(deciles["split"] == "test") & (deciles["risk_decile"] <= 2)]
    if not np.isclose(top.frauds.sum()/report["test"]["fraud_count"], report["test"]["capture_top_2_deciles"]):
        raise ValueError("Captura nos decis diverge do JSON.")
    return Bundle(report, predictions, deciles, importance, output_dir)


def frozen_counts(part, threshold):
    alert = part["fraud_probability"].to_numpy() >= threshold
    fraud = part["Class"].to_numpy() == 1
    return {"n": len(part), "tp": int((alert & fraud).sum()), "fp": int((alert & ~fraud).sum()),
            "fn": int((~alert & fraud).sum()), "tn": int((~alert & ~fraud).sum())}


def simulate_validation(part, threshold, review_cost, missed_cost):
    if set(part["split"]) != {"validation"}:
        raise ValueError("Simulacao permitida somente na validacao; teste final permanece congelado.")
    if not np.isfinite([threshold, review_cost, missed_cost]).all() or threshold < 0 or threshold > np.nextafter(1.0, np.inf):
        raise ValueError("Limiar/custos invalidos.")
    if min(review_cost, missed_cost) < 0:
        raise ValueError("Custos negativos.")
    result = frozen_counts(part, threshold)
    alerts = result["tp"]+result["fp"]
    total_fraud = result["tp"]+result["fn"]
    cost = review_cost*alerts + missed_cost*result["fn"]
    if not np.isfinite(cost):
        raise ValueError("Custo simulado excede o intervalo numerico suportado.")
    return {**result, "split": "validation", "threshold": threshold,
            "alerts": alerts, "recall": result["tp"]/total_fraud if total_fraud else 0,
            "precision": result["tp"]/alerts if alerts else 0,
            "simulated_cost": cost, "cost_per_1000": cost/result["n"]*1000,
            "hypothetical_costs": True, "review_cost": review_cost, "missed_fraud_cost": missed_cost}


def scenario_curve(part, review_cost, missed_cost, frozen_threshold):
    if set(part["split"]) != {"validation"}:
        raise ValueError("Curva de cenario requer validacao.")
    thresholds = np.unique(np.r_[np.linspace(0, 1, 201), frozen_threshold, np.nextafter(1.0, np.inf)])
    return pd.DataFrame([simulate_validation(part, t, review_cost, missed_cost) for t in thresholds])
