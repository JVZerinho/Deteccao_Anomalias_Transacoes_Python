"""Estudo educacional de fraude v2; Python 3.12, sem score proprietario.

Treino 60%, validacao 20%, teste 20%, apos deduplicacao exata.
Selecao de modelo/limiar exclusivamente na validacao; teste final uma vez.
Uso: python desafio.py --data data/creditcard.csv --output-dir outputs
Ou:  python desafio.py --openml-id 1597 --output-dir outputs
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import platform
from pathlib import Path
import shutil
import sys
import time
from urllib.request import urlopen

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
import shap
from xgboost import XGBClassifier

FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
REQUIRED = FEATURES + ["Class"]
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib",
            "xgboost", "shap", "joblib", "numba", "llvmlite"]
LIMITATIONS = [
    "Score educacional 1000*(1-p_fraude); nao e score Serasa nem score de credito.",
    "Probabilidades de modelos com pesos de classe nao foram calibradas.",
    "Divisao aleatoria estratificada; nao e validacao fora do tempo.",
    "Sem identificador de cliente/cartao: isolamento por entidade nao verificavel.",
    "Features V1..V28 sao PCA preexistente: nao ha acesso ao ajuste original.",
    "Custos sao hipoteses; nao representam perdas ou economia reais.",
    "PSI entre amostras aleatorias e demonstracao, nao monitoramento em producao.",
    "Selecao repetida na mesma validacao pode se ajustar a ela; teste permanece reservado.",
]


@dataclass(frozen=True)
class Costs:
    review: float = 5.0
    missed_fraud: float = 500.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_value(obj):
    if isinstance(obj, dict):
        return {str(k): json_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [json_value(v) for v in obj]
    if isinstance(obj, np.generic):
        return json_value(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path: Path, obj) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_value(obj), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_data(args) -> tuple[pd.DataFrame, Path, dict]:
    if args.openml_id:
        if args.openml_id != 1597:
            raise ValueError("Este estudo fixa o corpus OpenML 1597; use --data para outro CSV compativel.")
        cache = Path(args.data_home)
        cache.mkdir(parents=True, exist_ok=True)
        metadata_path = cache / "metadata-1597.json"
        if not metadata_path.exists():
            with urlopen("https://www.openml.org/api/v1/json/data/1597", timeout=120) as response:
                metadata_path.write_bytes(response.read())
        details = json.loads(metadata_path.read_text(encoding="utf-8"))["data_set_description"]
        arff = cache / "creditcard-1597.arff"
        if not arff.exists():
            temporary = arff.with_suffix(".download")
            with urlopen(details["url"], timeout=120) as response, temporary.open("wb") as target:
                shutil.copyfileobj(response, target)
            temporary.replace(arff)
        with arff.open("rb") as stream:
            digest = hashlib.file_digest(stream, "md5").hexdigest()
        if digest != details["md5_checksum"]:
            raise ValueError("Arquivo OpenML difere do checksum publicado; preserve o cache para investigar.")
        # fetch_openml removes Time because this dataset marks it as row_id_attribute.
        # Read the raw numeric ARFF to retain the REAL Time column.
        header_columns = []
        with arff.open(encoding="utf-8") as stream:
            for line in stream:
                if line.lower().startswith("@attribute "):
                    header_columns.append(line.split()[1].strip("'\""))
                if line.lower().startswith("@data"):
                    break
        if header_columns != REQUIRED:
            raise ValueError(f"Esquema ARFF inesperado: {header_columns}")
        frame = pd.read_csv(arff, comment="@", header=None, names=REQUIRED, quotechar="'")
        path = cache.parent / "creditcard-openml-1597.csv"
        frame.to_csv(path, index=False)
        provenance = {"kind": "public_benchmark", "openml_id": 1597,
                      "url": "https://www.openml.org/search?type=data&id=1597",
                      "name": details.get("name"), "version": details.get("version"),
                      "license": details.get("licence"), "raw_arff_sha256": sha256(arff),
                      "published_md5_verified": True, "time_column_preserved": True,
                      "mirror_note": "Espelho OpenML; nao usa o antigo link Dropbox."}
    else:
        path = Path(args.data)
        frame = pd.read_csv(path)
        provenance = {"kind": "local_csv", "path_name": path.name,
                      "source_note": args.source_note or "Origem externa nao declarada pelo operador."}
    provenance["input_file_sha256"] = sha256(path)
    return frame, path, provenance


def audit_and_deduplicate(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Checks deterministic; no scaler, imputer or feature selection is fitted here."""
    if set(raw.columns) != set(REQUIRED) or raw.columns.duplicated().any():
        raise ValueError(f"CSV deve conter exatamente {REQUIRED}; obtido {list(raw.columns)}")
    frame = raw.loc[:, REQUIRED].copy()
    original_types = frame.dtypes.astype(str).to_dict()
    missing = frame.isna().sum().to_dict()
    converted = frame.apply(pd.to_numeric, errors="coerce")
    invalid = (converted.isna() & ~frame.isna()).sum().to_dict()
    if sum(missing.values()) or sum(invalid.values()):
        raise ValueError(f"Dados ausentes/nao numericos; faltantes={missing}; invalidos={invalid}")
    if not np.isfinite(converted.to_numpy(dtype=float)).all():
        raise ValueError("Base contem valores infinitos.")
    if set(converted["Class"].unique()) != {0, 1}:
        raise ValueError("Class precisa conter as duas classes binarias 0 e 1.")
    if (converted[["Time", "Amount"]] < 0).any().any():
        raise ValueError("Time e Amount devem ser nao negativos.")
    converted["Class"] = converted["Class"].astype(int)
    feature_hash = pd.util.hash_pandas_object(converted[FEATURES], index=False)
    conflicts = converted.groupby(feature_hash)["Class"].nunique().gt(1).sum()
    if conflicts:
        raise ValueError(f"{conflicts} grupos com features identicas e rotulos conflitantes.")
    duplicates = converted.duplicated(keep="first")
    frame = converted.loc[~duplicates].copy()
    frame.index.name = "source_row_id"
    counts = frame["Class"].value_counts()
    if counts.min() < 10:
        raise ValueError("Cada classe precisa de pelo menos 10 observacoes apos deduplicacao.")
    quality = {
        "rows_before": len(raw), "rows_after": len(frame),
        "duplicates_removed": int(duplicates.sum()),
        "duplicate_policy": "Primeira ocorrencia mantida antes da divisao; rotulos conflitantes rejeitados.",
        "original_dtypes": original_types, "numeric_dtypes": frame.dtypes.astype(str).to_dict(),
        "missing_values": missing, "invalid_numeric_values": invalid,
        "fraud_count_before": int(converted["Class"].sum()),
        "fraud_count_after": int(frame["Class"].sum()),
        "fraud_rate_after": float(frame["Class"].mean()),
        "temporal_checks": {
            "time_min_seconds": float(frame["Time"].min()),
            "time_max_seconds": float(frame["Time"].max()),
            "negative_time_count": 0,
            "backward_steps_in_input_order": int(frame["Time"].diff().lt(0).sum()),
            "equal_time_steps": int(frame["Time"].diff().eq(0).sum()),
            "time_is_integer_seconds": bool(np.equal(frame["Time"] % 1, 0).all()),
            "note": "Time relativo ao inicio da base; sem data civil ou identificador de entidade.",
        },
    }
    return frame, quality


def split_data(frame: pd.DataFrame, seed: int):
    remaining, test_ids = train_test_split(frame.index.to_numpy(), test_size=0.20,
                                         stratify=frame["Class"], random_state=seed)
    train_ids, validation_ids = train_test_split(
        remaining, test_size=0.25, stratify=frame.loc[remaining, "Class"], random_state=seed)
    result = {"train": frame.loc[train_ids].copy(),
              "validation": frame.loc[validation_ids].copy(), "test": frame.loc[test_ids].copy()}
    for name, part in result.items():
        if part["Class"].nunique() != 2:
            raise ValueError(f"Conjunto {name} sem as duas classes.")
    hashes = {name: set(pd.util.hash_pandas_object(part[FEATURES], index=False))
              for name, part in result.items()}
    assert not (hashes["train"] & hashes["validation"] or hashes["train"] & hashes["test"]
                or hashes["validation"] & hashes["test"]), "Sobreposicao de features entre conjuntos."
    return result


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("robust", RobustScaler(), ["Time", "Amount"]),
        ("log_amount", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), ["Amount"]),
        ("components", "passthrough", [f"V{i}" for i in range(1, 29)]),
    ], remainder="drop", verbose_feature_names_out=True)


def make_models(y_train: pd.Series, seed: int, jobs: int) -> dict[str, Pipeline]:
    weight = float((y_train == 0).sum() / (y_train == 1).sum())
    estimators = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=3000,
                                                   solver="lbfgs", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10,
                                                min_samples_leaf=2, class_weight="balanced",
                                                random_state=seed, n_jobs=jobs),
        "xgboost": XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                                 scale_pos_weight=weight, eval_metric="logloss",
                                 tree_method="hist", random_state=seed, n_jobs=jobs),
    }
    return {name: Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
            for name, estimator in estimators.items()}


def threshold_table(y, probability, costs: Costs) -> pd.DataFrame:
    """Exact candidates from validation PR curve, including the no-alert policy."""
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    _, _, thresholds = precision_recall_curve(y, probability)
    thresholds = np.append(thresholds, np.nextafter(1.0, np.inf))
    order = np.argsort(probability, kind="stable")
    sorted_p, sorted_y = probability[order], y[order]
    cumulative = np.r_[0, np.cumsum(sorted_y)]
    below = np.searchsorted(sorted_p, thresholds, side="left")
    fn = cumulative[below]
    tp = int(y.sum()) - fn
    alerts = len(y) - below
    fp = alerts - tp
    tn = len(y) - int(y.sum()) - fp
    precision = np.divide(tp, alerts, out=np.zeros_like(tp, dtype=float), where=alerts != 0)
    recall = tp / int(y.sum())
    cost = costs.review * alerts + costs.missed_fraud * fn
    return pd.DataFrame({"threshold": thresholds, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                         "alerts": alerts, "precision": precision, "recall": recall,
                         "f1": np.divide(2 * precision * recall, precision + recall,
                                         out=np.zeros_like(recall), where=precision+recall != 0),
                         "simulated_cost": cost, "cost_per_1000": cost / len(y) * 1000})


def best_policy(table: pd.DataFrame) -> dict:
    # Deterministic ties: fewer manual reviews, then higher threshold.
    row = table.sort_values(["simulated_cost", "alerts", "threshold"],
                            ascending=[True, True, False], kind="stable").iloc[0]
    return json_value(row.to_dict())


def metrics(y, probability, threshold: float, costs: Costs) -> dict:
    y = np.asarray(y, dtype=int)
    predicted = (np.asarray(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y, probability, drop_intermediate=False)
    precision, recall, _ = precision_recall_curve(y, probability)
    roc_auc = roc_auc_score(y, probability)
    cost = costs.review * (tp + fp) + costs.missed_fraud * fn
    return json_value({
        "n": len(y), "fraud_count": int(y.sum()), "fraud_rate": float(y.mean()),
        "threshold": threshold, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "alerts": tp + fp, "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted), "f1": f1_score(y, predicted, zero_division=0),
        "average_precision": average_precision_score(y, probability),
        "pr_auc_trapezoid": auc(recall[::-1], precision[::-1]),
        "roc_auc": roc_auc, "gini": 2 * roc_auc - 1,
        "ks": np.max(np.abs(tpr - fpr)),
        "simulated_cost": cost, "cost_per_1000": cost / len(y) * 1000,
    })


def to_score(probability) -> np.ndarray:
    """Higher score = lower model risk. No calibration or proprietary equivalence."""
    p = np.asarray(probability, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilidades invalidas.")
    return np.clip(1000.0 * (1.0 - p), 0, 1000)


def reference_cuts(training_scores) -> np.ndarray:
    # Keep repeated cuts: always 10 bands, some may be empty if predictions tie.
    return np.quantile(training_scores, np.arange(1, 10) / 10, method="linear")


def risk_decile(scores, cuts) -> np.ndarray:
    """1 = lowest scores / highest risk. Boundaries fitted on TRAIN only."""
    return np.searchsorted(cuts, np.asarray(scores), side="right") + 1


def population_stability_index(reference_scores, actual_scores, cuts, smoothing=0.5) -> dict:
    if smoothing <= 0 or not len(reference_scores) or not len(actual_scores):
        raise ValueError("PSI requer amostras nao vazias e suavizacao positiva.")
    expected_counts = np.bincount(risk_decile(reference_scores, cuts), minlength=11)[1:11]
    actual_counts = np.bincount(risk_decile(actual_scores, cuts), minlength=11)[1:11]
    expected = (expected_counts + smoothing) / (expected_counts.sum() + 10*smoothing)
    actual = (actual_counts + smoothing) / (actual_counts.sum() + 10*smoothing)
    contributions = (actual - expected) * np.log(actual / expected)
    return json_value({"value": contributions.sum(), "reference": "validation", "actual": "test",
                       "cutoff_source": "train_score_quantiles", "smoothing_pseudocount": smoothing,
                       "reference_counts": expected_counts, "actual_counts": actual_counts,
                       "reference_proportions": expected, "actual_proportions": actual,
                       "band_contributions": contributions,
                       "interpretation": "Comparacao descritiva entre amostras aleatorias; sem alerta de drift operacional."})


def decile_table(part, probability, cuts, split: str) -> pd.DataFrame:
    score = to_score(probability)
    data = pd.DataFrame({"decile": risk_decile(score, cuts),
                         "fraud": part["Class"].to_numpy(), "score": score,
                         "amount": part["Amount"].to_numpy()})
    rows, cumulative_n, cumulative_fraud = [], 0, 0
    total_fraud = int(data["fraud"].sum())
    bounds = np.r_[0.0, cuts, 1000.0]
    for decile in range(1, 11):
        group = data[data["decile"] == decile]
        count, frauds = len(group), int(group["fraud"].sum())
        cumulative_n += count
        cumulative_fraud += frauds
        rows.append({"split": split, "risk_decile": decile,
                     "score_lower": bounds[decile-1], "score_upper": bounds[decile],
                     "transactions": count, "population_share": count/len(data), "frauds": frauds,
                     "fraud_rate": frauds/count if count else np.nan,
                     "fraud_capture": frauds/total_fraud,
                     "cumulative_transactions": cumulative_n,
                     "cumulative_population_share": cumulative_n/len(data),
                     "cumulative_capture": cumulative_fraud/total_fraud,
                     "fraud_amount_observed": float(group.loc[group.fraud == 1, "amount"].sum())})
    return pd.DataFrame(rows)


def make_predictions(part, probability, threshold, cuts, split):
    out = part[["Time", "Amount", "Class"]].copy().reset_index()
    out.insert(0, "split", split)
    out["fraud_probability"] = probability
    out["score"] = to_score(probability)
    out["risk_decile"] = risk_decile(out["score"], cuts)
    out["alert"] = (probability >= threshold).astype(int)
    return out


def explain_validation(pipeline, train, validation, output: Path, seed: int, size: int) -> dict:
    rng = np.random.default_rng(seed)
    train_positions = rng.choice(len(train), size=min(100, len(train)), replace=False)
    positions = rng.choice(len(validation), size=min(size, len(validation)), replace=False)
    preprocessor, model = pipeline.named_steps["preprocess"], pipeline.named_steps["model"]
    background = preprocessor.transform(train.iloc[train_positions][FEATURES])
    sample = preprocessor.transform(validation.iloc[positions][FEATURES])
    names = preprocessor.get_feature_names_out().tolist()
    if isinstance(model, LogisticRegression):
        explanation = shap.LinearExplainer(model, background)(sample)
        units = "log_odds"
    else:
        explanation = shap.TreeExplainer(model)(sample)
        units = "log_odds" if isinstance(model, XGBClassifier) else "fraud_probability"
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if values.shape != sample.shape or not np.isfinite(values).all():
        raise ValueError("Formato/valores SHAP invalidos; relatorio final nao sera emitido.")
    shap.summary_plot(values, sample, feature_names=names, show=False,
                      max_display=12, plot_size=(10, 7), rng=np.random.default_rng(seed))
    plt.title(f"SHAP na validacao — {units}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output/"shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close("all")
    importance = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)})
    importance.sort_values("mean_abs_shap", ascending=False).to_csv(output/"shap_importance.csv", index=False)
    return {"split": "validation", "n_explained": len(sample), "background_split": "train",
            "units": units, "sample_row_ids": validation.iloc[positions].index.tolist(),
            "note": "Associacoes do modelo, nao explicacao causal; V1..V28 sao componentes PCA anonimos."}


def save_plots(validation, test, validation_p, test_p, threshold, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, part, prob, color, linestyle in [
        ("Validacao", validation, validation_p, "#a3452d", "-"),
        ("Teste final", test, test_p, "#344c47", "--"),
    ]:
        precision, recall, _ = precision_recall_curve(part["Class"], prob)
        ax.plot(recall, precision, label=name, color=color, linestyle=linestyle)
    validation_alert = (validation_p >= threshold).astype(int)
    ax.scatter([recall_score(validation["Class"], validation_alert)],
               [precision_score(validation["Class"], validation_alert, zero_division=0)], color="#a3452d", marker="o",
               label="Limiar escolhido na validacao", zorder=3)
    ax.set(xlabel="Recall", ylabel="Precisao", xlim=(0, 1), ylim=(0, 1.02),
           title="Curva Precision-Recall — modelo selecionado")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/"pr_curve.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1000, 31)
    for name, prob, color, linestyle in [("Validacao", validation_p, "#a3452d", "-"),
                                         ("Teste final", test_p, "#344c47", "--")]:
        ax.hist(to_score(prob), bins=bins, weights=np.ones(len(prob))/len(prob),
                histtype="step", linewidth=1.8, label=name, color=color, linestyle=linestyle)
    ax.set(xlabel="Score educacional (alto = menor risco previsto)", ylabel="Proporcao da amostra",
           title="Distribuicao do score — sem calibracao de credito")
    ax.legend(); fig.tight_layout(); fig.savefig(output/"score_distribution.png", dpi=160); plt.close(fig)


def run(args) -> dict:
    started = time.monotonic()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Diretorio de saida nao vazio. Use outro --output-dir para preservar o teste anterior.")
    output.mkdir(parents=True, exist_ok=True)
    costs = Costs(args.review_cost, args.missed_fraud_cost)
    raw, data_path, provenance = load_data(args)
    frame, quality = audit_and_deduplicate(raw)
    write_json(output/"data_quality_report.json", quality)
    parts = split_data(frame, args.seed)
    train, validation, test = (parts[key] for key in ("train", "validation", "test"))
    print(f"Dados: {len(raw)} -> {len(frame)} linhas; duplicatas removidas: {quality['duplicates_removed']}", flush=True)
    assignment = pd.concat([pd.DataFrame({"source_row_id": part.index, "split": name})
                            for name, part in parts.items()], ignore_index=True)
    assignment.sort_values("source_row_id").to_csv(output/"split_assignments.csv", index=False)
    pipelines = make_models(train["Class"], args.seed, args.jobs)
    candidates, probabilities, policy_tables = [], {}, []
    expected_center = np.median(train[["Time", "Amount"]].to_numpy(), axis=0)
    for name, pipeline in pipelines.items():
        print(f"Treinando {name} exclusivamente no treino...", flush=True)
        pipeline.fit(train[FEATURES], train["Class"])
        actual_center = pipeline.named_steps["preprocess"].named_transformers_["robust"].center_
        np.testing.assert_allclose(actual_center, expected_center, rtol=0, atol=0)
        p = pipeline.predict_proba(validation[FEATURES])[:, 1]
        probabilities[name] = p
        table = threshold_table(validation["Class"], p, costs)
        table.insert(0, "model", name)
        policy_tables.append(table)
        policy = best_policy(table)
        result = metrics(validation["Class"], p, policy["threshold"], costs)
        candidates.append({"model": name, **result})
    # No test probabilities or test labels enter this decision.
    candidates.sort(key=lambda row: (row["cost_per_1000"], row["alerts"],
                                      -row["average_precision"], row["model"]))
    selected = candidates[0]
    name, threshold = selected["model"], selected["threshold"]
    pipeline, validation_p = pipelines[name], probabilities[name]
    pd.concat(policy_tables, ignore_index=True).to_csv(output/"validation_thresholds.csv", index=False)
    pd.DataFrame(candidates).to_csv(output/"validation_model_comparison.csv", index=False)
    print(f"Selecao congelada na validacao: {name}, limiar={threshold:.8f}", flush=True)
    training_p = pipeline.predict_proba(train[FEATURES])[:, 1]
    cuts = reference_cuts(to_score(training_p))
    shap_info = explain_validation(pipeline, train, validation, output, args.seed, args.shap_sample)
    # The ONLY test prediction call. No refit or retuning follows this block.
    test_p = pipeline.predict_proba(test[FEATURES])[:, 1]
    test_metrics = metrics(test["Class"], test_p, threshold, costs)
    deciles = pd.concat([decile_table(validation, validation_p, cuts, "validation"),
                         decile_table(test, test_p, cuts, "test")], ignore_index=True)
    deciles.to_csv(output/"score_deciles.csv", index=False)
    test_deciles = deciles[deciles.split == "test"]
    test_metrics["capture_top_2_deciles"] = float(test_deciles.loc[test_deciles.risk_decile <= 2, "frauds"].sum()/test_metrics["fraud_count"])
    test_metrics["population_top_2_deciles"] = float(test_deciles.loc[test_deciles.risk_decile <= 2, "transactions"].sum()/len(test))
    stability = population_stability_index(to_score(validation_p), to_score(test_p), cuts)
    pd.concat([make_predictions(validation, validation_p, threshold, cuts, "validation"),
               make_predictions(test, test_p, threshold, cuts, "test")], ignore_index=True).to_csv(output/"predictions.csv", index=False)
    save_plots(validation, test, validation_p, test_p, threshold, output)
    joblib.dump(pipeline, output/"selected_pipeline.joblib", compress=3)
    train_bounds = pipeline.named_steps["preprocess"].named_transformers_["robust"]
    report = {
        "schema_version": "2.0", "status": "completed", "dataset_kind": provenance["kind"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.monotonic()-started, "seed": args.seed,
        "provenance": provenance, "data_quality": quality,
        "split_policy": {"kind": "stratified_random", "requested_fractions": [0.6, 0.2, 0.2],
                         "actual": {key: {"n": len(part), "fraction": len(part)/len(frame),
                                           "fraud_count": int(part.Class.sum()),
                                           "fraud_rate": float(part.Class.mean())} for key, part in parts.items()},
                         "feature_overlap_between_splits": 0},
        "preprocessing_audit": {"fit_split": "train", "columns": ["Time", "Amount"],
                                "robust_center": train_bounds.center_, "robust_scale": train_bounds.scale_,
                                "train_median_assertion_passed": True},
        "selection": {"source": "validation_only", "model": name, "threshold": threshold,
                      "objective": "Minimizar custo simulado total; desempate por menos alertas, maior AP, nome.",
                      "threshold_tie_break": "Menos alertas; depois maior limiar.",
                      "models_compared": candidates, "test_prediction_calls": 1,
                      "test_used_for_selection": False, "refit_after_selection": False},
        "cost_assumptions": {**asdict(costs), "units": "unidades monetarias hipoteticas",
                             "formula": "review*(TP+FP) + missed_fraud*FN",
                             "note": "Revisao de todo alerta; perda fixa por fraude nao detectada; beneficio de TP nao comprovado."},
        "validation": selected, "test": test_metrics,
        "score": {"formula": "1000*(1-p_fraud)", "higher_is": "lower_predicted_risk",
                  "calibrated": False, "risk_decile_1": "highest_risk",
                  "reference": "train", "cutpoints": cuts,
                  "ties": "Scores iguais nao sao separados; faixas podem ficar vazias ou nao ter 10% da amostra.",
                  "boundary_policy": "Limite inferior inclusivo; superior exclusivo, salvo ultima faixa inclui 1000."},
        "psi": stability, "shap": shap_info, "limitations": LIMITATIONS,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "packages": {package: metadata.version(package) for package in PACKAGES}},
        "model_configurations": {key: value.named_steps["model"].get_params()
                                 for key, value in pipelines.items()},
        "code_sha256": sha256(Path(__file__)),
        "requirements_sha256": sha256(Path(__file__).with_name("requirements.txt")),
    }
    report["artifact_sha256"] = {file.name: sha256(file) for file in sorted(output.iterdir()) if file.is_file()}
    # Last successful artifact: no metrics placeholder is emitted as a measured result.
    write_json(output/"metrics_report.json", report)
    print(json.dumps({"status": "completed", "model": name, "test": test_metrics,
                      "psi": stability["value"], "output_dir": str(output)}, ensure_ascii=False), flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="CSV com Time, V1..V28, Amount, Class.")
    source.add_argument("--openml-id", type=int, help="1597: espelho publico do benchmark creditcard.")
    parser.add_argument("--data-home", default="data/openml-cache")
    parser.add_argument("--source-note", default="")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--shap-sample", type=int, default=300)
    parser.add_argument("--review-cost", type=float, default=5.0)
    parser.add_argument("--missed-fraud-cost", type=float, default=500.0)
    args = parser.parse_args()
    if args.jobs < 1 or args.shap_sample < 1:
        parser.error("--jobs e --shap-sample devem ser positivos.")
    if not np.isfinite([args.review_cost, args.missed_fraud_cost]).all() or min(args.review_cost, args.missed_fraud_cost) < 0:
        parser.error("Custos devem ser finitos e nao negativos.")
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as exc:
        print(f"Execucao interrompida: {type(exc).__name__}: {exc}. Nenhuma metrica deve ser inferida.", file=sys.stderr)
        raise
