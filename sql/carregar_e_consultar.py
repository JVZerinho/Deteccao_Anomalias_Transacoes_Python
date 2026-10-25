"""Carga parametrizada e execucao do mesmo SQL em SQLite ou PostgreSQL."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analytics_data import load_bundle


def read_sections(path=None):
    text = (Path(path) if path else Path(__file__).with_name("queries_analiticas.sql")).read_text(encoding="utf-8")
    parts = re.split(r"^-- @section (\w+)\s*$", text, flags=re.MULTILINE)
    sections = {parts[i]: parts[i+1].strip() for i in range(1, len(parts), 2)}
    if list(sections) != ["schema", "q1_faixa_valor_hora", "q2_janelas", "q3_falsos_positivos_alto_valor", "q4_coorte_temporal"]:
        raise ValueError("Secoes SQL inesperadas.")
    return sections


def load_tables(connection, engine, predictions, report, high_value_cutoff):
    sections = read_sections()
    cursor = connection.cursor()
    for statement in sections["schema"].split(";"):
        if statement.strip():
            cursor.execute(statement)
    cursor.execute("SELECT COUNT(*) FROM transacoes")
    if cursor.fetchone()[0]:
        raise ValueError("Banco ja contem transacoes; use um banco/schema vazio.")
    placeholder = "?" if engine == "sqlite" else "%s"
    transaction_sql = "INSERT INTO transacoes VALUES (" + ",".join([placeholder]*5) + ")"
    rows = [(int(r.source_row_id), r.split, int(r.Time), float(r.Amount), int(r.Class))
            for r in predictions.itertuples(index=False)]
    scores = [(int(r.source_row_id), float(r.fraud_probability), float(r.score), int(r.risk_decile), int(r.alert))
              for r in predictions.itertuples(index=False)]
    # scores_modelo has five columns (transaction_id plus four score fields).
    score_sql = "INSERT INTO scores_modelo VALUES (" + ",".join([placeholder]*5) + ")"
    cursor.executemany(transaction_sql, rows)
    cursor.executemany(score_sql, scores)
    params = (1, float(report["selection"]["threshold"]), high_value_cutoff,
              float(report["cost_assumptions"]["review"]), float(report["cost_assumptions"]["missed_fraud"]))
    cursor.execute("INSERT INTO parametros_analise VALUES ("+",".join([placeholder]*5)+")", params)
    connection.commit()
    return sections


def export_queries(connection, sections, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, query in sections.items():
        if name == "schema":
            continue
        cursor = connection.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [d[0] for d in cursor.description]
        with (destination/(name+".csv")).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
        result[name] = {"rows": len(rows), "columns": columns}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["sqlite", "postgres"], default="sqlite")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs")
    parser.add_argument("--database", type=Path, default=ROOT/"outputs"/"analytics.sqlite")
    parser.add_argument("--sql-results", type=Path, default=ROOT/"outputs"/"sql_results")
    parser.add_argument("--high-value-cutoff", type=float, default=1000.0)
    args = parser.parse_args()
    if not 0 <= args.high_value_cutoff < float("inf"):
        parser.error("Limite de alto valor deve ser finito e nao negativo.")
    bundle = load_bundle(args.output_dir)
    if args.engine == "sqlite":
        args.database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(args.database)
        connection.execute("PRAGMA foreign_keys = ON")
    else:
        import psycopg
        if not os.environ.get("DATABASE_URL"):
            parser.error("Defina DATABASE_URL no ambiente; nao passe credenciais na linha de comando.")
        connection = psycopg.connect(os.environ["DATABASE_URL"])
    try:
        sections = load_tables(connection, args.engine, bundle.predictions, bundle.report, args.high_value_cutoff)
        results = export_queries(connection, sections, args.sql_results)
        manifest = {"engine": args.engine, "loaded_transactions": len(bundle.predictions),
                    "results": results, "parameters": {"high_value_cutoff": args.high_value_cutoff},
                    "note": "Tempo relativo; valores fraudulentos observados nao sao perdas realizadas."}
        (args.sql_results/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
