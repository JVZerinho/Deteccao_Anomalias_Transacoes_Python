"""Dashboard de resultados congelados + simulacao exclusivamente na validacao."""
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analytics_data import artifact_paths, load_bundle, scenario_curve, simulate_validation

st.set_page_config(page_title="Fraude, sob análise — João Victor", page_icon=":material/query_stats:", layout="wide")
st.caption("JOÃO VICTOR PEREIRA SANTOS / CADERNO DE ANÁLISE")
st.title("Fraude, sob análise.")
st.markdown("Um estudo de decisões: **o que o modelo encontra, quantos alertas gera e o que cada hipótese de custo muda.**")


@st.cache_data(ttl=3600, max_entries=4, show_spinner="Conferindo os artefatos da execução…")
def read_artifacts(directory, signature):
    return load_bundle(Path(directory))


def pct(value, digits=2):
    return f"{value*100:.{digits}f}%".replace(".", ",")


def br_number(value, digits=0):
    return f"{value:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


directory = Path(os.environ.get("FRAUD_OUTPUT_DIR", str(ROOT/"outputs"))).resolve()
try:
    files = [directory/"metrics_report.json", *artifact_paths(directory).values()]
    signature = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    bundle = read_artifacts(str(directory), signature)
except Exception as exc:
    st.error(f"Não foi possível conferir os resultados: {exc}")
    st.info("Use uma execução completa do Módulo 2: metrics_report.json, score_deciles.csv, predictions.csv, shap_importance.csv e shap_summary.png da mesma pasta.")
    st.stop()

report, test = bundle.report, bundle.report["test"]
model_name = {"random_forest": "Random Forest", "xgboost": "XGBoost", "logistic_regression": "Regressão logística"}.get(report["selection"]["model"], report["selection"]["model"])
run_time = datetime.fromisoformat(report["created_at_utc"]).astimezone(ZoneInfo("America/Sao_Paulo"))
st.caption(f"Benchmark público OpenML {report['provenance'].get('openml_id', '—')} · {model_name} · execução em {run_time:%d/%m/%Y %H:%M} (Brasília).")

st.header("01 / Teste final, política congelada")
st.caption(f"{test['n']:,} transações e {test['fraud_count']} fraudes. Estes indicadores não mudam com o simulador abaixo.".replace(",", "."))
with st.container(horizontal=True, gap="medium"):
    st.metric("Taxa de fraude", pct(test["fraud_rate"], 3), border=True)
    st.metric("Captura nos 2 primeiros decis", pct(test["capture_top_2_deciles"]), border=True)
    st.metric("Falsos positivos", str(test["fp"]), border=True)
    st.metric("KS", pct(test["ks"]), border=True)
st.markdown(f"**{test['tp']} fraudes detectadas**, {test['fn']} não detectadas e {test['alerts']} alertas. "
            f"Recall: **{pct(test['recall'])}** · precisão: **{pct(test['precision'])}** · "
            f"Gini: **{pct(test['gini'])}**.")
st.caption(f"Decil 1 = maior risco. As duas primeiras faixas contêm {pct(test['population_top_2_deciles'])} das transações do teste; não necessariamente 20%, pois os limites vêm do treino e preservam empates.")

st.subheader("Onde estão as fraudes?")
deciles = bundle.deciles[bundle.deciles["split"] == "test"].copy()
chart = alt.Chart(deciles).mark_bar(color="#A3452D").encode(
    x=alt.X("risk_decile:O", title="Decil de risco — 1 = maior risco", sort=list(range(1, 11))),
    y=alt.Y("frauds:Q", title="Fraudes observadas", axis=alt.Axis(tickMinStep=1)),
    tooltip=[alt.Tooltip("risk_decile:O", title="Decil"), alt.Tooltip("transactions:Q", title="Transações", format=","),
             alt.Tooltip("frauds:Q", title="Fraudes"), alt.Tooltip("fraud_capture:Q", title="Captura", format=".2%"),
             alt.Tooltip("population_share:Q", title="Parcela da população", format=".2%")],
).properties(height=280)
st.altair_chart(chart, width="stretch")
with st.expander("Ver tabela de faixas e captura acumulada"):
    table = deciles[["risk_decile", "score_lower", "score_upper", "transactions", "frauds", "fraud_capture", "cumulative_capture"]].copy()
    table.columns = ["Decil de risco", "Score mínimo", "Limite superior", "Transações", "Fraudes", "Captura", "Captura acumulada"]
    table["Captura"] = table["Captura"].map(pct)
    table["Captura acumulada"] = table["Captura acumulada"].map(pct)
    st.dataframe(table, hide_index=True)
    st.caption("Limite inferior inclusivo e superior exclusivo; a última faixa inclui 1000. Faixas com limites iguais podem ficar vazias.")

st.header("02 / Laboratório de decisão")
st.info("Simulação somente na validação. Alterar os controles não retreina o modelo, não muda o limiar original nem recalcula os KPIs do teste final.")
validation = bundle.predictions[bundle.predictions["split"] == "validation"]
frozen_threshold = float(report["selection"]["threshold"])
with st.container(border=True):
    frozen = st.checkbox("Usar o limiar original da validação", value=True, key="use_frozen")
    no_alerts = st.checkbox("Simular política sem alertas", value=False, key="no_alerts")
    chosen = st.slider("Limiar de probabilidade de fraude", 0.0, 1.0,
                       value=min(round(frozen_threshold, 4), 1.0), step=0.0001,
                       format="%.4f", disabled=frozen or no_alerts, key="threshold")
    st.caption(f"Limiar original: {frozen_threshold:.8f}. Score educacional = 1000 × (1 − probabilidade); não é score de crédito calibrado ou produto da Serasa.")
    with st.container(horizontal=True):
        review = st.number_input("Custo de revisão por alerta (hipótese)", min_value=0.0,
                                 value=float(report["cost_assumptions"]["review"]), step=1.0, key="review")
        missed = st.number_input("Custo por fraude não detectada (hipótese)", min_value=0.0,
                                 value=float(report["cost_assumptions"]["missed_fraud"]), step=10.0, key="missed")
    st.caption("Unidades monetárias hipotéticas. Custo = revisão × (TP + FP) + fraude não detectada × FN. Toda transação alertada exige revisão; não há economia real comprovada.")

threshold = float(np.nextafter(1.0, np.inf)) if no_alerts else (frozen_threshold if frozen else chosen)
scenario = simulate_validation(validation, threshold, review, missed)
with st.container(horizontal=True):
    st.metric("Alertas na validação", str(scenario["alerts"]), border=True)
    st.metric("Fraudes capturadas na validação", str(scenario["tp"]), border=True)
    st.metric("Falsos positivos na validação", str(scenario["fp"]), border=True)
    st.metric("Custo simulado total", f"{br_number(scenario['simulated_cost'], 2)} u.m.", border=True)
st.caption(f"Validação: {br_number(scenario['n'])} transações · recall {pct(scenario['recall'])} · precisão {pct(scenario['precision'])} · {scenario['fn']} fraudes não detectadas.")
if scenario["alerts"] == 0:
    st.caption("Sem alertas, a precisão é exibida como zero por convenção; não há previsões positivas para avaliar.")
curve = scenario_curve(validation, review, missed, frozen_threshold)
curve = curve[curve.threshold <= 1]
cost_chart = alt.Chart(curve).mark_line(color="#344C47").encode(
    x=alt.X("threshold:Q", title="Limiar de decisão", scale=alt.Scale(domain=[0, 1])),
    y=alt.Y("cost_per_1000:Q", title="Custo hipotético por 1.000 transações"),
    tooltip=[alt.Tooltip("threshold:Q", title="Limiar", format=".4f"),
             alt.Tooltip("cost_per_1000:Q", title="Custo / 1.000", format=".2f"),
             alt.Tooltip("alerts:Q", title="Alertas")],
).properties(height=250)
if threshold <= 1:
    point = pd.DataFrame({"threshold": [threshold], "cost_per_1000": [scenario["cost_per_1000"]]})
    cost_chart = cost_chart + alt.Chart(point).mark_point(color="#A3452D", filled=True, size=90).encode(
        x="threshold:Q", y="cost_per_1000:Q", tooltip=[alt.Tooltip("threshold:Q", title="Cenário atual", format=".4f")])
st.altair_chart(cost_chart, width="stretch")
st.caption("Curva de cenários na validação. Nenhuma nova escolha é aplicada automaticamente ao teste.")
st.download_button("Baixar este cenário de validação", json.dumps(scenario, indent=2, ensure_ascii=False),
                   file_name="cenario_validacao.json", mime="application/json")

st.header("03 / O que influencia o modelo?")
st.markdown("O SHAP foi calculado em uma amostra da **validação**, não para escolher variáveis a partir do teste. Os valores descrevem associações do modelo, sem interpretação causal.")
st.image(str(bundle.output_dir/"shap_summary.png"), width="stretch")
st.caption(f"Unidade da explicação registrada: {report['shap']['units']}. V1–V28 são componentes PCA anônimos; não é possível atribuir-lhes significados como renda, idade ou limite de crédito.")
with st.expander("Ver importância média absoluta das variáveis"):
    st.dataframe(bundle.importance.rename(columns={"feature": "Variável transformada", "mean_abs_shap": "SHAP absoluto médio"}), hide_index=True)

st.header("04 / Rastreabilidade e limites")
st.markdown(f"**PSI validação → teste:** {report['psi']['value']:.6f}. É uma comparação entre amostras aleatórias do benchmark, não evidência de monitoramento em produção.")
with st.expander("Protocolo, dados e arquivos de origem"):
    st.markdown("Treino/validação/teste estratificados em 60/20/20; transformações ajustadas no treino; modelo e limiar escolhidos na validação; teste final reservado.")
    st.markdown(f"**Duplicatas removidas:** {report['data_quality']['duplicates_removed']}. **Hash da base:** `{report['provenance']['input_file_sha256']}`.")
    for limitation in report["limitations"]:
        st.markdown(f"- {limitation}")
    st.download_button("Baixar relatório original", (bundle.output_dir/"metrics_report.json").read_bytes(),
                       file_name="metrics_report.json", mime="application/json")
st.caption("Projeto educacional de João Victor Pereira Santos. Não contém dados da Serasa Experian e não representa seus modelos ou políticas.")

