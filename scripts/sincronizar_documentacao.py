"""Render public documentation from the completed ML report; no invented metrics."""
import argparse
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dashboard-url', required=True)
parser.add_argument('--source', type=Path, default=Path('outputs'))
parser.add_argument('--output-dir', type=Path, default=Path('.'))
parser.add_argument('--portfolio-dir', type=Path)
args = parser.parse_args()
url = urlparse(args.dashboard_url)
if url.scheme != 'https' or not url.netloc:
    parser.error('Informe a URL HTTPS efetivamente publicada.')
root = args.output_dir
root.mkdir(exist_ok=True, parents=True)
source = args.source
report_bytes = (source/'metrics_report.json').read_bytes()
report = json.loads(report_bytes)
assert report['status'] == 'completed'
for name in ['score_deciles.csv','shap_summary.png','pr_curve.png','score_distribution.png']:
    assert hashlib.sha256((source/name).read_bytes()).hexdigest() == report['artifact_sha256'][name]
deciles = pd.read_csv(source/'score_deciles.csv', float_precision='round_trip')
test = report['test']
validation = report['validation']
split = report['split_policy']['actual']
date = datetime.fromisoformat(report['created_at_utc']).astimezone(ZoneInfo('America/Sao_Paulo')).strftime('%d/%m/%Y %H:%M')
dashboard = args.dashboard_url.rstrip('/')+'/'
threshold = report['selection']['threshold']
names = {'random_forest':'Random Forest','xgboost':'XGBoost','logistic_regression':'Regressão logística'}
model = names[report['selection']['model']]
def num(x, digits=0):
    return f'{x:,.{digits}f}'.replace(',','_').replace('.',',').replace('_','.')
def pct(x, digits=2):
    return num(x*100,digits)+'%'
def metric_rows():
    return [('Transações',num(test['n'])),('Fraudes reais',num(test['fraud_count'])),
            ('Verdadeiros positivos (TP)',num(test['tp'])),('Falsos positivos (FP)',num(test['fp'])),
            ('Falsos negativos (FN)',num(test['fn'])),('Verdadeiros negativos (TN)',num(test['tn'])),
            ('Recall',pct(test['recall'])),('Precisão',pct(test['precision'])),('F1',num(test['f1'],4)),
            ('Average precision (AP)',num(test['average_precision'],4)),('PR-AUC trapezoidal',num(test['pr_auc_trapezoid'],4)),
            ('ROC-AUC',num(test['roc_auc'],4)),('KS',num(test['ks'],4)),('Gini',num(test['gini'],4)),
            ('Captura nas duas primeiras faixas',pct(test['capture_top_2_deciles'])),
            ('População nas duas primeiras faixas',pct(test['population_top_2_deciles']))]
metric_md='\n'.join(f'| {label} | {value} |' for label,value in metric_rows())
comparison_md='\n'.join(f"| {names[r['model']]} | {num(r['threshold'],6)} | {pct(r['recall'])} | {pct(r['precision'])} | {num(r['average_precision'],4)} | {num(r['simulated_cost'],2)} |" for r in report['selection']['models_compared'])
test_deciles=deciles[deciles.split=='test']
decile_md='\n'.join(f"| {int(r.risk_decile)} | {num(r.score_lower,3)} a {num(r.score_upper,3)} | {num(r.transactions)} | {num(r.frauds)} | {pct(r.fraud_capture)} | {pct(r.cumulative_capture)} | {pct(r.cumulative_population_share)} |" for r in test_deciles.itertuples())
readme=f'''# Detecção de fraude: do modelo à decisão

Estudo educacional de **João Victor Pereira Santos**, iniciado no Bootcamp Santander 2026 / DIO e evoluído para um experimento reproduzível de classificação de fraude. Usa o benchmark público de transações de cartão; não utiliza dados, modelos ou políticas da Serasa Experian.

[Dashboard público]({dashboard}) · [Estudo de caso no portfólio](https://joaovictor-portfolio-five.vercel.app/estudo-fraude.html) · [Relatório original](outputs/metrics_report.json) · [Consultas SQL](sql/queries_analiticas.sql)

## Problema de negócio

Como identificar fraudes sem gerar um volume impraticável de alertas? Acurácia isolada não responde a essa pergunta em uma base desbalanceada. O estudo compara classificadores e escolhe uma política considerando captura, falsos positivos e **custos explicitamente hipotéticos**.

## Dados e execução

- Fonte: [OpenML 1597 — creditcard](https://www.openml.org/search?type=data&id=1597), espelho do benchmark ULB/Worldline.
- Base original: {num(report['data_quality']['rows_before'])} transações e {num(report['data_quality']['fraud_count_before'])} fraudes.
- Após remover {num(report['data_quality']['duplicates_removed'])} duplicatas exatas: {num(report['data_quality']['rows_after'])} transações e {num(report['data_quality']['fraud_count_after'])} fraudes.
- Execução de referência: **{date}**, horário de Brasília; semente **{report['seed']}**.
- O ARFF é verificado contra o checksum publicado; a coluna `Time` original é preservada, mesmo sendo marcada como identificador de linha no OpenML.
- Hash SHA-256 do CSV processado como entrada: `{report['provenance']['input_file_sha256']}`.

## Da v1 à v2: correção metodológica

Na [v1](https://github.com/JVZerinho/Deteccao_Anomalias_Transacoes_Python/blob/dd6d69c45b5af4c6c8615d1eba8e29251cf7bad3/desafio.py), o escalonamento era ajustado na base completa e o conjunto de teste era usado para escolher o limiar. Esses procedimentos foram corrigidos na v2:

1. Verificações de esquema, tipos, valores ausentes, infinitos, duplicatas, conflitos de rótulo e consistência temporal.
2. Deduplicação determinística antes da divisão, evitando cópias idênticas em conjuntos diferentes.
3. Divisão estratificada **60/20/20** antes do ajuste das transformações: treino com {num(split['train']['n'])} linhas, validação com {num(split['validation']['n'])} e teste com {num(split['test']['n'])}.
4. `Pipeline` + `ColumnTransformer`: `RobustScaler.fit` somente no treino; validação e teste recebem apenas transformações. O código verifica o centro do scaler contra a mediana do treino.
5. Regressão logística, Random Forest e XGBoost treinados somente no treino. Modelos e limiares comparados exclusivamente na validação.
6. Escolha congelada; somente o modelo selecionado produz probabilidades para a avaliação final do teste. Não há retreino com a validação após a escolha.
7. SHAP calculado em uma amostra da validação; limites das faixas de score definidos no treino.

Os números da v1 foram substituídos pelos artefatos da execução v2. As versões não constituem uma comparação causal de desempenho: protocolo, divisão e base após deduplicação mudaram.

## Comparação na validação

Cada modelo usa seu limiar escolhido na validação. Critério: menor custo simulado total; desempate por menos alertas, maior AP e nome do modelo. Dentro de cada modelo, empates de custo favorecem menos alertas e maior limiar.

| Modelo | Limiar | Recall | Precisão | AP | Custo hipotético total |
| --- | ---: | ---: | ---: | ---: | ---: |
{comparison_md}

**Modelo selecionado: {model}**, limiar `{threshold:.12f}`. A escolha vale para este cenário, não significa superioridade universal.

Hipóteses de referência: revisão por alerta = **{num(report['cost_assumptions']['review'],2)} u.m.**; fraude não detectada = **{num(report['cost_assumptions']['missed_fraud'],2)} u.m.**. Fórmula: `revisão × (TP + FP) + fraude não detectada × FN`. Todo alerta demanda revisão. Esses valores não são custos de um banco nem economia financeira comprovada.

## Resultado no teste reservado

| Indicador | Resultado |
| --- | ---: |
{metric_md}

O modelo detectou **{test['tp']} das {test['fraud_count']} fraudes**, com **{test['fp']} falsos positivos** e **{test['fn']} fraudes não detectadas**. AP e PR-AUC trapezoidal são calculadas separadamente. O relatório contém todos os valores com precisão numérica original.

![Curva Precision-Recall do modelo selecionado](outputs/pr_curve.png)

## Score, decis e estabilidade

O score educacional é `1000 × (1 − p_fraude)`: score maior significa menor risco previsto. Não é score Serasa, score de crédito calibrado ou probabilidade de inadimplência. Os modelos usam pesos de classe e suas probabilidades não foram calibradas.

As nove fronteiras de quantis vêm do **treino**. Decil de risco 1 reúne menores scores e maior risco. Empates são preservados; faixas podem ter tamanhos diferentes ou ficar vazias. As duas primeiras faixas capturam **{pct(test['capture_top_2_deciles'])}** das fraudes e abrangem **{pct(test['population_top_2_deciles'])}** das transações do teste.

| Decil de risco | Intervalo de score* | Transações | Fraudes | Captura | Captura acumulada | População acumulada |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{decile_md}

*Intervalos arredondados nesta tabela. Limite inferior inclusivo e superior exclusivo; última faixa inclui 1000. Use os limites completos em [score_deciles.csv](outputs/score_deciles.csv).

- **KS:** máximo absoluto da diferença entre TPR e FPR.
- **Gini:** `2 × ROC_AUC − 1`.
- **PSI validação → teste:** **{num(report['psi']['value'],6)}**, usando limites do treino e pseudocontagem de 0,5 por faixa. É comparação entre amostras aleatórias, não evidência de drift temporal ou monitoramento em produção.

![Distribuição do score educacional](outputs/score_distribution.png)

## Interpretação com SHAP

O gráfico explica uma amostra de {report['shap']['n_explained']} observações da validação. Para o modelo selecionado, a unidade registrada é `{report['shap']['units']}`. SHAP mostra associações do modelo, não causalidade. V1–V28 são componentes PCA anônimos e não podem ser interpretados como renda, idade ou limite de crédito.

![SHAP na validação](outputs/shap_summary.png)

## SQL analítico

O [arquivo SQL completo](sql/queries_analiticas.sql) contém esquema e quatro análises; o [carregador](sql/carregar_e_consultar.py) importa previsões com comandos parametrizados:

1. `GROUP BY` e `CASE WHEN`: taxa de fraude por faixa de valor e hora relativa, valor fraudulento observado e custos simulados.
2. Funções de janela: média das últimas dez transações, `LAG` e alertas acumulados por conjunto.
3. CTEs e JOINs: falsos positivos de alto valor na política original. Com limite hipotético de 1.000, a execução de referência retorna zero linhas; não há dados fictícios nos resultados.
4. Coorte por dia relativo de transação. Não é safra de originação ou maturação de crédito.

As quatro consultas foram executadas em SQLite e PostgreSQL 18.6 via PGlite, sobre {num(split['validation']['n']+split['test']['n'])} transações, com valores equivalentes e reconciliação independente em Pandas. A conexão psycopg a servidor PostgreSQL nativo não foi exercitada. A base não oferece ID de cliente/cartão; as janelas são globais por conjunto. `Time` não é horário civil e valor fraudulento não é perda realizada.

## Dashboard

**[Abrir o dashboard público]({dashboard})**

O dashboard lê o relatório e as previsões desta execução. Os KPIs iniciais são do teste final e permanecem congelados. O simulador usa **somente validação**, com limiar e hipóteses de custo ajustáveis, sem treinar ou selecionar novos modelos. As previsões compactadas são conferidas contra o hash do CSV original antes de exibir números.

![Dashboard com resultados da execução](assets/dashboard-preview.jpg)

## Instalação e execução completas

Use **Python 3.12**. Clone o repositório e entre na pasta:

```bash
git clone https://github.com/JVZerinho/Deteccao_Anomalias_Transacoes_Python.git
cd Deteccao_Anomalias_Transacoes_Python
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-dashboard.txt
.\\.venv\\Scripts\\python.exe -m pip check
.\\.venv\\Scripts\\python.exe -m unittest -v test_metodologia
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
.\\.venv\\Scripts\\python.exe -m streamlit run dashboard/app_dashboard.py
```

Linux/macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-dashboard.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest -v test_metodologia
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m streamlit run dashboard/app_dashboard.py
```

Para abrir o dashboard e executar SQL, basta instalar `requirements-dashboard.txt`: os artefatos verificados acompanham o repositório. O modelo serializado e a base original não são necessários para a aplicação.

Para gerar um **novo experimento**, preservando o resultado publicado, instale `requirements.txt` e use outro diretório de saída:

```bash
python desafio.py --openml-id 1597 --output-dir outputs-nova-execucao
```

Ou forneça CSV com exatamente `Time`, `V1`–`V28`, `Amount`, `Class`:

```bash
python desafio.py --data data/creditcard.csv --source-note "Origem e versão do CSV" --output-dir outputs-nova-execucao
```

Custos são parâmetros de validação: `--review-cost 5 --missed-fraud-cost 500`. Não escolha novos parâmetros em resposta ao desempenho do teste. O programa recusa saída não vazia para não misturar execuções. O download original não é redistribuído e o antigo Dropbox não é necessário.

Para carregar e consultar SQLite:

```bash
python sql/carregar_e_consultar.py --engine sqlite
```

Para PostgreSQL, defina `DATABASE_URL` no ambiente local ou no serviço de segredos e use `--engine postgres`. Não coloque credenciais no código. Mais detalhes em [EXECUCAO_MODULO_3.md](EXECUCAO_MODULO_3.md).

## Artefatos e rastreabilidade

- [metrics_report.json](outputs/metrics_report.json): origem, hashes, ambiente, hipóteses, divisão, seleção e métricas.
- [score_deciles.csv](outputs/score_deciles.csv): faixas e captura de validação/teste.
- `outputs/predictions.csv.gz`: previsões de validação/teste, sem treinamento no início do dashboard; a leitura confere o hash do conteúdo descompactado.
- [SQL, interface e reconciliação](VALIDACAO_MODULO_3.json); [verificação metodológica](VALIDACAO_MODULO_2.json).
- Gerador desta documentação: `scripts/sincronizar_documentacao.py --dashboard-url URL_PUBLICADA`; os números são derivados do JSON/CSV, não digitados como estimativas.

Não carregue pipelines `joblib` de terceiros. O dashboard utiliza apenas JSON, CSV/GZip e imagens geradas pelo projeto.

## Limitações reais

Divisão aleatória não mede generalização futura; não há isolamento por cliente/cartão verificável. A PCA foi fornecida pronta e não há acesso ao seu ajuste original. O teste contém apenas {test['fraud_count']} fraudes, sem intervalo de confiança implementado. O score não foi calibrado, custos são hipóteses e não há impacto financeiro real ou implantação de modelo em produção demonstrados. Nenhuma métrica é automaticamente transferível para dados da Serasa Experian.
'''
(root/'README.md').write_text(readme,encoding='utf-8')

if args.portfolio_dir:
    site = args.portfolio_dir
    assets=site/'assets/fraude'
    assets.mkdir(parents=True,exist_ok=True)
    for name in ['metrics_report.json','score_deciles.csv','pr_curve.png','shap_summary.png','score_distribution.png']:
        shutil.copy2(source/name,assets/name)

    metric_html=''.join(f'<tr><th scope="row">{html.escape(label)}</th><td>{html.escape(value)}</td></tr>' for label,value in metric_rows())
    decile_html=''.join(f'<tr><td>{int(r.risk_decile)}</td><td>{num(r.transactions)}</td><td>{num(r.frauds)}</td><td>{pct(r.fraud_capture)}</td><td>{pct(r.cumulative_capture)}</td></tr>' for r in test_deciles.itertuples())
    head=(site/'estudo-fraude.html').read_text(encoding='utf-8').split('<main id="estudo"')[0]
    foot='<footer>'+ (site/'estudo-fraude.html').read_text(encoding='utf-8').split('<footer>')[1]
    case=f'''<main id="estudo" class="case-page">
    <section class="case-hero"><p class="eyebrow">ESTUDO 01 / VERSÃO 2 · PYTHON · SQL · ANALYTICS</p><h1>Fraude: o custo<br>de cada <em>decisão.</em></h1><p class="lead">Um experimento reproduzível que conecta comparação de modelos, avaliação de alertas e interpretação dos resultados.</p><div class="tags"><span>Projeto educacional</span><span>Benchmark público</span><span>Execução: {date}</span></div><div class="case-actions"><a class="button" href="{html.escape(dashboard)}" target="_blank" rel="noopener noreferrer">Explorar dashboard <span>↗</span></a><a class="text-link" href="https://github.com/JVZerinho/Deteccao_Anomalias_Transacoes_Python" target="_blank" rel="noopener noreferrer">Código e reprodução ↗</a></div></section>
    <section class="case-section"><p class="eyebrow">01 / PROBLEMA & DADOS</p><div class="about-grid"><h2>Detectar a exceção.<br><em>Medir as consequências.</em></h2><div><p>A base pública contém {num(report['data_quality']['rows_before'])} transações e {num(report['data_quality']['fraud_count_before'])} fraudes. Removi {num(report['data_quality']['duplicates_removed'])} duplicatas exatas antes da divisão, restando {num(report['data_quality']['rows_after'])} transações e {num(report['data_quality']['fraud_count_after'])} fraudes.</p><p>A pergunta é como detectar fraudes sem tratar acurácia alta como sucesso e sem ignorar o volume de alertas falsos.</p><p class="source-note">Fonte: OpenML 1597. Os números desta página foram gerados a partir de metrics_report.json e score_deciles.csv da execução v2.</p></div></div></section>
    <section class="case-section"><p class="eyebrow">02 / O QUE FOI CORRIGIDO</p><h2>Da exploração<br><em>ao teste reservado.</em></h2><ol class="analysis-steps"><li><span>01</span><div><h3>Auditar e separar</h3><p>Esquema, nulos, tipos, duplicatas e tempo verificados. Divisão estratificada em 60% treino, 20% validação e 20% teste antes de ajustar transformações.</p></div></li><li><span>02</span><div><h3>Ajustar somente no treino</h3><p>Pipeline e ColumnTransformer mantêm o ajuste do RobustScaler restrito ao treino. A v1 ajustava o escalonamento na base inteira; esse procedimento foi corrigido.</p></div></li><li><span>03</span><div><h3>Escolher na validação</h3><p>Regressão logística, Random Forest e XGBoost comparados com limiares da curva Precision-Recall e custos hipotéticos. A v1 usava o teste na escolha do limiar; a v2 utiliza validação.</p></div></li><li><span>04</span><div><h3>Congelar e avaliar</h3><p>Modelo e limiar congelados antes do teste. Sem retreino após a escolha e sem ajuste a partir do resultado final. SHAP calculado na validação.</p></div></li></ol></section>
    <section class="case-section metric-section"><p class="eyebrow">03 / RESULTADO MEDIDO</p><h2>{test['tp']} fraudes detectadas.<br><em>{test['fp']} falsos positivos.</em></h2><p>O {model} foi escolhido na validação para o cenário de custos declarado. No teste reservado, encontrou {test['tp']} das {test['fraud_count']} fraudes em {num(test['n'])} transações, com recall de {pct(test['recall'])}, precisão de {pct(test['precision'])} e {test['fn']} fraudes não detectadas.</p><p class="source-note">Limiar congelado: {threshold:.8f}. Comparações entre os três modelos usam a validação; o teste final foi aplicado somente ao selecionado.</p><div class="table-wrap"><table><caption>Teste final · política escolhida na validação</caption><thead><tr><th scope="col">Indicador</th><th scope="col">Resultado</th></tr></thead><tbody>{metric_html}</tbody></table></div><img class="evidence-image" src="assets/fraude/pr_curve.png" alt="Curvas Precision-Recall da validação e teste do modelo selecionado" loading="lazy"><div class="decision-note"><p class="eyebrow">LEITURA DE NEGÓCIO</p><p>Custo por revisão: {num(report['cost_assumptions']['review'],2)} u.m.; por fraude não detectada: {num(report['cost_assumptions']['missed_fraud'],2)} u.m. São hipóteses parametrizáveis. O dashboard simula cenários somente na validação e mantém o teste congelado. Não há economia financeira real comprovada.</p></div></section>
    <section class="case-section"><p class="eyebrow">04 / SCORE & CAPTURA</p><h2>Uma escala para<br><em>ordenar o risco.</em></h2><p>Score educacional = 1000 × (1 − probabilidade de fraude). O decil 1 reúne menores scores e maior risco. As fronteiras vêm do treino e preservam empates. As duas primeiras faixas capturam {pct(test['capture_top_2_deciles'])} das fraudes e abrangem {pct(test['population_top_2_deciles'])} das transações do teste.</p><div class="table-wrap"><table><caption>Captura por faixa de risco no teste</caption><thead><tr><th>Decil</th><th>Transações</th><th>Fraudes</th><th>Captura</th><th>Acumulada</th></tr></thead><tbody>{decile_html}</tbody></table></div><p>KS: {num(test['ks'],4)} · Gini: {num(test['gini'],4)} · PSI validação → teste: {num(report['psi']['value'],6)}.</p><p class="source-note">O score não é calibrado nem equivale a produto da Serasa. O PSI compara amostras aleatórias; não comprova monitoramento em produção.</p></section>
    <section class="case-section"><p class="eyebrow">05 / SQL & EXPLICAÇÃO</p><h2>Consultar, auditar<br><em>e explicar.</em></h2><div class="skills"><div><span>SQL ANALÍTICO</span><h4>Quatro consultas completas</h4><p>Faixas de valor/hora, média móvel e LAG, auditoria de falsos positivos com CTEs e JOINs e coortes por dia relativo.</p></div><div><span>VERIFICAÇÃO</span><h4>Dois motores de SQL</h4><p>Resultados equivalentes em SQLite e PostgreSQL via PGlite, reconciliados com Pandas. Não foi testada conexão com servidor PostgreSQL nativo.</p></div><div><span>SHAP</span><h4>Associações do modelo</h4><p>Explicações sobre uma amostra da validação. V1–V28 são componentes PCA anônimos; não permitem atribuir significados como renda ou idade.</p></div></div><img class="evidence-image" src="assets/fraude/shap_summary.png" alt="SHAP calculado na validação para o Random Forest selecionado" loading="lazy"><a class="text-link" href="https://github.com/JVZerinho/Deteccao_Anomalias_Transacoes_Python/blob/main/sql/queries_analiticas.sql" target="_blank" rel="noopener noreferrer">Inspecionar consultas SQL ↗</a></section>
    <section class="case-section"><p class="eyebrow">06 / DASHBOARD PÚBLICO</p><h2>Explore o resultado.<br><em>Questione o cenário.</em></h2><p class="lead">Indicadores do teste, captura por faixa, simulador restrito à validação e evidências do SHAP, com hipóteses e limites visíveis.</p><a class="button" href="{html.escape(dashboard)}" target="_blank" rel="noopener noreferrer">Abrir dashboard <span>↗</span></a><img class="evidence-image dashboard-evidence" src="assets/fraude/dashboard-preview.jpg" alt="Captura de tela do dashboard publicado" loading="lazy"></section>
    <section class="case-section"><p class="eyebrow">07 / LIMITAÇÕES REAIS</p><div class="about-grid"><h2>Um estudo completo.<br><em>Um escopo definido.</em></h2><div><p>A divisão aleatória não mede desempenho futuro. A base não possui ID de cliente/cartão, e o ajuste da PCA original não está disponível. O teste contém apenas {test['fraud_count']} fraudes, sem intervalo de confiança implementado.</p><p>Score não calibrado, custos hipotéticos e ausência de validação em operação real limitam as conclusões. As métricas da v1 não são uma referência de ganho causal, pois protocolo e dados após deduplicação mudaram.</p><p>Os dois problemas metodológicos identificados na v1 foram corrigidos e a v2 foi executada. Isso não demonstra generalização para dados ou políticas de uma instituição financeira.</p><a class="text-link" href="assets/fraude/metrics_report.json" target="_blank" rel="noopener noreferrer">Consultar relatório original da execução ↗</a></div></div></section>
    <section class="case-section case-end"><p class="eyebrow">PRÓXIMO PASSO / ANALYTICS</p><h2>Uma base para<br><em>aprender com o time.</em></h2><p class="lead">Quero contribuir com exploração de dados, avaliação de modelos e comunicação de resultados, aprofundando essa prática em uma oportunidade de estágio.</p><a class="button" href="mailto:jvps.2014.jvps@gmail.com">Vamos conversar <span>↗</span></a></section></main>'''
    (site/'estudo-fraude.html').write_text(head+case+foot,encoding='utf-8')

    p=site/'index.html'
    index=p.read_text(encoding='utf-8')
    index=index.replace('Estudo de machine learning para identificar fraudes em uma base desbalanceada. Comparo três modelos e exploro como o limiar de decisão altera o equilíbrio entre fraudes detectadas e falsos alertas.', f'Estudo reproduzível com treino, validação e teste separados. O modelo escolhido na validação detectou {test["tp"]} de {test["fraud_count"]} fraudes no teste, com {test["fp"]} falsos positivos. Recall: {pct(test["recall"])}; precisão: {pct(test["precision"])}.')
    index=index.replace('O estudo documenta RobustScaler, divisão estratificada, comparação entre regressão logística, Random Forest e XGBoost e ajuste do limiar de decisão. O SHAP ajuda a interpretar as variáveis que influenciam o modelo. O estudo de caso apresenta o raciocínio e as limitações metodológicas identificadas na versão atual. As métricas do README ainda precisam de revalidação.', 'Pipeline com transformações ajustadas apenas no treino; modelo e limiar escolhidos na validação. A versão 2 foi executada e exporta métricas, faixas de score, KS, Gini, PSI e SHAP. SQL analítico verificado em SQLite e PostgreSQL via PGlite. Score e custos são educacionais; o estudo não comprova desempenho em produção.')
    index=index.replace('<span>scikit-learn</span><span>SHAP</span>', '<span>scikit-learn</span><span>SQL</span><span>Streamlit</span><span>SHAP</span>')
    index=re.sub(r'<a class="text-link dashboard-card-link".*?</a>', '', index)
    index=index.replace('<a class="case-link" href="estudo-fraude.html">Ler estudo de caso <span>↗</span></a>', '<a class="case-link" href="estudo-fraude.html">Ler estudo de caso <span>↗</span></a><a class="text-link dashboard-card-link" href="'+html.escape(dashboard)+'" target="_blank" rel="noopener noreferrer">Explorar dashboard público ↗</a>')
    index=index.replace('SQL, PostgreSQL e SQLite em desenvolvimento.', 'SQL aplicado a consultas analíticas com GROUP BY, CTEs, JOINs e funções de janela.')
    index=index.replace('Documentação das escolhas, comparação de métricas e interpretação das variáveis com SHAP.', 'Estudo reproduzível, dashboard com KPIs e simulação na validação, consultas SQL e interpretação com SHAP.')
    index=index.replace('Conhecimentos em SQL aplicado', 'SQL aplicado')
    p.write_text(index,encoding='utf-8')
    p=site/'analytics.css'
    css=p.read_text(encoding='utf-8')
    if '.evidence-image{' not in css:
        p.write_text(css+'\n.evidence-image{display:block;width:100%;height:auto;max-width:1000px;margin:30px auto;border:1px solid var(--line)}.dashboard-card-link{width:max-content;margin-bottom:20px}.case-section>p{max-width:900px}.dashboard-evidence{margin-top:35px}\n',encoding='utf-8')
    for name in ['index.html','estudo-fraude.html']:
        shutil.copy2(site/name,root/name)
(root/'fonte-documentacao.json').write_text(json.dumps({'metrics_report_sha256':hashlib.sha256(report_bytes).hexdigest(),'dashboard_url':dashboard,'schema_version':report['schema_version'],'generated_from_artifacts':True},indent=2)+'\n',encoding='utf-8')
print('README, card e estudo renderizados dos artefatos verificados. Dashboard:', dashboard)

