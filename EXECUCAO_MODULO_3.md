# Módulo 3 — SQL analítico e dashboard

O pacote entrega consultas executáveis e um dashboard Streamlit sobre os resultados reais do Módulo 2. Não treina modelos, não altera o teste final e não publica automaticamente um dashboard na internet.

## Abrir o dashboard

Use Python 3.12. Execute os comandos na raiz do pacote, para carregar também `.streamlit/config.toml`.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m streamlit run dashboard/app_dashboard.py
```

Linux/macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dashboard.txt
.venv/bin/python -m pip check
.venv/bin/python -m streamlit run dashboard/app_dashboard.py
```

`requirements-dashboard-lock.txt` registra as dependências transitivas do ambiente verificado. Para reproduzir esse ambiente Windows/Python 3.12, use esse arquivo no lugar de `requirements-dashboard.txt`.

O endereço exibido por `streamlit run` é local, não um link público para recrutadores. A publicação do site do portfólio continua independente. Não há implantação pública do Streamlit neste módulo.

## Contrato de dados

O ZIP inclui os arquivos necessários em `outputs/`:

- `metrics_report.json` e `score_deciles.csv`;
- `predictions.csv`, necessário para simular limiares sem retreinar;
- `shap_importance.csv` e `shap_summary.png`.

O dashboard verifica os hashes registrados no relatório, a matriz de confusão, os scores, as faixas e as contagens. Um artefato ausente ou incompatível interrompe o carregamento em vez de produzir KPIs fictícios. O cache considera o tamanho e a data de modificação dos arquivos.

O GitHub contém código e resultados resumidos; arquivos grandes de previsões continuam fora do repositório. Para executar a partir de um clone, use os artefatos do pacote completo ou gere uma execução completa do pipeline do Módulo 2. Para gerar novamente, instale também `requirements.txt`, use um diretório vazio e respeite o protocolo do teste reservado. Não ajuste o modelo depois de inspecionar o teste.

Para apontar para outra execução completa, defina `FRAUD_OUTPUT_DIR` no ambiente antes de iniciar o Streamlit. O caminho é configurado pelo operador; não existe envio público de arquivos ou acesso a modelos serializados no dashboard.

## Como ler o dashboard

- Os quatro KPIs iniciais são do **teste final**, na política congelada: taxa de fraude, captura nas duas primeiras faixas de risco, falsos positivos e KS.
- Decil de risco 1 significa menor score e maior risco previsto. A cobertura dos dois primeiros decis pode diferir de 20% e é exibida junto à captura.
- O simulador usa exclusivamente as previsões da **validação**. Os controles não recalculam os KPIs de teste nem alteram o JSON original.
- Os custos de revisão e de fraude não detectada são hipóteses em unidades monetárias. A fórmula é `revisão × (TP+FP) + fraude não detectada × FN`. Nenhum valor representa preço, política ou economia de uma instituição.
- O botão de cenário exporta somente um JSON de simulação identificado como validação. O botão do relatório baixa o arquivo original sem modificação.
- O SHAP é o artefato real calculado na validação no Módulo 2. As variáveis PCA não permitem inferir atributos como renda ou idade.
- PSI entre validação e teste não comprova monitoramento em produção. Score 0–1000 é educacional, não score Serasa ou de crédito calibrado.

## Executar as consultas em SQLite

Com o ambiente ativado ou usando o caminho de seu Python:

```bash
python sql/carregar_e_consultar.py --engine sqlite
```

O carregador cria `outputs/analytics.sqlite`, carrega as tabelas com parâmetros, aplica as consultas do arquivo `sql/queries_analiticas.sql` e exporta quatro CSVs em `outputs/sql_results/`, mais um manifesto. O arquivo SQL contém o esquema e os SELECTs completos, divididos por comentários `-- @section`.

SQL não possui um comando universal de importação de CSV entre os dois bancos. Por isso, a carga usa o adaptador Python e as consultas analíticas permanecem no mesmo arquivo SQL para ambos.

O banco precisa ter as tabelas de destino vazias. Para executar um novo cenário sem sobrescrever o banco anterior:

```bash
python sql/carregar_e_consultar.py --engine sqlite --database outputs/analytics-cenario-2.sqlite --sql-results outputs/sql-cenario-2 --high-value-cutoff 1000
```

## Executar em PostgreSQL

Use um banco/schema vazio próprio para este estudo. Defina `DATABASE_URL` no ambiente local ou no gerenciador de segredos do seu serviço, sem colocar credenciais no código, no Git ou na linha de comando. Depois:

```bash
python sql/carregar_e_consultar.py --engine postgres --sql-results outputs/sql-postgres
```

O adaptador usa psycopg e comandos de carga parametrizados. Não apaga tabelas nem substitui transações já existentes.

## O que cada consulta demonstra

1. **GROUP BY + CASE WHEN:** taxa de fraude e valor de transações fraudulentas por faixa de valor e hora relativa; perda simulada por fraudes não detectadas e custo simulado de revisão. Valor da compra fraudulenta não é perda financeira realizada.
2. **Window functions:** média móvel das últimas 10 transações, diferença de tempo e alertas acumulados. A partição é por conjunto, com desempate pelo ID. São janelas globais, não de comportamento de cliente: a base não oferece ID de cartão/cliente.
3. **CTEs + JOINs:** falsos positivos de alto valor, segundo a política original de alertas. O limite configurável de 1.000 é uma hipótese analítica. Nesta execução, a consulta retorna zero linhas; não foram criados registros artificiais.
4. **Coorte temporal:** taxa de fraude por dia relativo desde o início da base. Não é safra de originação de crédito, atraso ou maturação de inadimplência.

`Time` está em segundos inteiros neste contrato. A carga rejeita valores fracionários para impedir divergências de arredondamento entre SQLite e PostgreSQL. Dia/hora relativos não são datas nem horários civis.

## Verificação

```bash
python -m unittest discover -s tests -v
```

Os dez testes cobrem consultas com dados de fronteira, carga repetida, simulador exclusivamente na validação, ausência de artefatos e interação da interface via `streamlit.testing.v1.AppTest`. Os KPIs do teste permanecem iguais depois de alterar custo e limiar.

Também foi executado o mesmo SQL em PostgreSQL compilado para WebAssembly, via **PGlite 0.5.11 / PostgreSQL 18.6**, sobre as 113.491 transações reais exportadas. As quatro saídas foram comparadas com SQLite e reconciliadas independentemente com Pandas. Isso verifica os cálculos SQL nos dois motores; não comprova conexão, permissões ou desempenho de um servidor PostgreSQL nativo.

Para reproduzir essa comparação opcional, com Node.js/npm instalados:

```bash
python sql/carregar_e_consultar.py --engine sqlite
python tests/preparar_postgres.py
npm --prefix tests ci
node tests/verify_postgres.mjs
python tests/validar_resultados.py
```

Se o SQLite já estiver carregado, pule o primeiro comando ou use um novo banco/diretório de resultados. `tests/validar_resultados.py` espera as saídas padrão em `outputs/sql_results` e `outputs/postgres_results`.

Os arquivos intermediários de carga PGlite e o banco SQLite não acompanham o ZIP; os comandos os geram novamente. O pacote inclui as saídas SQL e o relatório de validação.

## Referências

- [PostgreSQL: funções de janela](https://www.postgresql.org/docs/current/tutorial-window.html).
- [SQLite: funções de janela](https://www.sqlite.org/windowfunctions.html).
- [PGlite: PostgreSQL em WebAssembly](https://pglite.dev/docs/).

O README principal e a página de estudo do portfólio serão sincronizados no Módulo 4. O DevOps permanece reservado ao Módulo 5.
