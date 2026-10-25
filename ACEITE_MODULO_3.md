# Aceite do Módulo 3

## Estado

Implementado e executado com os artefatos reais do Módulo 2. Nenhum treinamento ou ajuste de modelo foi realizado neste módulo. O relatório original do Módulo 2 e seus KPIs do teste permanecem preservados.

## Critérios de aceite

| Item | Evidência |
| --- | --- |
| SQL completo e carga | `sql/queries_analiticas.sql` + carga parametrizada em `sql/carregar_e_consultar.py` |
| GROUP BY / CASE WHEN | Taxa de fraude por faixa de valor e hora relativa, com separação entre valor observado e custo hipotético |
| Window functions | Média das últimas 10 transações, LAG e alertas acumulados por conjunto |
| CTEs / JOINs | Auditoria de falsos positivos de alto valor, com limite configurável |
| Coorte temporal | Taxa de fraude por dia relativo; sem alegar safra de crédito |
| Compatibilidade SQL | Quatro consultas executadas em SQLite e PostgreSQL 18.6 via PGlite 0.5.11; valores comparados |
| KPIs do dashboard | Taxa de fraude, captura nos dois primeiros decis, falsos positivos e KS lidos do relatório e reconciliados com os CSVs |
| Simulador | Somente validação; custos hipotéticos explícitos; política do teste não muda |
| Score e SHAP | Gráfico de captura por faixa e SHAP real da validação |
| Rastreabilidade | Hashes, contagens e faixas conferidos antes de exibir resultados |
| Testes | Dez testes aprovados, incluindo AppTest com alteração de custo e limiar |
| Verificação visual | Página inicial inspecionada no navegador, KPIs e gráfico de captura legíveis; servidor respondeu HTTP 200 |

## Quantidades verificadas

- Transações carregadas: 113.491 (somente validação e teste).
- Consulta 1: 380 grupos.
- Consulta 2: 113.491 linhas.
- Consulta 3: zero falsos positivos com valor >= 1.000 na política original. O caso não vazio é coberto por fixture identificada nos testes, sem adicionar dados fictícios aos resultados reais.
- Consulta 4: quatro grupos de conjunto/dia relativo.

## Limites de verificação

- A comparação PostgreSQL usa motor PostgreSQL real compilado para WebAssembly (PGlite), não um servidor PostgreSQL nativo. O adaptador psycopg está implementado, mas sua conexão a servidor externo não foi exercitada.
- O dashboard foi executado e validado localmente. Não existe implantação pública do Streamlit neste módulo.
- O conjunto não contém identificador de cliente/cartão; as janelas não demonstram análise comportamental individual.
- Valor fraudulento observado não é perda financeira realizada. Custos de revisão e fraude são hipóteses.
- O pacote contém os artefatos necessários para abrir a aplicação. O GitHub mantém o código e os resultados resumidos; previsões volumosas precisam do pacote ou de geração pelo Módulo 2.

## Sequência

O README principal de fraude e a página `estudo-fraude.html` continuam reservados ao Módulo 4. O DevOps continua reservado ao Módulo 5.

Aguardar comando expresso: **Avançar para o Módulo 4**.
