# Módulo 2 — execução reproduzível

Este pacote contém o pipeline v2 e verificações metodológicas. Não implementa os módulos 3–5 e não substitui ainda o README principal ou o portfólio.

## Instalação

Use Python **3.12**. Na pasta que contém `desafio.py` e `requirements.txt`:

`requirements.txt` fixa as dependências diretas. `requirements-lock.txt` registra também as dependências transitivas do ambiente Windows/Python 3.12 que foi usado na verificação; para reproduzir esse ambiente, substitua `-r requirements.txt` por `-r requirements-lock.txt` nos comandos de instalação.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest -v test_metodologia
.\.venv\Scripts\python.exe desafio.py --openml-id 1597 --output-dir outputs
```

Linux/macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest -v test_metodologia
.venv/bin/python desafio.py --openml-id 1597 --output-dir outputs
```

Alternativa com arquivo local, sem download automático:

```bash
python desafio.py --data data/creditcard.csv --source-note "Origem e versão do CSV" --output-dir outputs
```

O CSV deve conter exatamente `Time`, `V1` a `V28`, `Amount` e `Class`. Não inclua índice exportado como coluna adicional. O antigo link Dropbox não é necessário. O download opcional usa o espelho público [OpenML 1597](https://www.openml.org/search?type=data&id=1597), com origem e hash registrados. Não confunda a base com dados da Serasa Experian. A base de entrada não acompanha o pacote nem deve ser publicada automaticamente no repositório.

O carregador verifica o MD5 publicado pelo OpenML e lê o ARFF original. Isso preserva `Time`, que esse registro do OpenML marca como `row_id_attribute` e que o carregador padrão `fetch_openml` exclui das features. A coluna temporal não é reconstruída a partir da ordem das linhas.

## Método

1. Verificações de esquema, tipos, valores ausentes, infinitos, duplicatas, rótulos conflitantes e consistência temporal. Problemas de entrada interrompem a execução em vez de produzir métricas fictícias.
2. Duplicatas exatas são removidas antes de dividir. O número removido e a mudança na proporção de fraudes ficam registrados.
3. Divisão estratificada em 60% treino, 20% validação e 20% teste, com arredondamento necessário de linhas. Os IDs de origem ficam em `split_assignments.csv`.
4. Um `Pipeline` independente para cada modelo. O `ColumnTransformer` ajusta o `RobustScaler` somente em `Time` e `Amount` do treino. O logaritmo de `Amount` não estima distribuição. As componentes PCA fornecidas pela base são preservadas.
5. Cada modelo é treinado somente no treino. Os limiares são os pontos da curva Precision-Recall da validação, mais a política de não emitir alertas. Modelo e limiar minimizam o custo simulado da validação. Empates favorecem menos alertas, maior average precision e nome do modelo; dentro de um modelo, menos alertas e maior limiar.
6. A escolha fica congelada. Não há retreino com validação nem escolha de novo limiar depois de ver o teste. Somente o modelo escolhido produz probabilidades no teste, em uma chamada.
7. SHAP explica uma amostra da validação, com fundo do treino quando aplicável. O teste não é usado para selecionar variáveis ou explicar candidatos durante a escolha.

O script faz uma avaliação final do teste por execução. Isso não impede um operador de ajustar o experimento e executar novamente: se fizer isso após observar resultados do teste, esse conjunto deixa de ser uma avaliação final independente para a nova escolha. Preserve o protocolo e registre as execuções.

## Hipóteses de custo

```bash
python desafio.py --data data/creditcard.csv --review-cost 5 --missed-fraud-cost 500 --output-dir outputs-cenario-1
```

Esses valores padrão são **hipóteses em unidades monetárias**, não custos de uma instituição. Fórmula: `custo_revisão × (TP + FP) + custo_fraude_perdida × FN`. Toda transação alertada exige revisão, inclusive um verdadeiro positivo. Não é assumida perda zero em fraudes detectadas como um fato operacional nem demonstrada economia real. A escolha não precisa maximizar F1: ela atende a esse cenário explícito de custos. Use cenários previamente definidos na validação, sem procurar um cenário que favoreça o resultado do teste.

## Score, faixas e estabilidade

- Score educacional contínuo: `1000 × (1 − p_fraude)`. Score alto significa menor risco previsto. As probabilidades não foram calibradas; o score não equivale a um bureau, probabilidade de crédito ou produto proprietário.
- As nove fronteiras de quantis vêm exclusivamente do score do **treino** do modelo escolhido. Aplicam-se sem alteração à validação e ao teste.
- Decil de risco 1 = menores scores, maior risco. Top 2 decis = faixas 1 e 2. Empates não são separados artificialmente; algumas faixas podem ficar vazias e os dois primeiros grupos podem não conter exatamente 20% das transações. A cobertura real da população é exportada.
- KS é o máximo absoluto da diferença entre TPR e FPR. Gini = `2 × ROC_AUC − 1`. Ambos são registrados na escala 0–1 (Gini pode ser negativo).
- PSI compara validação como referência com teste, usando faixas fixadas no treino. Há pseudocontagem de 0,5 por faixa e renormalização para tratar zeros. É comparação entre amostras do benchmark; não comprova drift temporal nem monitoramento em produção.

## Artefatos

- `outputs/metrics_report.json`: status, origem, hashes, ambiente, qualidade, divisão, seleção, hipóteses, métricas, score e PSI; escrito somente ao terminar com sucesso.
- `outputs/score_deciles.csv`: 10 faixas por conjunto (validação e teste), captura, taxa, cobertura e captura acumuladas.
- `outputs/pr_curve.png`, `shap_summary.png`, `score_distribution.png`: gráficos gerados na execução.
- `outputs/validation_model_comparison.csv` e `validation_thresholds.csv`: evidência de seleção exclusivamente na validação.
- `outputs/predictions.csv`: IDs de origem, conjunto, tempo relativo, valor, rótulo, probabilidade, score, faixa e alerta. Suporte para o próximo módulo, ainda sem SQL ou dashboard.
- `outputs/data_quality_report.json`, `split_assignments.csv`, `shap_importance.csv`: rastreabilidade.
- `outputs/selected_pipeline.joblib`: pipeline selecionado. Carregue somente arquivos locais/confiáveis: o formato não é seguro para objetos recebidos de terceiros.

O diretório de saída precisa estar vazio ou não existir. Isso evita sobrescrever e misturar resultados de execuções distintas. Em caso de erro, preserve o diagnóstico e use um novo diretório depois da correção. Arquivos parciais sem `metrics_report.json` com status `completed` não constituem uma execução concluída.

## Limitações

Divisão aleatória não mede generalização futura. A base não oferece ID de cartão/cliente; não é possível verificar isolamento por entidade. A PCA foi fornecida pronta, sem acesso ao seu ajuste original. As classes são raras, as métricas podem variar com a amostra e não há intervalo de confiança implementado. Nenhum resultado é transferível automaticamente para a Serasa Experian.

`average_precision` e `pr_auc_trapezoid` são campos distintos; não devem ser apresentados como métricas idênticas. Se um componente futuro precisar de um resultado ainda não executado, deve usar `[METRICA_APOS_EXECUCAO]` e preenchê-lo pelo JSON, sem estimar números.

## Referências do método

- [scikit-learn: evitar data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
- [scikit-learn: ajuste do limiar de classificação](https://scikit-learn.org/stable/modules/classification_threshold.html).
- [SHAP: TreeExplainer e unidades da saída](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).
