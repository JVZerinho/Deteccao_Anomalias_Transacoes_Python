# Detecção de Fraude em Cartão de Crédito com Machine Learning

Projeto desenvolvido durante o **Bootcamp Santander 2026 - Fundamentos de Dados (DIO)**, ministrado pela Expert Isadora Garcia Ferrão.  
**Autor:** João Victor Santos  

Aqui compartilho como estruturei o projeto, os testes que realizei e o que aprendi lidando com dados reais e extremamente desbalanceados.

---

## 1. O Problema e por que a Acurácia engana

O dataset contém **284.807 transações** de cartões europeus, mas apenas **492 são fraudes** (cerca de **0,17%**). Todo o resto (99,83%) são compras normais de clientes.

No início do estudo de Machine Learning, é comum olharmos direto para a **acurácia**, mas aqui ela é uma pegadinha:
> Se criarmos um modelo que simplesmente responda que *"nenhuma transação é fraude"*, ele vai acertar **99,83%** das vezes. Parece um resultado incrível, mas na prática esse modelo é inútil porque deixou passar **100% dos golpes**!

Por isso, aprendi que as métricas que realmente importam aqui são:
* **Recall (Sensibilidade):** De todas as fraudes reais, quantas o modelo conseguiu pegar? (Para um banco, deixar uma fraude escapar custa muito caro).
* **Precisão:** Das transações que o modelo apontou como fraude, quantas realmente eram? (Evita bloquear compras de clientes honestos à toa).
* **F1-Score e PR-AUC:** Ajudam a avaliar o equilíbrio entre pegar fraudes e não disparar alarmes falsos desnecessários.

---

## 2. Preparando os Dados

Seguindo as orientações da aula, apliquei os seguintes tratamentos:
1. **Escalonamento Robusto (`RobustScaler`):** As colunas `Time` (tempo) e `Amount` (valor da compra) tinham valores muito dispersos e com muitos outliers (compras de centavos até milhares de reais). Usei o `RobustScaler` porque ele utiliza a mediana e o IQR, não sendo distorcido por valores gigantes.
2. **Transformação Logarítmica (`log_amount`):** Apliquei `np.log1p(Amount)` para suavizar a cauda longa dos valores.
3. **Divisão Estratificada (`stratify=y`):** Na hora de separar treino (80%) e teste (20%), usei `stratify=y` para garantir que a proporção exata de 0,17% de fraudes ficasse igual nos dois grupos.

---

## 3. Testando e Comparando os Modelos

Testei três algoritmos usando técnicas de peso nas classes (`class_weight='balanced'` e `scale_pos_weight`) para obrigar os modelos a prestarem atenção nas fraudes:

| Modelo | Recall (Fraude) | Precisão (Fraude) | F1-Score | PR-AUC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Regressão Logística (Baseline)** | 91,84% | 5,69% | 0,1072 | 0,7245 | 0,9760 |
| **Random Forest** | 78,57% | 88,51% | 0,8324 | 0,8596 | 0,9488 |
| **XGBoost (Melhor resultado)** | **82,65%** | **88,04%** | **0,8526** | **0,8648** | **0,9807** |

* **O que notei:** A Regressão Logística pegou bastante fraude (91,84%), mas com precisão de só 5,69% (deu centenas de alarmes falsos, o que sobrecarregaria qualquer equipe). O **XGBoost** foi o mais equilibrado, pegando mais de 82% das fraudes com 88% de precisão e o melhor PR-AUC.

---

## 4. Ajustando o Limiar de Decisão (Threshold Tuning)

Por padrão, os modelos consideram fraude qualquer transação com probabilidade acima de 0.50. Mas como em fraude é melhor investigar uma suspeita a mais do que deixar um golpe passar, fiz testes variando esse limiar no XGBoost:

| Limiar de Corte | Fraudes Pegas (TP) | Fraudes Perdidas (FN) | Alarmes Falsos (FP) | Recall | Precisão | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.10 | 87 | 11 | 24 | 88,78% | 78,38% | 0,8325 |
| 0.20 | 86 | 12 | 14 | 87,76% | 86,00% | 0,8687 |
| **0.30 (Escolhido)** | **85** | **13** | **11** | **86,73%** | **88,54%** | **0,8763** |
| 0.50 (Padrão) | 81 | 17 | 11 | 82,65% | 88,04% | 0,8526 |
| 0.70 | 78 | 20 | 8 | 79,59% | 90,70% | 0,8478 |

**Minha escolha:** Baixei o limiar para **0.30**.
* O Recall subiu de **82,65% para 86,73%** (conseguimos interceptar mais 4 fraudes no conjunto de teste).
* A precisão continuou muito boa em **88,54%** (apenas 11 alarmes falsos em mais de 56 mil compras testadas).
* Alcançou o maior **F1-Score (0,8763)** de todos os testes.

---

## 5. O que o SHAP me ensinou sobre as decisões do modelo

Para não deixar o modelo como uma "caixa preta", usei a biblioteca **SHAP** (`TreeExplainer`) para entender quais variáveis mais pesavam na decisão:

1. **V14:** É a variável mais importante de todas. Quando o valor de V14 cai muito para números negativos, o modelo quase imediatamente sinaliza alto risco de fraude.
2. **V4:** Valores positivos altos de V4 também são fortes indicativos de fraude.
3. **V12 e V10:** Apresentam comportamento similar a V14, sendo essenciais para separar compras normais de golpes.
4. **Tempo e Valor:** Ajudam a complementar o contexto (compras suspeitas fora do padrão habitual).

---

## 6. O que fiz além do básico da aula

* Apliquei o `RobustScaler` junto com o `log_amount` para não sofrer com a distorção dos valores de compras extravagantes.
* Fiz o teste sistemático de limiar de probabilidade, explicando o impacto prático de cada valor.
* Avaliei o modelo usando **PR-AUC**, que reflete muito melhor a realidade de datasets desbalanceados do que a ROC-AUC.
* Utilizei o **SHAP** para enxergar de forma transparente o porquê de cada decisão do modelo.

---

## Como rodar o projeto

1. Instale as bibliotecas:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost shap
