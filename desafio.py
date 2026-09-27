import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve
)
import shap

sns.set_theme(style='whitegrid')
print('Bibliotecas importadas com sucesso!')
print()

url = 'https://www.dropbox.com/s/b44o3t3ehmnx2b7/creditcard.csv?dl=1'
df = pd.read_csv(url)

print(f'Dimensões do dataset: {df.shape[0]:,} linhas e {df.shape[1]} colunas.')
print(f'Valores nulos: {df.isnull().sum().sum()}')
df.head()

# Medindo o desbalanceamento da variável alvo (Class)
contagem = df['Class'].value_counts()
prop_fraude = (contagem[1] / len(df)) * 100

print(f'Transações Normais (0): {contagem[0]:,} ({100 - prop_fraude:.3f}%)')
print(f'Transações Fraude   (1): {contagem[1]:,} ({prop_fraude:.3f}%)')
print()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Class', palette=['#2ecc71', '#e74c3c'])
plt.title('Distribuição das Classes (0 = Legítimo, 1 = Fraude)')
plt.yscale('log')
plt.ylabel('Quantidade (Escala Logarítmica)')
plt.show()

df_clean = df.copy()

scaler = RobustScaler()
df_clean['scaled_amount'] = scaler.fit_transform(df_clean[['Amount']])
df_clean['scaled_time'] = scaler.fit_transform(df_clean[['Time']])
df_clean['log_amount'] = np.log1p(df_clean['Amount'])

X = df_clean.drop(['Time', 'Amount', 'Class'], axis=1)
y = df_clean['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f'Treino: {X_train.shape[0]:,} transações | {y_train.sum()} fraudes ({y_train.mean()*100:.3f}%)')
print(f'Teste : {X_test.shape[0]:,} transações | {y_test.sum()} fraudes ({y_test.mean()*100:.3f}%)')
print()

# 1. Regressão Logística (Baseline)
model_lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model_lr.fit(X_train, y_train)

# 2. Random Forest
model_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=10, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)

# 3. XGBoost
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
model_xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
model_xgb.fit(X_train, y_train)

print('Treinamento dos 3 modelos concluído com sucesso!')
print()

modelos = {
    'Regressão Logística': model_lr,
    'Random Forest': model_rf,
    'XGBoost': model_xgb
}

tabela_comparativa = []

for nome, modelo in modelos.items():
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]
    
    tabela_comparativa.append({
        'Modelo': nome,
        'Recall (Fraude)': recall_score(y_test, y_pred),
        'Precisão (Fraude)': precision_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'PR-AUC (Avg Precision)': average_precision_score(y_test, y_prob)
    })

df_comp = pd.DataFrame(tabela_comparativa)
print(df_comp.to_string(index=False))
print()

# Visualização das Matrizes de Confusão
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (nome, modelo) in zip(axes, modelos.items()):
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(f'Matriz de Confusão - {nome}')
    ax.set_xlabel('Classe Predita')
    ax.set_ylabel('Classe Real')
    ax.set_xticklabels(['Normal', 'Fraude'])
    ax.set_yticklabels(['Normal', 'Fraude'])

plt.tight_layout()
plt.show()


# Recall
y_prob_xgb = model_xgb.predict_proba(X_test)[:, 1]
limiares = np.arange(0.1, 1.0, 0.1)

tabela_th = []
for th in limiares:
    y_th = (y_prob_xgb >= th).astype(int)
    tabela_th.append({
        'Limiar': round(th, 2),
        'Recall': recall_score(y_test, y_th),
        'Precisão': precision_score(y_test, y_th),
        'F1-Score': f1_score(y_test, y_th)
    })

df_th = pd.DataFrame(tabela_th)
print(df_th.to_string(index=False))
print()

# Curva Precision-Recall vs Threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_xgb)
plt.figure(figsize=(8, 4))
plt.plot(thresholds, precisions[:-1], 'b--', label='Precisão')
plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
plt.xlabel('Limiar de Decisão (Threshold)')
plt.ylabel('Score')
plt.title('Trade-off entre Precisão e Recall pelo Limiar (XGBoost)')
plt.legend()
plt.show()

# Explicando com SHAP
explainer = shap.TreeExplainer(model_xgb)
X_sample = X_test.sample(500, random_state=42)
shap_values = explainer(X_sample)

plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_values, max_display=12, show=False)
plt.title('Impacto das Features na Detecção de Fraude (SHAP Beeswarm)', fontsize=14)
plt.tight_layout()
plt.show()
