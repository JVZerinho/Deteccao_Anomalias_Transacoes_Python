-- SQL comum a PostgreSQL e SQLite >= 3.25.
-- Carga: python sql/carregar_e_consultar.py --engine sqlite --output-dir outputs
-- O arquivo e dividido em secoes pelo carregador; cada SELECT pode ser usado sozinho.
-- Tempo e RELATIVO ao inicio da base, nao horario civil. Nao ha ID de cliente/cartao.

-- @section schema
CREATE TABLE IF NOT EXISTS transacoes (
    transaction_id BIGINT PRIMARY KEY,
    split TEXT NOT NULL CHECK (split IN ('validation', 'test')),
    time_seconds BIGINT NOT NULL CHECK (time_seconds >= 0),
    amount DOUBLE PRECISION NOT NULL CHECK (amount >= 0),
    is_fraud INTEGER NOT NULL CHECK (is_fraud IN (0, 1))
);
CREATE TABLE IF NOT EXISTS scores_modelo (
    transaction_id BIGINT PRIMARY KEY REFERENCES transacoes(transaction_id),
    fraud_probability DOUBLE PRECISION NOT NULL CHECK (fraud_probability >= 0 AND fraud_probability <= 1),
    score DOUBLE PRECISION NOT NULL CHECK (score >= 0 AND score <= 1000),
    risk_decile INTEGER NOT NULL CHECK (risk_decile BETWEEN 1 AND 10),
    alert INTEGER NOT NULL CHECK (alert IN (0, 1))
);
CREATE TABLE IF NOT EXISTS parametros_analise (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    threshold DOUBLE PRECISION NOT NULL,
    high_value_cutoff DOUBLE PRECISION NOT NULL CHECK (high_value_cutoff >= 0),
    review_cost DOUBLE PRECISION NOT NULL CHECK (review_cost >= 0),
    missed_fraud_cost DOUBLE PRECISION NOT NULL CHECK (missed_fraud_cost >= 0)
);
CREATE INDEX IF NOT EXISTS ix_transacoes_split_time ON transacoes(split, time_seconds, transaction_id);

-- @section q1_faixa_valor_hora
-- Taxa de fraude e valor observado das transacoes fraudulentas.
-- O valor de uma compra fraudulenta NAO e perda realizada.
-- "perda_simulada_nao_detectada" usa custo unitario hipotetico e politica congelada.
WITH faixas AS (
    SELECT t.*, s.alert,
           CAST(t.time_seconds / 3600 AS INTEGER) AS relative_hour,
           CASE WHEN t.amount < 10 THEN '01: abaixo de 10'
                WHEN t.amount < 100 THEN '02: de 10 a menos de 100'
                WHEN t.amount < 1000 THEN '03: de 100 a menos de 1000'
                ELSE '04: 1000 ou mais' END AS amount_band
    FROM transacoes t JOIN scores_modelo s ON s.transaction_id = t.transaction_id
)
SELECT f.split, f.relative_hour, f.amount_band, COUNT(*) AS transactions,
       SUM(f.is_fraud) AS frauds, 1.0 * SUM(f.is_fraud) / COUNT(*) AS fraud_rate,
       SUM(CASE WHEN f.is_fraud = 1 THEN f.amount ELSE 0 END) AS fraud_amount_observed,
       SUM(CASE WHEN f.is_fraud = 1 AND f.alert = 0 THEN p.missed_fraud_cost ELSE 0 END) AS simulated_missed_fraud_loss,
       SUM(CASE WHEN f.alert = 1 THEN p.review_cost ELSE 0 END) AS simulated_review_cost
FROM faixas f CROSS JOIN parametros_analise p
GROUP BY f.split, f.relative_hour, f.amount_band
ORDER BY f.split, f.relative_hour, f.amount_band;

-- @section q2_janelas
-- Media das ultimas 10 observacoes, nao de 10 minutos.
-- PARTITION BY split impede misturar validacao e teste.
-- As janelas sao GLOBAIS por conjunto; nao representam comportamento de cliente.
SELECT t.transaction_id, t.split, t.time_seconds, t.amount, t.is_fraud, s.alert,
       AVG(t.amount) OVER (
           PARTITION BY t.split ORDER BY t.time_seconds, t.transaction_id
           ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
       ) AS moving_average_10_transactions,
       t.time_seconds - LAG(t.time_seconds) OVER (
           PARTITION BY t.split ORDER BY t.time_seconds, t.transaction_id
       ) AS seconds_since_previous_transaction,
       SUM(s.alert) OVER (
           PARTITION BY t.split ORDER BY t.time_seconds, t.transaction_id
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ) AS cumulative_alerts
FROM transacoes t JOIN scores_modelo s ON s.transaction_id = t.transaction_id
ORDER BY t.split, t.time_seconds, t.transaction_id;

-- @section q3_falsos_positivos_alto_valor
-- CTE + JOIN: auditar alertas falsos de alto valor sob a politica original.
-- Limite de alto valor e parametro de analise, nao regra de uma instituicao.
WITH falsos_positivos AS (
    SELECT t.transaction_id, t.split, t.time_seconds, t.amount,
           s.fraud_probability, s.score, s.risk_decile
    FROM transacoes t JOIN scores_modelo s ON s.transaction_id = t.transaction_id
    CROSS JOIN parametros_analise p
    WHERE t.is_fraud = 0 AND s.alert = 1 AND t.amount >= p.high_value_cutoff
), totais AS (
    SELECT split, COUNT(*) AS total_high_value_false_positives,
           SUM(amount) AS high_value_flagged_amount
    FROM falsos_positivos GROUP BY split
)
SELECT f.*, a.total_high_value_false_positives, a.high_value_flagged_amount
FROM falsos_positivos f JOIN totais a ON a.split = f.split
ORDER BY f.split, f.amount DESC, f.transaction_id;

-- @section q4_coorte_temporal
-- Coorte por dia relativo de transacao (intervalos de 86400 segundos).
-- Nao e safra de originacao de credito nem analise de maturacao/inadimplencia.
WITH coortes AS (
    SELECT split, CAST(time_seconds / 86400 AS INTEGER) AS relative_day,
           COUNT(*) AS transactions, SUM(is_fraud) AS frauds,
           SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) AS fraud_amount_observed
    FROM transacoes
    GROUP BY split, CAST(time_seconds / 86400 AS INTEGER)
)
SELECT split, relative_day, transactions, frauds,
       1.0 * frauds / transactions AS fraud_rate,
       fraud_amount_observed,
       SUM(frauds) OVER (
           PARTITION BY split ORDER BY relative_day
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ) AS cumulative_frauds
FROM coortes ORDER BY split, relative_day;
