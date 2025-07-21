
DROP TABLE IF EXISTS {{schema}}.ohlcv_{{instrument}} CASCADE; 
CREATE TABLE {{schema}}.ohlcv_{{instrument}} (
    time TIMESTAMP WITHOUT TIME ZONE NOT NULL PRIMARY KEY,
    price_open NUMERIC(18,8) NOT NULL,
    price_high NUMERIC(18,8) NOT NULL,
    price_low NUMERIC(18,8) NOT NULL,
    price_close NUMERIC(18,8) NOT NULL,
    volume_traded NUMERIC(22,8) NOT NULL,
    trades_count BIGINT
);

SELECT create_hypertable('{{schema}}.ohlcv_{{instrument}}', 'time', chunk_time_interval => INTERVAL '12 hour');

DROP MATERIALIZED VIEW IF EXISTS {{schema}}.ohlcv_{{instrument}}_5m CASCADE;
CREATE MATERIALIZED VIEW {{schema}}.ohlcv_{{instrument}}_5m
WITH (timescaledb.continuous) AS
SELECT
  time_bucket('5 minute', time) AS time_,
  first(price_open, time) AS price_open,
  max(price_high) AS price_high,
  min(price_low) AS price_low,
  last(price_close, time) AS price_close,
  sum(volume_traded) AS volume_traded,
  SUM(trades_count) AS trades_count
FROM {{schema}}.ohlcv_{{instrument}}
GROUP BY time_;


DROP MATERIALIZED VIEW IF EXISTS {{schema}}.ohlcv_{{instrument}}_15m CASCADE;
CREATE MATERIALIZED VIEW {{schema}}.ohlcv_{{instrument}}_15m
WITH (timescaledb.continuous) AS
SELECT
  time_bucket('15 minute', time) AS time_,
  first(price_open, time) AS price_open,
  max(price_high) AS price_high,
  min(price_low) AS price_low,
  last(price_close, time) AS price_close,
  sum(volume_traded) AS volume_traded,
  SUM(trades_count) AS trades_count
FROM {{schema}}.ohlcv_{{instrument}}
GROUP BY time_;


DROP MATERIALIZED VIEW IF EXISTS  {{schema}}.ohlcv_{{instrument}}_1h CASCADE;
CREATE MATERIALIZED VIEW {{schema}}.ohlcv_{{instrument}}_1h
WITH (timescaledb.continuous) AS
SELECT
  time_bucket('1 hour', time) AS time_,
  first(price_open, time) AS price_open,
  max(price_high) AS price_high,
  min(price_low) AS price_low,
  last(price_close, time) AS price_close,
  sum(volume_traded) AS volume_traded,
  SUM(trades_count) AS trades_count
FROM {{schema}}.ohlcv_{{instrument}}
GROUP BY time_;

DROP MATERIALIZED VIEW IF EXISTS {{schema}}.ohlcv_{{instrument}}_4h CASCADE;
CREATE MATERIALIZED VIEW {{schema}}.ohlcv_{{instrument}}_4h
WITH (timescaledb.continuous) AS
SELECT
  time_bucket('4 hour', time) AS time_,
  first(price_open, time) AS price_open,
  max(price_high) AS price_high,
  min(price_low) AS price_low,
  last(price_close, time) AS price_close,
  sum(volume_traded) AS volume_traded,
  SUM(trades_count) AS trades_count
FROM {{schema}}.ohlcv_{{instrument}}
GROUP BY time_;

DROP MATERIALIZED VIEW IF EXISTS {{schema}}.ohlcv_{{instrument}}_1day CASCADE;
CREATE MATERIALIZED VIEW {{schema}}.ohlcv_{{instrument}}_1day
WITH (timescaledb.continuous) AS
SELECT
  time_bucket('1 day', time) AS time_,
  first(price_open, time) AS price_open,
  max(price_high) AS price_high,
  min(price_low) AS price_low,
  last(price_close, time) AS price_close,
  sum(volume_traded) AS volume_traded,
  SUM(trades_count) AS trades_count
FROM {{schema}}.ohlcv_{{instrument}}
GROUP BY time_;

DROP TABLE IF EXISTS {{schema}}.trades_{{instrument}} CASCADE; 
CREATE TABLE {{schema}}.trades_{{instrument}} (
        uuid TEXT PRIMARY KEY,
        exchange_ticker VARCHAR(255) DEFAULT '{{instrument}}',
        time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        price DECIMAL(18, 8) NOT NULL,
        size DECIMAL(18, 8) NOT NULL,
        taker_side VARCHAR(6) /*should be either 'buy' or 'sell' */
);