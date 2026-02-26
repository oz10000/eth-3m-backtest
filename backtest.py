import sys
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
from itertools import product

# ==================== CONFIGURACIÓN ====================

SYMBOL = 'ETHUSDT'
INTERVAL_BASE = '1m'
HOURS = 24
LIMIT = 1000
REQUEST_TIMEOUT = 10

SLIPPAGE = 0.001
COMMISSION = 0.001
BASE_CAPITAL = 1000
MAX_LEVERAGE = 100
MIN_WIN_RATE_FOR_LEVERAGE = 0.4

PERIOD_RANGE = [2,4,6,8,10,12,14,16]
ADX_TH_RANGE = [20,25,30]
RSI_LOW_RANGE = [20,25,30,35,40]
RSI_HIGH_RANGE = [60,65,70,75,80]
MULT_STOP_RANGE = [1.0,1.5,2.0,2.5,3.0]
MULT_TP_RANGE = [1.0,1.5,2.0,2.5,3.0]
USE_SLOPE_OPTIONS = [False, True]

TIMEFRAMES = {
    '1m': '1min',
    '3m': '3min',
    '5m': '5min'
}

# ==================== DESCARGA BINANCE ====================

def fetch_binance(symbol, interval, hours):
    print("Descargando desde Binance...")
    url = "https://api.binance.com/api/v3/klines"

    end = int(time.time() * 1000)
    start = end - hours * 60 * 60 * 1000

    all_data = []
    current = start

    while current < end:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current,
            "limit": LIMIT
        }

        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        if not data:
            break

        all_data.extend(data)
        current = data[-1][0] + 1

        if len(data) < LIMIT:
            break

    if not all_data:
        return None

    df = pd.DataFrame(all_data, columns=[
        "timestamp","open","high","low","close","volume",
        "c1","c2","c3","c4","c5","c6"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)

    return df[["open","high","low","close","volume"]]

# ==================== INDICADORES ====================

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_atr(df, period):
    high = df['high']
    low = df['low']
    close = df['close']

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()

def compute_adx(df, period):
    high = df['high']
    low = df['low']
    close = df['close']

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    up = high.diff()
    down = -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)

    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(period).mean()

    return adx

def resample_ohlc(df, rule):
    return df.resample(rule).agg({
        'open':'first',
        'high':'max',
        'low':'min',
        'close':'last',
        'volume':'sum'
    }).dropna()

# ==================== BACKTEST ====================

def backtest_direction(df, direction, params):

    df = df.copy()
    df["RSI"] = compute_rsi(df["close"], params["rsi_period"])
    df["ATR"] = compute_atr(df, params["atr_period"])
    df["ADX"] = compute_adx(df, params["adx_period"])

    position = None
    entry_price = 0
    profits = []

    for i in range(1, len(df)):
        row = df.iloc[i]

        if pd.isna(row["RSI"]) or pd.isna(row["ATR"]) or pd.isna(row["ADX"]):
            continue

        if position is None:

            if direction == "long":
                if row["ADX"] > params["adx_th"] and row["RSI"] < params["rsi_th"]:
                    position = "long"
                    entry_price = row["close"] * (1 + SLIPPAGE)

            else:
                if row["ADX"] > params["adx_th"] and row["RSI"] > params["rsi_th"]:
                    position = "short"
                    entry_price = row["close"] * (1 - SLIPPAGE)

        else:
            atr = row["ATR"]

            if direction == "long":
                tp = entry_price + params["mult_tp"] * atr
                sl = entry_price - params["mult_stop"] * atr

                if row["low"] <= sl or row["high"] >= tp:
                    exit_price = sl if row["low"] <= sl else tp
                    ret = (exit_price - entry_price) / entry_price - COMMISSION
                    profits.append(ret)
                    position = None

            else:
                tp = entry_price - params["mult_tp"] * atr
                sl = entry_price + params["mult_stop"] * atr

                if row["high"] >= sl or row["low"] <= tp:
                    exit_price = sl if row["high"] >= sl else tp
                    ret = (entry_price - exit_price) / entry_price - COMMISSION
                    profits.append(ret)
                    position = None

    if profits:
        win_rate = sum(1 for p in profits if p > 0) / len(profits)
        total_profit = sum(profits)
    else:
        win_rate = 0
        total_profit = 0

    return total_profit, win_rate, len(profits)

# ==================== MAIN ====================

def main():
    print("Iniciando optimización...")

    df_1m = fetch_binance(SYMBOL, INTERVAL_BASE, HOURS)

    if df_1m is None:
        print("No se pudieron descargar datos.")
        return

    print("Velas 1m:", len(df_1m))

    df = resample_ohlc(df_1m, "3min")

    best_profit = -999
    best_params = None

    combinations = list(product(
        PERIOD_RANGE,
        PERIOD_RANGE,
        PERIOD_RANGE,
        ADX_TH_RANGE,
        RSI_LOW_RANGE,
        MULT_STOP_RANGE,
        MULT_TP_RANGE
    ))

    print("Total combinaciones:", len(combinations))

    for adx_p, rsi_p, atr_p, adx_th, rsi_th, mult_stop, mult_tp in combinations:

        params = {
            "adx_period": adx_p,
            "rsi_period": rsi_p,
            "atr_period": atr_p,
            "adx_th": adx_th,
            "rsi_th": rsi_th,
            "mult_stop": mult_stop,
            "mult_tp": mult_tp
        }

        profit, win_rate, trades = backtest_direction(df, "long", params)

        if profit > best_profit:
            best_profit = profit
            best_params = params

    print("\nMejor configuración encontrada:")
    print(best_params)
    print("Profit:", best_profit)


if __name__ == "__main__":
    main()
