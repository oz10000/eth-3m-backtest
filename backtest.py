import sys
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
from itertools import product

# ==================== CONFIG ====================

SYMBOL = "ETHUSDT"
INTERVAL_BASE = "1m"
HOURS = 24
LIMIT = 1000
REQUEST_TIMEOUT = 10

SLIPPAGE = 0.001
COMMISSION = 0.001
BASE_CAPITAL = 1000

PERIOD_RANGE = [2,4,6,8,10,12,14,16]
ADX_TH_RANGE = [20,25,30]
RSI_LOW_RANGE = [20,25,30,35,40]
RSI_HIGH_RANGE = [60,65,70,75,80]
MULT_STOP_RANGE = [1.0,1.5,2.0,2.5,3.0]
MULT_TP_RANGE = [1.0,1.5,2.0,2.5,3.0]
USE_SLOPE_OPTIONS = [False,True]

TIMEFRAMES = {
    "1m":"1min",
    "3m":"3min",
    "5m":"5min"
}

# ==================== FETCH BINANCE (SIN CLOUDFRONT) ====================

def fetch_klines_binance(symbol, interval, hours):
    endpoints = [
        "https://api.binance.com/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://api2.binance.com/api/v3/klines",
        "https://api3.binance.com/api/v3/klines"
    ]

    end_time = int(time.time() * 1000)
    start_time = end_time - hours * 60 * 60 * 1000

    for endpoint in endpoints:
        try:
            print(f"Intentando Binance: {endpoint}")
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_time,
                "limit": LIMIT
            }

            resp = requests.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            if data:
                return process_klines(data)

        except Exception as e:
            print("Error:", str(e)[:60])

    return None


def fetch_klines_bybit(symbol, interval, hours):
    try:
        print("Intentando Bybit...")
        url = "https://api.bybit.com/v5/market/kline"

        interval_map = {"1m":"1","3m":"3","5m":"5"}
        end_time = int(time.time())
        start_time = end_time - hours*60*60

        params = {
            "category":"spot",
            "symbol":symbol,
            "interval":interval_map.get(interval,"1"),
            "start":start_time*1000,
            "end":end_time*1000,
            "limit":LIMIT
        }

        resp = requests.get(url,params=params,timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if data["retCode"] == 0:
            klines = data["result"]["list"]
            klines.reverse()

            formatted = []
            for k in klines:
                formatted.append([
                    int(k[0]),
                    float(k[1]),
                    float(k[2]),
                    float(k[3]),
                    float(k[4]),
                    float(k[5]),
                    0,0,0,0,0,0
                ])

            return process_klines(formatted)

    except Exception as e:
        print("Bybit error:", str(e)[:60])

    return None


def process_klines(klines):
    cols = [
        "timestamp","open","high","low","close","volume",
        "c1","c2","c3","c4","c5","c6"
    ]

    df = pd.DataFrame(klines,columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"],unit="ms")
    df.set_index("timestamp",inplace=True)

    for c in ["open","high","low","close"]:
        df[c] = df[c].astype(float)

    return df[["open","high","low","close","volume"]]


def fetch_with_fallback():
    df = fetch_klines_binance(SYMBOL,INTERVAL_BASE,HOURS)
    if df is not None and not df.empty:
        return df

    df = fetch_klines_bybit(SYMBOL,INTERVAL_BASE,HOURS)
    return df


# ==================== INDICADORES ====================

def compute_rsi(series,period):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain/loss
    return 100 - (100/(1+rs))


def compute_atr(df,period):
    high,low,close = df["high"],df["low"],df["close"]
    tr = pd.concat([
        high-low,
        (high-close.shift()).abs(),
        (low-close.shift()).abs()
    ],axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ==================== MAIN ====================

def main():
    print("Iniciando optimización ETHUSDT...")

    df = fetch_with_fallback()

    if df is None or df.empty:
        print("No se pudieron descargar datos.")
        return

    print("Velas descargadas:",len(df))

    # Ejemplo simple de cálculo
    df["RSI"] = compute_rsi(df["close"],10)
    df["ATR"] = compute_atr(df,10)

    print("Último RSI:",df["RSI"].iloc[-1])
    print("Último ATR:",df["ATR"].iloc[-1])

    print("Proceso completado correctamente.")


if __name__ == "__main__":
    main()
