import ccxt
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange
from datetime import datetime, timedelta

SYMBOL = "ETH/USDT"
TIMEFRAME = "1m"
HOURS = 24 * 7

exchange = ccxt.binance()

def fetch_data():
    since = exchange.parse8601(
        (datetime.utcnow() - timedelta(hours=HOURS)).isoformat()
    )

    all_ohlc = []

    while True:
        ohlc = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
        if not ohlc:
            break
        since = ohlc[-1][0] + 1
        all_ohlc += ohlc
        if len(ohlc) < 1000:
            break

    df = pd.DataFrame(all_ohlc, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    return df


def resample_3m(df):
    df_3m = df.resample("3min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })
    df_3m.dropna(inplace=True)
    return df_3m


def backtest(df):

    df["RSI"] = RSIIndicator(df["close"], window=10).rsi()
    df["ADX"] = ADXIndicator(df["high"], df["low"], df["close"], window=10).adx()
    df["ATR"] = AverageTrueRange(df["high"], df["low"], df["close"], window=5).average_true_range()

    capital = 10000
    position = None
    entry_price = 0
    stop = 0
    tp = 0
    extreme = 0
    trades = []

    for i in range(20, len(df)):
        row = df.iloc[i]

        if position is None:

            if row["ADX"] > 20 and row["RSI"] < 30:
                position = "long"
                entry_price = row["close"]
                stop = entry_price - 2 * row["ATR"]
                tp = entry_price + 2.5 * row["ATR"]
                extreme = entry_price

            elif row["ADX"] > 20 and row["RSI"] > 70:
                position = "short"
                entry_price = row["close"]
                stop = entry_price + 2 * row["ATR"]
                tp = entry_price - 2.5 * row["ATR"]
                extreme = entry_price

        else:

            if position == "long":
                extreme = max(extreme, row["high"])
                stop = extreme - 2 * row["ATR"]

                if row["low"] <= stop or row["high"] >= tp:
                    exit_price = stop if row["low"] <= stop else tp
                    profit = exit_price - entry_price
                    capital += profit
                    trades.append(profit)
                    position = None

            elif position == "short":
                extreme = min(extreme, row["low"])
                stop = extreme + 2 * row["ATR"]

                if row["high"] >= stop or row["low"] <= tp:
                    exit_price = stop if row["high"] >= stop else tp
                    profit = entry_price - exit_price
                    capital += profit
                    trades.append(profit)
                    position = None

    return capital, trades


if __name__ == "__main__":
    df = fetch_data()
    df = resample_3m(df)
    final_capital, trades = backtest(df)

    print("Capital final:", final_capital)
    print("Trades:", len(trades))

    if trades:
        winrate = sum(1 for t in trades if t > 0) / len(trades)
        print("Winrate:", winrate)
