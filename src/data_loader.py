import numpy as np
import pandas as pd
import yfinance as yf

from src.instruments import Instrument
from src.utils import normalize_option_type


def fetch_yahoo_history(yahoo_symbol: str, start_date: str) -> pd.DataFrame:
    df = yf.download(
        yahoo_symbol,
        start=start_date,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No Yahoo Finance data found for {yahoo_symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df = df.rename(columns={"Date": "date", "Datetime": "date", "Close": "close"})
    df = df[["date", "close"]].dropna()
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna()

    return df


def fetch_latest_yahoo_quote(yahoo_symbol: str):
    df = yf.download(
        yahoo_symbol,
        period="5d",
        interval="5m",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No latest quote found for {yahoo_symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].dropna()

    if close.empty:
        raise ValueError(f"No close price found for {yahoo_symbol}")

    price = float(close.iloc[-1])
    timestamp = pd.to_datetime(close.index[-1])

    return price, timestamp


def calculate_return_stats(price_df: pd.DataFrame, trading_days: int) -> dict:
    df = price_df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna()

    if df.empty:
        raise ValueError("Not enough data to calculate volatility.")

    return {
        "historical_vol": float(df["log_return"].std() * np.sqrt(trading_days)),
        "annualized_return": float(df["log_return"].mean() * trading_days),
        "skewness": float(df["log_return"].skew()),
        "kurtosis": float(df["log_return"].kurtosis()),
    }


def load_option_chain(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = [
        "instrument_id",
        "expiry",
        "strike",
        "option_type",
        "market_price",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in option chain CSV: {missing}")

    df["instrument_id"] = df["instrument_id"].astype(str).str.upper().str.strip()
    df["expiry"] = df["expiry"].astype(str).str.replace("-", "").str.replace("/", "")
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["option_type"] = df["option_type"].apply(normalize_option_type)
    df["market_price"] = pd.to_numeric(df["market_price"], errors="coerce")

    if "volume" not in df.columns:
        df["volume"] = np.nan

    if "open_interest" not in df.columns:
        df["open_interest"] = np.nan

    if "multiplier" not in df.columns:
        df["multiplier"] = 1

    df = df.dropna(subset=["strike", "market_price"])
    df = df[df["strike"] > 0]
    df = df[df["market_price"] > 0]

    return df


def select_atm_option(
    option_chain: pd.DataFrame,
    instrument: Instrument,
    spot_or_future: float,
    trade_date: pd.Timestamp,
):
    chain = option_chain[
        option_chain["instrument_id"] == instrument.id.upper()
    ].copy()

    if chain.empty:
        return None

    chain["expiry_dt"] = pd.to_datetime(
        chain["expiry"],
        format="%Y%m%d",
        errors="coerce",
    )

    chain = chain.dropna(subset=["expiry_dt"])
    chain = chain[chain["expiry_dt"] >= trade_date.normalize()]

    if chain.empty:
        return None

    chain = chain[chain["option_type"] == normalize_option_type(instrument.option_type)]

    if chain.empty:
        return None

    nearest_expiry = chain["expiry_dt"].min()
    chain = chain[chain["expiry_dt"] == nearest_expiry].copy()
    chain["atm_distance"] = (chain["strike"] - spot_or_future).abs()

    return chain.sort_values("atm_distance").iloc[0]
