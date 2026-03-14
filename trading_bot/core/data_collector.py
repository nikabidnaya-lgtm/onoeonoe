from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

import pandas as pd
from pybit.unified_trading import HTTP


@dataclass(slots=True)
class DataCollector:
    session: HTTP

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "DataCollector":
        exchange = cfg["exchange"]
        session = HTTP(
            testnet=bool(exchange.get("testnet", True)),
            api_key=exchange.get("api_key"),
            api_secret=exchange.get("api_secret"),
        )
        return cls(session=session)

    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        return {
            symbol: float(self.session.get_tickers(category="linear", symbol=symbol)["result"]["list"][0]["lastPrice"])
            for symbol in symbols
        }

    def get_funding_rates(self, symbols: list[str]) -> dict[str, float]:
        return {
            symbol: float(self.session.get_tickers(category="linear", symbol=symbol)["result"]["list"][0]["fundingRate"]) * 100
            for symbol in symbols
        }

    def get_ohlcv(self, symbol: str, interval: str = "60", limit: int = 200) -> pd.DataFrame:
        resp = self.session.get_kline(category="linear", symbol=symbol, interval=interval, limit=limit)
        rows = resp["result"]["list"]
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
        return df.sort_values("ts").reset_index(drop=True)

    def get_volume_ratio(self, symbol: str, interval: str, sample_window: int, baseline_window: int | None = None) -> float:
        baseline_window = baseline_window or sample_window
        df = self.get_ohlcv(symbol, interval=interval, limit=max(sample_window + baseline_window + 2, 30))
        if len(df) < (sample_window + baseline_window):
            return 0.0
        latest = float(df.tail(sample_window)["volume"].mean())
        baseline = float(df.iloc[-(sample_window + baseline_window):-sample_window]["volume"].mean())
        return latest / baseline if baseline > 0 else 0.0

    def get_spread_pct(self, symbol: str) -> float:
        resp = self.session.get_orderbook(category="linear", symbol=symbol, limit=1)
        bid = float(resp["result"]["b"][0][0])
        ask = float(resp["result"]["a"][0][0])
        mid = (bid + ask) / 2
        return ((ask - bid) / mid) * 100

    def get_top_book(self, symbol: str) -> tuple[float, float]:
        resp = self.session.get_orderbook(category="linear", symbol=symbol, limit=1)
        bid = float(resp["result"]["b"][0][0])
        ask = float(resp["result"]["a"][0][0])
        return bid, ask

    def get_avg_daily_notional_7d(self, symbol: str) -> float:
        df = self.get_ohlcv(symbol, interval="D", limit=8)
        if df.empty:
            return 0.0
        return float(mean(df.tail(7)["turnover"].tolist()))

    def get_btc_dominance(self) -> float:
        return 0.0

    def get_etf_flows(self) -> dict[str, float]:
        return {"inflow": 0.0}

    def get_economic_calendar(self) -> list[dict[str, Any]]:
        # TODO: интеграция с Investing/ForexFactory parser.
        return []
