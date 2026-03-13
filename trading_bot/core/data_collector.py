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
        out: dict[str, float] = {}
        for symbol in symbols:
            resp = self.session.get_tickers(category="linear", symbol=symbol)
            out[symbol] = float(resp["result"]["list"][0]["lastPrice"])
        return out

    def get_funding_rates(self, symbols: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for symbol in symbols:
            resp = self.session.get_tickers(category="linear", symbol=symbol)
            out[symbol] = float(resp["result"]["list"][0]["fundingRate"]) * 100
        return out

    def get_ohlcv(self, symbol: str, interval: str = "60", limit: int = 200) -> pd.DataFrame:
        resp = self.session.get_kline(category="linear", symbol=symbol, interval=interval, limit=limit)
        rows = resp["result"]["list"]
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
        return df.sort_values("ts").reset_index(drop=True)

    def get_volume_ratio(self, symbol: str, interval: str = "60", sample_window: int = 3) -> float:
        df = self.get_ohlcv(symbol, interval=interval, limit=max(sample_window + 1, 10))
        if len(df) < sample_window + 1:
            return 0.0
        latest = float(df.iloc[-1]["volume"])
        baseline = mean(df.iloc[-(sample_window + 1):-1]["volume"].tolist())
        return latest / baseline if baseline > 0 else 0.0

    def get_spread_pct(self, symbol: str) -> float:
        resp = self.session.get_orderbook(category="linear", symbol=symbol, limit=1)
        bid = float(resp["result"]["b"][0][0])
        ask = float(resp["result"]["a"][0][0])
        mid = (bid + ask) / 2
        return ((ask - bid) / mid) * 100

    def get_btc_dominance(self) -> float:
        # TODO: подключить внешний API доминации BTC (CoinGecko/TradingView/Glassnode)
        return 0.0

    def get_etf_flows(self) -> dict[str, float]:
        # TODO: подключить поставщика ETF-потоков
        return {"inflow": 0.0}
