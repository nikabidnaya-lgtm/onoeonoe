from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.data_collector import DataCollector


@dataclass(slots=True)
class AnalyticsEngine:
    collector: DataCollector
    config: dict[str, Any]

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, pd.NA)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        plus_dm = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0)
        return dx.rolling(period).mean()

    def check_monday_conditions(self, symbol: str) -> dict[str, bool | float]:
        cfg = self.config
        funding = self.collector.get_funding_rates([symbol])[symbol]
        funding_ok = funding > cfg["monday"]["funding_threshold_pct"]
        volume_ratio = self.collector.get_volume_ratio(symbol, interval="60", sample_window=3)
        volume_ok = volume_ratio >= cfg["monday"]["volume_multiplier"]

        h1 = self.collector.get_ohlcv(symbol, interval="60", limit=4)
        price_open_10 = float(h1.iloc[-2]["open"])
        price_now = float(h1.iloc[-1]["close"])
        price_ok = price_now < price_open_10

        btc_d = self.collector.get_btc_dominance()
        btc_ok = btc_d >= cfg["monday"]["btc_dominance_threshold"]

        etf = self.collector.get_etf_flows().get("inflow", 0.0)
        etf_ok = etf <= -50_000_000

        return {
            "funding_ok": funding_ok,
            "funding_pct": round(funding, 6),
            "volume_ok": volume_ok,
            "volume_ratio": round(volume_ratio, 3),
            "price_ok": price_ok,
            "btc_dominance_ok": btc_ok,
            "btc_dominance": btc_d,
            "etf_ok": etf_ok,
            "etf_inflow": etf,
        }

    def check_sunday_breakout(self, symbol: str) -> dict[str, bool | float]:
        cfg = self.config["sunday"]
        h1 = self.collector.get_ohlcv(symbol, interval="60", limit=10)
        m15 = self.collector.get_ohlcv(symbol, interval="15", limit=40)

        level = float(h1.iloc[-cfg["breakout_period_hours"]:]["high"].max())
        current = float(m15.iloc[-1]["close"])
        breakout_ok = current > level

        volume_ratio = self.collector.get_volume_ratio(symbol, interval="60", sample_window=4)
        volume_ok = volume_ratio >= cfg["volume_multiplier"]

        rsi = float(self._rsi(h1["close"], period=cfg["rsi_period"]).iloc[-1])
        rsi_ok = rsi < cfg["rsi_threshold"]

        spread = self.collector.get_spread_pct(symbol)
        spread_ok = spread <= cfg["max_spread_pct"]

        return {
            "breakout_ok": breakout_ok,
            "volume_ok": volume_ok,
            "rsi_ok": rsi_ok,
            "spread_ok": spread_ok,
            "breakout_level": level,
            "price": current,
            "volume_ratio": round(volume_ratio, 3),
            "rsi": round(rsi, 2),
            "spread_pct": round(spread, 4),
        }

    def calculate_trend_strength(self, symbol: str) -> dict[str, str | float | bool]:
        thresholds = self.config["wednesday"]
        m5 = self.collector.get_ohlcv(symbol, interval="5", limit=120)
        h1 = self.collector.get_ohlcv(symbol, interval="60", limit=100)

        adx = float(self._adx(h1, period=thresholds["adx_period"]).iloc[-1])
        adx_ok = adx >= thresholds["adx_thresholds"][symbol]

        last = m5.tail(8)
        direction = "up" if last["close"].iloc[-1] > last["close"].iloc[0] else "down"
        extremes_ok = bool((last["high"].diff().tail(3) > 0).sum() >= 2) if direction == "up" else bool((last["low"].diff().tail(3) < 0).sum() >= 2)

        volume_ratio = self.collector.get_volume_ratio(symbol, interval="5", sample_window=24)
        volume_threshold = 2.0 if symbol == "SOLUSDT" else 1.8
        volume_ok = volume_ratio >= volume_threshold

        pre_news_price = float(m5.iloc[-30]["close"])
        current_price = float(m5.iloc[-1]["close"])
        move_pct = abs((current_price - pre_news_price) / pre_news_price) * 100
        move_ok = move_pct <= thresholds["max_move_pct"][symbol]

        valid = all([adx_ok, extremes_ok, volume_ok, move_ok])
        return {
            "direction": direction,
            "adx": round(adx, 2),
            "adx_ok": adx_ok,
            "extremes_ok": extremes_ok,
            "volume_ratio": round(volume_ratio, 3),
            "volume_ok": volume_ok,
            "move_pct": round(move_pct, 3),
            "move_ok": move_ok,
            "valid": valid,
        }
