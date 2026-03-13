from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pybit.unified_trading import HTTP
from sqlalchemy.ext.asyncio import async_sessionmaker

from database import Trade


@dataclass(slots=True)
class PositionExecutor:
    session: HTTP
    db_session_factory: async_sessionmaker
    config: dict[str, Any]

    async def _log_trade(self, payload: dict[str, Any]) -> None:
        async with self.db_session_factory() as db:
            db.add(Trade(**payload))
            await db.commit()

    async def open_position(self, symbol: str, side: str, leverage: int, capital_share: float) -> str:
        price = float(self.session.get_tickers(category="linear", symbol=symbol)["result"]["list"][0]["lastPrice"])
        qty = max((capital_share * leverage) / price, 1)
        self.session.set_leverage(category="linear", symbol=symbol, buyLeverage=str(leverage), sellLeverage=str(leverage))
        order = self.session.place_order(
            category="linear",
            symbol=symbol,
            side="Buy" if side.lower() == "long" else "Sell",
            orderType="Market",
            qty=f"{qty:.0f}",
            reduceOnly=False,
        )
        order_id = order["result"]["orderId"]
        await self._log_trade(
            {
                "order_id": order_id,
                "timestamp": datetime.now(timezone.utc),
                "symbol": symbol,
                "side": side,
                "entry_price": price,
                "size": qty,
                "leverage": leverage,
                "sl_price": None,
                "tp1_price": None,
                "tp2_price": None,
                "exit_price": None,
                "pnl_usd": None,
                "exit_reason": None,
            }
        )
        return order_id

    def set_sl_tp(self, symbol: str, side: str, sl_pct: float, tp1_pct: float) -> None:
        mark = float(self.session.get_tickers(category="linear", symbol=symbol)["result"]["list"][0]["markPrice"])
        is_long = side.lower() == "long"
        sl = mark * (1 - sl_pct / 100) if is_long else mark * (1 + sl_pct / 100)
        tp = mark * (1 + tp1_pct / 100) if is_long else mark * (1 - tp1_pct / 100)
        self.session.set_trading_stop(
            category="linear",
            symbol=symbol,
            stopLoss=f"{sl:.6f}",
            takeProfit=f"{tp:.6f}",
            tpslMode="Full",
            positionIdx=0,
        )

    def close_all_positions(self) -> list[str]:
        ids: list[str] = []
        for symbol in self.config["trading"]["symbols"]:
            positions = self.session.get_positions(category="linear", symbol=symbol)["result"]["list"]
            for pos in positions:
                size = float(pos["size"])
                if size <= 0:
                    continue
                side = "Sell" if pos["side"] == "Buy" else "Buy"
                order = self.session.place_order(
                    category="linear", symbol=symbol, side=side, orderType="Market", qty=str(int(size)), reduceOnly=True
                )
                ids.append(order["result"]["orderId"])
        return ids
