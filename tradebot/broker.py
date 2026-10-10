"""Thin wrapper over Alpaca's *paper* trading API. Paper mode is hard-coded on purpose."""
from datetime import datetime

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest


def _pos_symbol(symbol: str) -> str:
    # Orders use "BTC/USD"; positions are keyed as "BTCUSD".
    return symbol.replace("/", "")


class PaperBroker:
    def __init__(self, api_key: str, secret_key: str):
        if not api_key or not secret_key:
            raise SystemExit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY (paper keys) in .env")
        self.client = TradingClient(api_key, secret_key, paper=True)

    def equity(self) -> float:
        return float(self.client.get_account().equity)

    def day_start_equity(self) -> float:
        """Equity at the previous daily close, as Alpaca records it. Survives bot restarts."""
        return float(self.client.get_account().last_equity)

    def buys_since(self, since: datetime) -> int:
        """Number of non-cancelled buy orders placed since `since`. Survives bot restarts."""
        orders = self.client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.ALL, side=OrderSide.BUY, after=since, limit=500)
        )
        return sum(o.status not in (OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED) for o in orders)

    def entry_price(self, symbol: str) -> float | None:
        """Average entry price of the open position, or None if flat."""
        try:
            pos = self.client.get_open_position(_pos_symbol(symbol))
        except APIError:
            return None
        return float(pos.avg_entry_price) if float(pos.qty) > 0 else None

    def has_open_order(self, symbol: str) -> bool:
        orders = self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]))
        return len(orders) > 0

    def clock(self):
        """Market clock: .is_open, .next_open, .next_close."""
        return self.client.get_clock()

    def buy(self, symbol: str, notional_usd: float):
        # Dollar-amount (fractional) stock orders must be DAY orders; crypto uses GTC.
        tif = TimeInForce.GTC if "/" in symbol else TimeInForce.DAY
        req = MarketOrderRequest(symbol=symbol, notional=round(notional_usd, 2), side=OrderSide.BUY, time_in_force=tif)
        return self.client.submit_order(req)

    def close(self, symbol: str):
        return self.client.close_position(_pos_symbol(symbol))

    def close_all(self):
        return self.client.close_all_positions(cancel_orders=True)
