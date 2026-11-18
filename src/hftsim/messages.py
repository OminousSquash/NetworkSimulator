from dataclasses import dataclass
from typing import Optional


@dataclass
class NewOrder:
    order_id: int
    side: str                 # "BUY" or "SELL"
    price: Optional[int]      # integer ticks; None for market orders
    size: int
    market: bool = False


@dataclass
class CancelOrder:
    order_id: int
