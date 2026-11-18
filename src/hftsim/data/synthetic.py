"""Synthetic order flow for one venue, driven by calibrated parameters.

The generator emits (time, venue, message) tuples in the same form as replay.py,
so DataFeed can schedule either one. It doesn't match orders or read the book:
the exchange does that. Its only input from the market is the shared mid price.
"""
import math

import numpy as np
import yaml

from hftsim.messages import CancelOrder, NewOrder


def load_params(path):
    with open(path) as f:
        return yaml.safe_load(f)


class CalibratedFlow:
    def __init__(self, params, fundamental, rng, venue):
        self.p = params
        self.fund = fundamental
        self.rng = rng
        self.venue = venue

        self.kinds = ["limit", "cancel", "market"]
        rates = params["rates"]
        self.total_rate = sum(rates[k] for k in self.kinds)
        if self.total_rate <= 0:
            raise ValueError("calibrated total event rate is zero")
        self.kind_probs = [rates[k] / self.total_rate for k in self.kinds]

        self.offset_vals = np.asarray(params["offsets"]["values"])
        self.offset_probs = np.asarray(params["offsets"]["probs"])
        self.size_vals = np.asarray(params["sizes"]["values"])
        self.size_probs = np.asarray(params["sizes"]["probs"])
        self.buy_share = params["buy_share"]

        self.own_ids = []       # live orders this generator submitted, for cancels
        self._last_id = 0

    def _new_id(self):
        self._last_id += 1
        return self._last_id

    def _draw_size(self):
        return int(self.rng.choice(self.size_vals, p=self.size_probs))

    def _draw_side(self):
        return "BUY" if self.rng.random() < self.buy_share else "SELL"

    def events(self, horizon):
        t = 0.0
        while True:
            t += self.rng.exponential(1.0 / self.total_rate)
            if t >= horizon:
                return

            mid = self.fund.advance(t)
            kind = self.rng.choice(self.kinds, p=self.kind_probs)

            if kind == "limit":
                side = self._draw_side()
                offset = int(self.rng.choice(self.offset_vals, p=self.offset_probs))
                base = math.floor(mid)          # same base as calibrate.py
                price = base - offset if side == "BUY" else base + offset
                oid = self._new_id()
                self.own_ids.append(oid)
                yield t, self.venue, NewOrder(oid, side, int(price), self._draw_size())

            elif kind == "cancel":
                if not self.own_ids:
                    continue
                i = int(self.rng.integers(len(self.own_ids)))
                oid = self.own_ids.pop(i)
                yield t, self.venue, CancelOrder(oid)

            else:  # market order
                yield t, self.venue, NewOrder(self._new_id(), self._draw_side(), None,
                                              self._draw_size(), market=True)
