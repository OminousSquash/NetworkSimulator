import numpy as np


class FundamentalPrice:
    """Shared mid-price in ticks, used by every venue.

    Both venues call advance(t) before reading the mid, so the mid steps once per
    scheduled move no matter how many generators ask for it. Real venues share a
    fundamental price, so the arbitrage gaps come from order flow and link delays,
    not from two independent random walks.
    """

    def __init__(self, start_mid, step_rate, step_values, step_probs, rng):
        self.mid = float(start_mid)
        self.step_rate = float(step_rate)
        self.step_values = np.asarray(step_values, dtype=float)
        self.step_probs = np.asarray(step_probs, dtype=float)
        self.rng = rng
        self._next_step = self._draw_gap() if self.step_rate > 0 else np.inf

    def _draw_gap(self):
        return self.rng.exponential(1.0 / self.step_rate)

    def advance(self, t):
        while self._next_step <= t:
            self.mid += self.rng.choice(self.step_values, p=self.step_probs)
            self._next_step += self._draw_gap()
        return self.mid

    @classmethod
    def from_params(cls, mid_params, start_mid, rng):
        return cls(start_mid=start_mid,
                   step_rate=mid_params["rate"],
                   step_values=mid_params["step_values"],
                   step_probs=mid_params["step_probs"],
                   rng=rng)
