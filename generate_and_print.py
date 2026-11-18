import numpy as np
from hftsim.data.synthetic import load_params, CalibratedFlow
from hftsim.data.fundamental import FundamentalPrice

p = load_params("params/mine.yaml")
rng = np.random.default_rng(42)
fund_rng, flow_rng = rng.spawn(2)
fund = FundamentalPrice.from_params(p["mid"], p["start_mid_ticks"], fund_rng)
flow = CalibratedFlow(p, fund, flow_rng, venue="A")

for t, venue, msg in flow.events(horizon=10.0):
    print(f"{t:8.3f}  {venue}  {msg}")