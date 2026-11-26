from pathlib import Path

import numpy as np
import pytest

from hftsim.messages import NewOrder, CancelOrder
from hftsim.simulator_engine.simulator import Simulator

ROOT = Path(__file__).resolve().parent.parent


class RecordingExchange:
    def __init__(self, name, log, sim_ref=None):
        self.name, self.log, self.sim_ref = name, log, sim_ref

    def receive(self, msg):
        self.log.append((self.sim_ref[0].time if self.sim_ref else None, self.name, msg))


def order(i):
    return NewOrder(order_id=i, side="BUY", price=100, size=1)


def make(sources, venues=("A", "B"), horizon=100.0):
    log, ref = [], [None]
    exch = {v: RecordingExchange(v, log, ref) for v in venues}
    sim = Simulator([iter(s) for s in sources], exch, horizon)
    ref[0] = sim
    return sim, exch, log


def test_merges_sources_in_time_order():
    a = [(1.0, "A", order(1)), (3.0, "A", order(3)), (5.0, "A", order(5))]
    b = [(2.0, "B", order(2)), (4.0, "B", order(4))]
    sim, _, log = make([a, b])
    sim.run()
    assert [m.order_id for _, _, m in log] == [1, 2, 3, 4, 5]
    assert [t for t, _, _ in log] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_horizon_stops_delivery():
    a = [(t, "A", order(int(t))) for t in (1.0, 2.0, 10.0, 11.0)]
    sim, _, log = make([a], horizon=5.0)
    sim.run()
    assert [m.order_id for _, _, m in log] == [1, 2]
    assert sim.time == 2.0


def test_event_exactly_at_horizon_is_delivered():
    sim, _, log = make([[(5.0, "A", order(1))]], horizon=5.0)
    sim.run()
    assert len(log) == 1


def test_ties_are_fifo_and_messages_never_compared():
    # NewOrder vs CancelOrder can't be compared with <; seq must break the tie.
    a = [(1.0, "A", order(1))]
    b = [(1.0, "B", CancelOrder(order_id=2))]
    sim, _, log = make([a, b])
    sim.run()
    assert [v for _, v, _ in log] == ["A", "B"]


def test_empty_and_exhausted_sources():
    sim, _, log = make([[], [(1.0, "A", order(1))]])
    sim.run()
    assert len(log) == 1


def test_schedule_from_exchange_is_delivered_and_not_refilled():
    log = []

    class Echo:
        def __init__(self, sim_box):
            self.box = sim_box

        def receive(self, msg):
            sim = self.box[0]
            log.append((sim.time, msg.order_id))
            if msg.order_id == 1:  # schedule a reply 0.5s later
                sim.schedule(sim.time + 0.5, "A", order(99))

    box = [None]
    sim = Simulator([iter([(1.0, "A", order(1)), (2.0, "A", order(2))])],
                    {"A": Echo(box)}, horizon=10.0)
    box[0] = sim
    sim.run()
    assert log == [(1.0, 1), (1.5, 99), (2.0, 2)]


def test_schedule_in_past_raises():
    sim, _, _ = make([[]])
    sim.time = 5.0
    with pytest.raises(ValueError):
        sim.schedule(4.0, "A", order(1))


def test_unknown_venue_raises():
    sim, _, _ = make([[(1.0, "Z", order(1))]])
    with pytest.raises(KeyError):
        sim.run()


def test_non_monotonic_source_lets_clock_go_backwards():
    """Documents current behaviour: nothing guards against a source emitting
    out-of-order times, so the simulated clock can move backwards."""
    a = [(1.0, "A", order(1)), (5.0, "A", order(2)), (3.0, "A", order(3))]
    sim, _, log = make([a])
    sim.run()
    times = [t for t, _, _ in log]
    assert times == [1.0, 5.0, 3.0]  # clock went 5 -> 3


def test_with_calibrated_flow_two_venues_deterministic():
    from hftsim.data.synthetic import load_params, CalibratedFlow
    from hftsim.data.fundamental import FundamentalPrice

    params = load_params(ROOT / "params/mine.yaml")

    def run_once(seed):
        rng = np.random.default_rng(seed)
        r_mid, r_a, r_b = rng.spawn(3)
        fund = FundamentalPrice.from_params(params["mid"], params["start_mid_ticks"], r_mid)
        srcs = [CalibratedFlow(params, fund, r_a, "A").events(60.0),
                CalibratedFlow(params, fund, r_b, "B").events(60.0)]
        log, ref = [], [None]
        exch = {v: RecordingExchange(v, log, ref) for v in ("A", "B")}
        sim = Simulator(srcs, exch, horizon=30.0)
        ref[0] = sim
        sim.run()
        return log

    log1, log2 = run_once(42), run_once(42)
    times = [t for t, _, _ in log1]
    assert len(log1) > 100
    assert times == sorted(times)
    assert max(times) <= 30.0
    assert {v for _, v, _ in log1} == {"A", "B"}
    assert [(t, v, repr(m)) for t, v, m in log1] == [(t, v, repr(m)) for t, v, m in log2]
    print(f"\n{len(log1)} events in 30s, ~{len(log1)/60:.1f}/s per venue")
