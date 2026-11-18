"""Calibrate the synthetic order flow from LOBSTER message and order book files.

Usage:
    python -m hftsim.data.calibrate MESSAGE.csv BOOK.csv --levels 10 --out params.yaml

Outputs a YAML file that synthetic.CalibratedFlow and fundamental.FundamentalPrice read.
Prices are converted to integer ticks ($0.01) before any statistic is computed.
"""
import argparse

import numpy as np
import pandas as pd
import yaml

PRICE_UNITS_PER_TICK = 100    # LOBSTER prices are dollars x 10000; a $0.01 tick is 100 units
MAX_ABS_OFFSET = 200          # ticks; guards against stray values
LIMIT = 1                     # submission of a new limit order
CANCEL = (2, 3)               # partial cancel and full deletion
EXECUTION = (4, 5)            # visible and hidden executions (used as the market-order proxy)
HALT = 7

MSG_COLUMNS = ["time", "type", "order_id", "size", "price", "direction"]


def load(message_path, book_path, levels, start=None, end=None):
    """Read both files, drop halts, mark empty book levels as NaN, add mid prices in ticks,
    then restrict to [start, end] seconds after midnight if given.

    The window is applied after the mid is computed, so the first event in the window
    still has a correct prev_mid from the event before it.
    """
    msg = pd.read_csv(message_path, header=None, names=MSG_COLUMNS)

    book_cols = []
    for lvl in range(1, levels + 1):
        book_cols += [f"ask_p{lvl}", f"ask_s{lvl}", f"bid_p{lvl}", f"bid_s{lvl}"]
    book = pd.read_csv(book_path, header=None, names=book_cols)

    if len(msg) != len(book):
        raise ValueError(f"row mismatch: {len(msg)} messages vs {len(book)} book rows")

    keep = msg["type"] != HALT
    msg = msg[keep].reset_index(drop=True)
    book = book[keep].reset_index(drop=True)

    # LOBSTER fills empty levels with sentinel prices; replace them so they don't look like real prices
    for lvl in range(1, levels + 1):
        empty_ask = book[f"ask_p{lvl}"] > 9e9
        empty_bid = book[f"bid_p{lvl}"] < -9e9
        book.loc[empty_ask, [f"ask_p{lvl}", f"ask_s{lvl}"]] = np.nan
        book.loc[empty_bid, [f"bid_p{lvl}", f"bid_s{lvl}"]] = np.nan

    # Mid after each event, in ticks; prev_mid is the mid before the event, used for offsets
    msg["mid"] = (book["ask_p1"] + book["bid_p1"]) / 2 / PRICE_UNITS_PER_TICK
    msg["prev_mid"] = msg["mid"].shift(1)

    if start is not None or end is not None:
        lo = -np.inf if start is None else start
        hi = np.inf if end is None else end
        in_window = msg["time"].between(lo, hi)
        msg = msg[in_window].reset_index(drop=True)
    return msg


def calibrate(msg, window_seconds, source_name):
    """Compute the parameters the generator needs from a loaded message frame."""
    msg = msg.dropna(subset=["mid"])
    counts = msg["type"].value_counts()

    def rate(types):
        n = sum(int(counts.get(t, 0)) for t in np.atleast_1d(types))
        return float(n / window_seconds)

    rates = {
        "limit": rate(LIMIT),
        "cancel": rate(CANCEL),
        # Executions are per resting order, so this approximates the market-order rate.
        # A market order that sweeps several levels produces several rows.
        "market": rate(EXECUTION),
    }

    subs = msg[msg["type"] == LIMIT].dropna(subset=["prev_mid"])
    direction = subs["direction"].to_numpy()
    price_ticks = subs["price"].to_numpy() / PRICE_UNITS_PER_TICK
    # The mid can sit on a half tick, so measure from a whole-tick base instead.
    # The generator uses the same floor(mid) base, so offsets line up exactly.
    base = np.floor(subs["prev_mid"].to_numpy())

    # Distance from the base. Positive means passive, negative means marketable.
    offsets = np.where(direction == 1, base - price_ticks, price_ticks - base).astype(int)
    offsets = offsets[np.abs(offsets) <= MAX_ABS_OFFSET]
    off_vals, off_counts = np.unique(offsets, return_counts=True)

    sz_vals, sz_counts = np.unique(subs["size"].to_numpy().astype(int), return_counts=True)

    mids = msg["mid"].to_numpy()
    diffs = np.diff(mids)
    changes = diffs[diffs != 0]
    step_vals, step_counts = np.unique(np.round(changes * 2) / 2, return_counts=True)

    if off_counts.sum() == 0 or sz_counts.sum() == 0:
        raise ValueError("no limit submissions found; check the file and the window")

    return {
        "source": source_name,
        "window_seconds": float(window_seconds),
        "start_mid_ticks": float(mids[0]),
        "rates": rates,
        "buy_share": float((direction == 1).mean()),
        "offsets": {
            "values": off_vals.tolist(),
            "probs": (off_counts / off_counts.sum()).tolist(),
        },
        "sizes": {
            "values": sz_vals.tolist(),
            "probs": (sz_counts / sz_counts.sum()).tolist(),
        },
        "mid": {
            "rate": float(len(changes) / window_seconds),
            "step_values": step_vals.tolist(),
            "step_probs": ((step_counts / step_counts.sum()) if step_counts.sum() else step_counts).tolist(),
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Calibrate synthetic flow from LOBSTER files")
    ap.add_argument("message", help="LOBSTER message CSV")
    ap.add_argument("book", help="LOBSTER orderbook CSV")
    ap.add_argument("--levels", type=int, default=10, help="book levels in the file")
    ap.add_argument("--start", type=float, default=None,
                    help="window start, seconds after midnight (e.g. 36000 for 10:00)")
    ap.add_argument("--end", type=float, default=None,
                    help="window end, seconds after midnight (e.g. 55800 for 15:30)")
    ap.add_argument("--out", required=True, help="output YAML path")
    args = ap.parse_args()

    msg = load(args.message, args.book, args.levels, start=args.start, end=args.end)
    if args.start is not None and args.end is not None:
        window = args.end - args.start
    else:
        window = float(msg["time"].max() - msg["time"].min())
    params = calibrate(msg, window, source_name=args.message)

    with open(args.out, "w") as f:
        yaml.safe_dump(params, f, sort_keys=False)
    print(f"wrote {args.out}: rates={params['rates']} buy_share={params['buy_share']:.3f}")


if __name__ == "__main__":
    main()
