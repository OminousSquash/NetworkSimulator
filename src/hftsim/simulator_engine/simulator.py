from typing import Iterator, Union, List
from hftsim.messages import NewOrder, CancelOrder
import heapq

Msg = Union[NewOrder, CancelOrder]
Event = tuple[float, str, Msg]

class Simulator:
    def __init__(self, sources: List[Iterator[Event]], exchanges: dict[str, object], horizon: float):
        self.sources = sources
        self.exchanges = exchanges
        self.horizon = horizon
        self.event_heap = []
        self.seq = 0
        self.time = 0.0


    def schedule(self, t: float, venue: str, msg: Msg) -> None:
        if t < self.time:
            raise ValueError(f"cannot schedule in the past: {t} < {self.time}")
        self._push(t, None, venue, msg)

    def _push(self, t: float, source_index, venue: str, msg) -> None:
        heapq.heappush(self.event_heap, (t, self.seq, source_index, venue, msg))
        self.seq += 1

    def _refill(self, i: int) -> None:
        """Pull the next event from source i and put it on the heap."""
        try:
            t, venue, msg = next(self.sources[i])
        except StopIteration:
            return  # this source has no more events
        self._push(t, i, venue, msg)

    def run(self) -> None:
        # Prime the heap with the first event from each source
        for i in range(len(self.sources)):
            self._refill(i)

        while self.event_heap:
            t, _, i, venue, msg = heapq.heappop(self.event_heap)
            if t > self.horizon:
                break  # heap is ordered, so everything left is later too

            self.time = t
            self.exchanges[venue].receive(msg)

            if i is not None:  # events from sources are refilled; exchange events are not
                self._refill(i)

