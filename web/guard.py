"""Public-demo protection: answer cache, per-visitor rate limit, daily cap. Thread-safe, in-memory."""
import json
import re
import threading
import time
from collections import OrderedDict, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


def normalize(question: str) -> str:
    """Cache key: case, spacing and trailing punctuation don't make a different question."""
    return re.sub(r"\s+", " ", question.lower()).strip().rstrip("?.! ")


class AnswerCache:
    """Question -> the recorded trace events and final answer, so a repeat can be replayed."""

    def __init__(self, max_entries: int = 500):
        self._items: OrderedDict[str, dict] = OrderedDict()
        self._max = max_entries
        self._lock = threading.Lock()

    def get(self, question: str) -> dict | None:
        with self._lock:
            return self._items.get(normalize(question))

    def put(self, question: str, events: list[dict], answer: dict):
        with self._lock:
            self._items[normalize(question)] = {"events": events, "answer": answer}
            while len(self._items) > self._max:
                self._items.popitem(last=False)  # drop the oldest

    def load(self, path: Path) -> int:
        """Seed from a JSON file of {question: {events, answer}}. Missing file is fine."""
        if not path.exists():
            return 0
        data = json.loads(path.read_text())
        for q, entry in data.items():
            self.put(q, entry["events"], entry["answer"])
        return len(data)


class RateLimiter:
    """At most `limit` events per `window` seconds per key (sliding window)."""

    def __init__(self, limit: int, window: float = 3600, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, float]:
        """Record an attempt. Returns (allowed, seconds until the next slot frees up)."""
        now = self.clock()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False, self.window - (now - q[0])
            q.append(now)
            return True, 0.0


class DailyCap:
    """At most `limit` uses per UTC day, across all visitors."""

    def __init__(self, limit: int, today=lambda: datetime.now(timezone.utc).date()):
        self.limit, self._today = limit, today
        self._day, self._used = today(), 0
        self._lock = threading.Lock()

    def _roll(self):
        if self._today() != self._day:
            self._day, self._used = self._today(), 0

    def take(self) -> bool:
        with self._lock:
            self._roll()
            if self._used >= self.limit:
                return False
            self._used += 1
            return True

    def remaining(self) -> int:
        with self._lock:
            self._roll()
            return max(0, self.limit - self._used)
