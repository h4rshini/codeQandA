from datetime import date

from web.guard import AnswerCache, DailyCap, RateLimiter, normalize


def test_normalize_ignores_case_spacing_and_trailing_punctuation():
    assert normalize("  Where is  X defined? ") == normalize("where is x defined") == "where is x defined"


def test_cache_roundtrip_and_eviction():
    c = AnswerCache(max_entries=2)
    c.put("a?", [{"type": "start"}], {"answer": "1"})
    c.put("b", [], {"answer": "2"})
    c.put("c", [], {"answer": "3"})  # evicts "a"
    assert c.get("A") is None and c.get("b")["answer"]["answer"] == "2"


def test_rate_limiter_sliding_window():
    now = [0.0]
    rl = RateLimiter(limit=2, window=60, clock=lambda: now[0])
    assert rl.allow("ip")[0] and rl.allow("ip")[0]
    ok, wait = rl.allow("ip")
    assert not ok and wait == 60
    assert rl.allow("other-ip")[0]         # limits are per visitor
    now[0] = 61
    assert rl.allow("ip")[0]               # window slid


def test_daily_cap_resets_each_day():
    day = [date(2026, 1, 1)]
    cap = DailyCap(limit=1, today=lambda: day[0])
    assert cap.take() and not cap.take() and cap.remaining() == 0
    day[0] = date(2026, 1, 2)
    assert cap.remaining() == 1 and cap.take()
