"""In-process sliding-window rate limiting for the chat endpoints.

Deliberately dependency-free and per-process: each worker keeps its own counters, so
with N workers/replicas a client can make up to N x the configured limit. That's an
acceptable bound for abuse protection on a single-worker deploy; a shared store (e.g.
Redis) would be needed for an exact global limit.
"""

import math
import time
from collections import OrderedDict, deque

from fastapi import Depends, HTTPException, Request

from app.core.auth import get_current_user_id
from app.core.config import get_settings

_WINDOW_SECONDS = 60.0
# Upper bound on tracked keys; least-recently-seen keys are evicted beyond it. Each key
# holds at most `limit` timestamps, so memory is bounded by _MAX_KEYS * limit.
_MAX_KEYS = 10_000


class SlidingWindowLimiter:
    def __init__(self, window_seconds: float = _WINDOW_SECONDS, max_keys: int = _MAX_KEYS) -> None:
        self._window = window_seconds
        self._max_keys = max_keys
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._last_sweep = 0.0

    def hit(self, key: str, limit: int, now: float | None = None) -> float | None:
        """Record a request for `key`; returns None if allowed, else seconds until a slot frees up."""
        now = time.monotonic() if now is None else now
        self._sweep(now)
        cutoff = now - self._window
        hits = self._hits.get(key)
        if hits is None:
            hits = self._hits[key] = deque()
        else:
            self._hits.move_to_end(key)
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= limit:
            return hits[0] + self._window - now
        hits.append(now)
        while len(self._hits) > self._max_keys:
            self._hits.popitem(last=False)
        return None

    def _sweep(self, now: float) -> None:
        """Drop keys with no hits inside the window, at most once per window."""
        if now - self._last_sweep < self._window:
            return
        self._last_sweep = now
        cutoff = now - self._window
        for key in [k for k, hits in self._hits.items() if not hits or hits[-1] <= cutoff]:
            del self._hits[key]

    def reset(self) -> None:
        self._hits.clear()
        self._last_sweep = 0.0

    def __len__(self) -> int:
        return len(self._hits)


chat_limiter = SlidingWindowLimiter()


def client_ip(request: Request) -> str:
    """The caller's IP: left-most X-Forwarded-For entry when present, else the socket peer.

    The app runs behind FastAPI Cloud's proxy, where `request.client.host` is the proxy
    for every caller. The left-most entry is client-supplied and therefore spoofable; it
    is good enough to separate ordinary anonymous users, not to stop a determined one.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


async def enforce_chat_rate_limit(request: Request, user_id: str | None = Depends(get_current_user_id)) -> None:
    """Reject with 429 + Retry-After once a user (or anonymous IP) exceeds the per-minute limit."""
    limit = get_settings().chat_rate_limit_per_minute
    if limit <= 0:
        return
    key = f"user:{user_id}" if user_id else f"ip:{client_ip(request)}"
    retry_after = chat_limiter.hit(key, limit)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="Too many messages. Please wait a moment and try again.",
            headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
        )
