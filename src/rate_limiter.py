"""
Tier 2 — Rate Limiter (Token Bucket, In-Memory)
================================================
Per-client, per-role token-bucket rate limiting.
In production: swap _buckets dict for Redis-backed flask-limiter.

Limits (requests / window):
  admin    → 500 / 60s
  operator → 100 / 60s
  viewer   →  50 / 60s
  anonymous→  20 / 60s
"""

import time
import threading
from typing import Dict, Tuple
from collections import defaultdict

RATE_LIMITS: Dict[str, Tuple[int, int]] = {
    "admin":     (500, 60),
    "operator":  (100, 60),
    "viewer":    (50,  60),
    "anonymous": (20,  60),
}

_buckets: Dict[str, Dict] = defaultdict(dict)
_lock = threading.Lock()


def _bucket_key(client_id: str, role: str) -> str:
    return f"{role}:{client_id}"


def _get_or_create(key: str, limit: int) -> Dict:
    if key not in _buckets:
        _buckets[key] = {"tokens": float(limit), "last_refill": time.time()}
    return _buckets[key]


def check_rate_limit(
    client_id: str,
    role: str = "anonymous",
) -> Tuple[bool, Dict[str, str]]:
    """
    Check if request is within rate limit.

    Returns:
        (allowed, http_headers_dict)
    """
    limit, window = RATE_LIMITS.get(role, RATE_LIMITS["anonymous"])
    key = _bucket_key(client_id, role)

    with _lock:
        bucket = _get_or_create(key, limit)
        now     = time.time()
        elapsed = now - bucket["last_refill"]

        # Refill proportional to elapsed time
        refill = elapsed * (limit / window)
        bucket["tokens"]      = min(float(limit), bucket["tokens"] + refill)
        bucket["last_refill"] = now

        allowed = bucket["tokens"] >= 1.0
        if allowed:
            bucket["tokens"] -= 1.0

        remaining = max(0, int(bucket["tokens"]))
        reset_at  = int(now + window)

    headers = {
        "X-RateLimit-Limit":     str(limit),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset":     str(reset_at),
        "X-RateLimit-Window":    f"{window}s",
        "X-RateLimit-Role":      role,
    }
    return allowed, headers


def reset_limit(client_id: str, role: str = "anonymous") -> None:
    """Reset bucket to full (used in tests and after manual override)."""
    key   = _bucket_key(client_id, role)
    limit, _ = RATE_LIMITS.get(role, RATE_LIMITS["anonymous"])
    with _lock:
        _buckets[key] = {"tokens": float(limit), "last_refill": time.time()}


def get_bucket_state(client_id: str, role: str = "anonymous") -> Dict:
    """Inspect current bucket state (for debugging)."""
    key   = _bucket_key(client_id, role)
    limit, window = RATE_LIMITS.get(role, RATE_LIMITS["anonymous"])
    with _lock:
        bucket = _get_or_create(key, limit)
        return {
            "client_id": client_id,
            "role": role,
            "tokens": round(bucket["tokens"], 2),
            "limit": limit,
            "window_seconds": window,
        }
