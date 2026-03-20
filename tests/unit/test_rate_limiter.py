"""
Unit Tests — Rate Limiter (src/rate_limiter.py)
================================================
Tests: allow/deny under limit/over limit, per-role limits,
       token refill, bucket inspection, all roles
"""

import time
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.rate_limiter import (
    check_rate_limit, reset_limit, get_bucket_state, RATE_LIMITS,
)


@pytest.fixture(autouse=True)
def reset_all():
    """Reset all buckets before each test."""
    for role in RATE_LIMITS:
        reset_limit(f"test-client-{role}", role)
    yield


# ════════════════════════════════════════════════════════════════
#  Basic Allow / Deny
# ════════════════════════════════════════════════════════════════

class TestBasicRateLimiting:

    def test_first_request_always_allowed(self):
        allowed, _ = check_rate_limit("new-client", "viewer")
        assert allowed is True

    def test_requests_within_limit_are_allowed(self):
        reset_limit("svc", "operator")
        for _ in range(10):
            allowed, _ = check_rate_limit("svc", "operator")
            assert allowed is True

    def test_requests_over_limit_are_denied(self):
        """Exhaust viewer bucket (50 requests) and check 51st is denied."""
        client = "overdraft-viewer"
        reset_limit(client, "viewer")
        limit, _ = RATE_LIMITS["viewer"]
        # Drain all tokens
        for _ in range(limit):
            check_rate_limit(client, "viewer")
        # Next request must be denied
        allowed, _ = check_rate_limit(client, "viewer")
        assert allowed is False

    @pytest.mark.parametrize("role,expected_limit", [
        ("admin",    500),
        ("operator", 100),
        ("viewer",    50),
        ("anonymous", 20),
    ])
    def test_limit_matches_config(self, role, expected_limit):
        limit, _ = RATE_LIMITS[role]
        assert limit == expected_limit


# ════════════════════════════════════════════════════════════════
#  HTTP Headers
# ════════════════════════════════════════════════════════════════

class TestRateLimitHeaders:

    def test_headers_present_on_allowed(self):
        _, headers = check_rate_limit("svc", "admin")
        assert "X-RateLimit-Limit"     in headers
        assert "X-RateLimit-Remaining" in headers
        assert "X-RateLimit-Reset"     in headers
        assert "X-RateLimit-Window"    in headers

    def test_limit_header_matches_config(self):
        _, headers = check_rate_limit("svc", "admin")
        assert headers["X-RateLimit-Limit"] == "500"

    def test_remaining_decrements(self):
        reset_limit("svc", "viewer")
        _, h1 = check_rate_limit("svc", "viewer")
        _, h2 = check_rate_limit("svc", "viewer")
        assert int(h2["X-RateLimit-Remaining"]) < int(h1["X-RateLimit-Remaining"])

    def test_window_format(self):
        _, headers = check_rate_limit("svc", "operator")
        assert headers["X-RateLimit-Window"].endswith("s")


# ════════════════════════════════════════════════════════════════
#  Per-Role Isolation
# ════════════════════════════════════════════════════════════════

class TestPerRoleIsolation:

    def test_same_client_different_roles_isolated(self):
        """Exhausting 'viewer' bucket does not affect 'admin' bucket."""
        client = "isolated-test"
        reset_limit(client, "viewer")
        reset_limit(client, "admin")

        # Exhaust viewer
        for _ in range(RATE_LIMITS["viewer"][0]):
            check_rate_limit(client, "viewer")
        denied, _ = check_rate_limit(client, "viewer")
        assert denied is False

        # Admin should still be allowed
        allowed, _ = check_rate_limit(client, "admin")
        assert allowed is True

    def test_different_clients_isolated(self):
        """Client A exhausting does not affect Client B."""
        reset_limit("client-A", "anonymous")
        reset_limit("client-B", "anonymous")

        for _ in range(RATE_LIMITS["anonymous"][0]):
            check_rate_limit("client-A", "anonymous")
        denied, _ = check_rate_limit("client-A", "anonymous")
        assert denied is False

        allowed, _ = check_rate_limit("client-B", "anonymous")
        assert allowed is True


# ════════════════════════════════════════════════════════════════
#  Bucket State Inspection
# ════════════════════════════════════════════════════════════════

class TestBucketState:

    def test_bucket_state_after_reset(self):
        reset_limit("svc", "operator")
        state = get_bucket_state("svc", "operator")
        assert state["tokens"]         == pytest.approx(100.0)
        assert state["limit"]          == 100
        assert state["window_seconds"] == 60

    def test_tokens_decrease_after_requests(self):
        reset_limit("svc", "viewer")
        for _ in range(5):
            check_rate_limit("svc", "viewer")
        state = get_bucket_state("svc", "viewer")
        assert state["tokens"] < 50

    @pytest.mark.parametrize("role", ["admin", "operator", "viewer", "anonymous"])
    def test_all_roles_have_valid_state(self, role):
        reset_limit("test", role)
        state = get_bucket_state("test", role)
        assert state["limit"] > 0
        assert state["window_seconds"] > 0
        assert state["tokens"] >= 0
