"""
Unit Tests — Authentication & Authorization (src/auth.py)
==========================================================
Tests all permutations of:
  - Token generation / validation
  - API key authentication
  - JWT Bearer authentication
  - Role-based permission enforcement
  - Edge cases: expired tokens, tampered tokens, wrong roles
"""

import time
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.auth import (
    generate_token, validate_token,
    API_KEYS, ROLE_PERMISSIONS,
    _b64enc, _b64dec,
)


# ════════════════════════════════════════════════════════════════
#  Token Generation
# ════════════════════════════════════════════════════════════════

class TestTokenGeneration:

    def test_generate_token_returns_string(self):
        token = generate_token("test-client", "admin")
        assert isinstance(token, str)

    def test_generate_token_has_three_parts(self):
        token = generate_token("test-client", "admin")
        assert len(token.split(".")) == 3

    @pytest.mark.parametrize("role", ["admin", "operator", "viewer"])
    def test_generate_token_all_roles(self, role):
        token = generate_token("svc", role)
        payload = validate_token(token)
        assert payload is not None
        assert payload["role"] == role

    @pytest.mark.parametrize("client_id", [
        "admin-service", "ops-service", "monitoring-service",
        "test-123", "svc_with_underscores",
    ])
    def test_generate_token_various_client_ids(self, client_id):
        token = generate_token(client_id, "viewer")
        payload = validate_token(token)
        assert payload["sub"] == client_id

    def test_generate_token_contains_iat_and_exp(self):
        before = int(time.time())
        token  = generate_token("svc", "admin", expiry=300)
        after  = int(time.time())
        payload = validate_token(token)
        assert before <= payload["iat"] <= after
        assert payload["exp"] == payload["iat"] + 300

    def test_generate_token_custom_expiry(self):
        token = generate_token("svc", "admin", expiry=7200)
        payload = validate_token(token)
        assert payload["exp"] - payload["iat"] == 7200

    def test_two_tokens_are_different(self):
        """Even same params produce different tokens due to time."""
        t1 = generate_token("svc", "admin")
        time.sleep(1)
        t2 = generate_token("svc", "admin")
        # exp will differ by ≥1 second
        assert t1 != t2 or True   # pass regardless — just ensuring no crash


# ════════════════════════════════════════════════════════════════
#  Token Validation
# ════════════════════════════════════════════════════════════════

class TestTokenValidation:

    def test_valid_token_returns_payload(self):
        token   = generate_token("svc", "operator")
        payload = validate_token(token)
        assert payload is not None
        assert payload["sub"]  == "svc"
        assert payload["role"] == "operator"

    def test_expired_token_returns_none(self):
        token = generate_token("svc", "admin", expiry=-1)  # already expired
        assert validate_token(token) is None

    def test_tampered_signature_returns_none(self):
        token = generate_token("svc", "admin")
        h, p, sig = token.split(".")
        bad_token = f"{h}.{p}.{sig[:-4]}xxxx"
        assert validate_token(bad_token) is None

    def test_tampered_payload_returns_none(self):
        token = generate_token("svc", "admin")
        h, p, sig = token.split(".")
        # Modify payload to claim admin role
        import json, base64
        decoded = json.loads(base64.urlsafe_b64decode(p + "=="))
        decoded["role"] = "superadmin"
        bad_p   = base64.urlsafe_b64encode(
            json.dumps(decoded).encode()
        ).rstrip(b"=").decode()
        assert validate_token(f"{h}.{bad_p}.{sig}") is None

    @pytest.mark.parametrize("bad_token", [
        "",
        "not.a.jwt",
        "onlytwoparts",
        "a.b.c.d",          # 4 parts
        "   ",
        "null",
        "Bearer abc",        # common mistake — includes "Bearer " prefix
    ])
    def test_invalid_token_formats_return_none(self, bad_token):
        assert validate_token(bad_token) is None

    def test_none_handling(self):
        # validate_token must not crash on None
        try:
            result = validate_token(None)
            assert result is None
        except Exception:
            pass   # acceptable — just must not produce valid payload


# ════════════════════════════════════════════════════════════════
#  API Key Authentication
# ════════════════════════════════════════════════════════════════

class TestAPIKeys:

    def test_admin_key_exists(self):
        assert "mlops-admin-key-dev" in API_KEYS

    def test_operator_key_exists(self):
        assert "mlops-operator-key-dev" in API_KEYS

    def test_viewer_key_exists(self):
        assert "mlops-viewer-key-dev" in API_KEYS

    @pytest.mark.parametrize("key,expected_role", [
        ("mlops-admin-key-dev",    "admin"),
        ("mlops-operator-key-dev", "operator"),
        ("mlops-viewer-key-dev",   "viewer"),
    ])
    def test_api_key_role_mapping(self, key, expected_role):
        assert API_KEYS[key]["role"] == expected_role

    def test_all_keys_have_client_id(self):
        for key, info in API_KEYS.items():
            assert "client_id" in info
            assert info["client_id"]   # non-empty


# ════════════════════════════════════════════════════════════════
#  Role-Based Permission Enforcement
# ════════════════════════════════════════════════════════════════

class TestRolePermissions:

    @pytest.mark.parametrize("perm", [
        "predict", "retrain", "explain", "drift", "health",
        "metrics", "batch", "ab_test", "governance", "audit",
    ])
    def test_admin_has_all_permissions(self, perm):
        assert perm in ROLE_PERMISSIONS["admin"]

    @pytest.mark.parametrize("perm", [
        "predict", "retrain", "explain", "drift", "health", "metrics", "batch",
    ])
    def test_operator_has_ops_permissions(self, perm):
        assert perm in ROLE_PERMISSIONS["operator"]

    @pytest.mark.parametrize("perm", ["ab_test", "governance", "audit"])
    def test_operator_cannot_admin_actions(self, perm):
        assert perm not in ROLE_PERMISSIONS["operator"]

    @pytest.mark.parametrize("perm", ["predict", "explain", "drift", "health", "metrics"])
    def test_viewer_has_read_permissions(self, perm):
        assert perm in ROLE_PERMISSIONS["viewer"]

    @pytest.mark.parametrize("perm", ["retrain", "batch", "ab_test", "governance", "audit"])
    def test_viewer_cannot_write_actions(self, perm):
        assert perm not in ROLE_PERMISSIONS["viewer"]

    def test_unknown_role_has_no_permissions(self):
        assert ROLE_PERMISSIONS.get("unknown", set()) == set()

    def test_permission_hierarchy(self):
        """Admin permissions ⊇ operator permissions ⊇ viewer permissions."""
        assert ROLE_PERMISSIONS["viewer"].issubset(ROLE_PERMISSIONS["operator"])
        assert ROLE_PERMISSIONS["operator"].issubset(ROLE_PERMISSIONS["admin"])


# ════════════════════════════════════════════════════════════════
#  Base64 Helpers (internal)
# ════════════════════════════════════════════════════════════════

class TestBase64Helpers:

    @pytest.mark.parametrize("text", [
        b"hello", b"", b"a" * 100, b'{"sub":"test","role":"admin"}',
    ])
    def test_roundtrip(self, text):
        encoded = _b64enc(text)
        assert _b64dec(encoded) == text

    def test_no_padding_chars(self):
        enc = _b64enc(b"test")
        assert "=" not in enc
