"""
Tier 1 — JWT Authentication & Authorization
=============================================
Protects API endpoints with JWT tokens.
- API key auth    (X-API-Key header)
- JWT Bearer auth (Authorization: Bearer <token>)
- Role-based access control: admin / operator / viewer
- Constant-time comparison to prevent timing-attack exploits
"""

import os
import time
import hashlib
import hmac
import base64
import json
from functools import wraps
from typing import Optional, Dict, Any
from flask import request, jsonify, g

JWT_SECRET         = os.getenv("JWT_SECRET_KEY", "dev-secret-change-in-production")
JWT_EXPIRY_SECONDS = int(os.getenv("JWT_EXPIRY_SECONDS", "3600"))   # 1 hour

# In production: load from HashiCorp Vault / AWS Secrets Manager / K8s Secret
API_KEYS: Dict[str, Dict[str, str]] = {
    "mlops-admin-key-dev":    {"role": "admin",    "client_id": "admin-service"},
    "mlops-operator-key-dev": {"role": "operator", "client_id": "ops-service"},
    "mlops-viewer-key-dev":   {"role": "viewer",   "client_id": "monitoring-service"},
}

ROLE_PERMISSIONS: Dict[str, set] = {
    "admin":    {"predict", "retrain", "explain", "drift", "health", "metrics",
                 "batch", "ab_test", "governance", "audit"},
    "operator": {"predict", "retrain", "explain", "drift", "health", "metrics", "batch"},
    "viewer":   {"predict", "explain", "drift", "health", "metrics"},
}


# ── JWT helpers ──────────────────────────────────────────────────────────────

def _b64enc(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64dec(s: str) -> bytes:
    pad = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * (pad % 4))


def generate_token(client_id: str, role: str,
                   expiry: int = JWT_EXPIRY_SECONDS) -> str:
    """Generate a signed HS256 JWT token."""
    header  = _b64enc(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64enc(json.dumps({
        "sub":  client_id,
        "role": role,
        "iat":  int(time.time()),
        "exp":  int(time.time()) + expiry,
    }).encode())
    signing_input = f"{header}.{payload}"
    sig = _b64enc(
        hmac.new(JWT_SECRET.encode(), signing_input.encode(), hashlib.sha256).digest()
    )
    return f"{signing_input}.{sig}"


def validate_token(token: str) -> Optional[Dict[str, Any]]:
    """Validate JWT. Returns payload dict or None if invalid/expired."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        h, p, sig = parts
        signing_input = f"{h}.{p}"
        expected = _b64enc(
            hmac.new(JWT_SECRET.encode(), signing_input.encode(), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(_b64dec(p))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


def get_current_user() -> Optional[Dict[str, str]]:
    """Extract authenticated user from request. Returns None if unauthenticated."""
    # 1. API Key
    api_key = request.headers.get("X-API-Key", "")
    if api_key and api_key in API_KEYS:
        info = API_KEYS[api_key]
        return {"client_id": info["client_id"], "role": info["role"],
                "auth_type": "api_key"}
    # 2. Bearer JWT
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        payload = validate_token(auth[7:])
        if payload:
            return {
                "client_id": payload.get("sub", "unknown"),
                "role":      payload.get("role", "viewer"),
                "auth_type": "jwt",
            }
    return None


def require_auth(permission: str = "predict"):
    """Route decorator: enforce auth + RBAC permission check."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            user = get_current_user()
            if not user:
                return jsonify({
                    "error": "Authentication required",
                    "code":  "UNAUTHORIZED",
                    "hint":  "Send X-API-Key header or Authorization: Bearer <token>",
                }), 401
            allowed = ROLE_PERMISSIONS.get(user["role"], set())
            if permission not in allowed:
                return jsonify({
                    "error":      f"Permission denied: requires '{permission}'",
                    "code":       "FORBIDDEN",
                    "your_role":  user["role"],
                    "required":   permission,
                }), 403
            g.current_user = user
            return f(*args, **kwargs)
        return decorated
    return decorator
