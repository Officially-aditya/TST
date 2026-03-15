import hashlib
import secrets
import hmac
from typing import Optional


SECRET_KEY = "tst-memory-system-secret-2026"


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """
    Hash a password with SHA-256 + salt.
    Returns (hash_hex: str, salt: str).
    If salt is not provided, generates a random 16-byte salt.
    """
    if salt is None:
        salt = secrets.token_hex(16)
    combined = f"{salt}:{password}"
    hash_hex = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return hash_hex, salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verify a password against a stored hash + salt."""
    computed_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(computed_hash, stored_hash)


def generate_session_token() -> str:
    """Generate a cryptographically secure session token."""
    return secrets.token_urlsafe(32)


def generate_api_key(user_id: str) -> str:
    """
    Generate a deterministic API key for a user.
    Key is derived from user_id + SECRET_KEY via HMAC-SHA256.
    Returns hex string.
    """
    return hmac.new(
        SECRET_KEY.encode("utf-8"),
        user_id.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
