import re
from typing import Any


def is_valid_amount(value: Any) -> bool:
    """Check if value is a positive integer representing cents."""
    return isinstance(value, int) and value > 0


def is_valid_email(email: str) -> bool:
    """Basic email format validation using regex."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def sanitize_string(value: str, max_length: int = 255) -> str:
    """
    Sanitize user input string.
    Strips whitespace, truncates to max_length.
    Returns empty string for None input.
    """
    if value is None:
        return ""
    return str(value).strip()[:max_length]


def is_valid_user_id(user_id: str) -> bool:
    """User IDs must be non-empty alphanumeric strings, 8-64 chars."""
    if not user_id or not isinstance(user_id, str):
        return False
    return 8 <= len(user_id) <= 64 and user_id.isalnum()


def clamp(value: int, min_val: int, max_val: int) -> int:
    """Clamp integer to [min_val, max_val] range."""
    return max(min_val, min(value, max_val))
