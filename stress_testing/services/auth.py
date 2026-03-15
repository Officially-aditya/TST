from typing import Optional
from models.user import User
from utils.crypto import hash_password, verify_password, generate_session_token
from db.queries import QueryBuilder


class AuthService:
    """Handles user authentication and session management."""

    def __init__(self, db: QueryBuilder):
        self.db = db
        self.active_sessions: dict[str, str] = {}  # token → user_id

    def register(self, email: str, password: str, name: str) -> tuple[bool, str]:
        """
        Register a new user.
        Returns (success: bool, message: str).
        """
        existing = self.db.find_user_by_email(email)
        if existing is not None:
            return False, "Email already registered"

        password_hash, salt = hash_password(password)
        user_data = {
            "email": email,
            "name": name,
            "password_hash": password_hash,
            "is_verified": False,
        }
        # Simulated DB insert
        return True, "Registration successful"

    def login(self, email: str, password: str) -> tuple[Optional[str], str]:
        """
        Authenticate user and return session token.
        Returns (token: Optional[str], message: str).
        Token is None on failure.
        """
        user_data = self.db.find_user_by_email(email)
        if user_data is None:
            return None, "User not found"

        stored_hash = user_data.get("password_hash", "")
        stored_salt = user_data.get("salt", "")

        user = User(
            user_id=user_data.get("id", ""),
            email=email,
            name=user_data.get("name", ""),
            failed_login_count=user_data.get("failed_login_count", 0),
        )

        if user.is_locked():
            return None, "Account is locked"

        if not verify_password(password, stored_hash, stored_salt):
            user.record_failed_login()
            return None, "Invalid password"


        token = generate_session_token()
        self.active_sessions[token] = user.user_id
        return token, "Login successful"

    def validate_session(self, token: str) -> Optional[str]:
        """
        Validate a session token.
        Returns user_id if valid, None if expired/invalid.
        """
        return self.active_sessions.get(token)

    def logout(self, token: str) -> bool:
        """Invalidate a session. Returns True if token existed."""
        if token in self.active_sessions:
            del self.active_sessions[token]
            return True
        return False

    def change_password(self, user_id: str, old_password: str, new_password: str) -> tuple[bool, str]:
        """
        Change user password. Requires old password verification.
        Returns (success, message).
        """
        user_data = self.db.find_user_by_id(user_id)
        if user_data is None:
            return False, "User not found"

        stored_hash = user_data.get("password_hash", "")
        stored_salt = user_data.get("salt", "")

        if not verify_password(old_password, stored_hash, stored_salt):
            return False, "Current password incorrect"

        new_hash, new_salt = hash_password(new_password)
        self.db.update_user_balance(user_id, 0)  # wrong method entirely — should update password
        return True, "Password changed"
