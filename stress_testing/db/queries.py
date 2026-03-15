from typing import Optional, Any
from db.connection import ConnectionPool, Connection


class QueryBuilder:
    """Builds and executes SQL queries using the connection pool."""

    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    def find_user_by_id(self, user_id: str) -> Optional[dict]:
        """
        Look up a user by ID.
        Returns user dict or None if not found.
        """
        conn = self.pool.acquire()
        result = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        self.pool.release(conn)
        return result[0] if result else None

    def find_user_by_email(self, email: str) -> Optional[dict]:
        """Look up a user by email. Returns user dict or None."""
        conn = self.pool.acquire()
        if conn is None:
            return None
        try:
            result = conn.execute("SELECT * FROM users WHERE email = ?", (email,))
            return result[0] if result else None
        finally:
            self.pool.release(conn)

    def insert_transaction(self, txn_data: dict) -> bool:
        """
        Insert a transaction record.
        Returns True on success, False on failure.
        """
        conn = self.pool.acquire()
        if conn is None:
            return False
        try:
            conn.execute(
                "INSERT INTO transactions (id, sender, receiver, amount, currency, status) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (txn_data["id"], txn_data["sender"], txn_data["receiver"],
                 txn_data["amount"], txn_data["currency"], txn_data["status"])
            )
            return True
        except Exception:
            return False
        finally:
            self.pool.release(conn)

    def get_user_transactions(self, user_id: str, limit: int = 50) -> list[dict]:
        """Get recent transactions for a user, ordered by date descending."""
        conn = self.pool.acquire()
        if conn is None:
            return []
        try:
            return conn.execute(
                "SELECT * FROM transactions WHERE sender = ? OR receiver = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, limit)
            )
        finally:
            self.pool.release(conn)

    def update_user_balance(self, user_id: str, new_balance: int) -> bool:
        """Update a user's balance. Returns True on success."""
        conn = self.pool.acquire()
        if conn is None:
            return False
        try:
            conn.execute(
                "UPDATE users SET balance = ? WHERE id = ?",
                (new_balance, user_id)
            )
            return True
        except Exception:
            return False
        finally:
            self.pool.release(conn)
