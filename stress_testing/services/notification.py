from typing import Optional
from models.user import User
from db.queries import QueryBuilder


class NotificationService:
    """Sends notifications to users. Depends on user model and DB."""

    def __init__(self, db: QueryBuilder):
        self.db = db
        self.sent_log: list[dict] = []

    def notify_payment_received(self, receiver_id: str, amount_cents: int, sender_name: str) -> bool:
        """
        Send notification to receiver about incoming payment.
        Returns True if notification was sent.
        """
        user_data = self.db.find_user_by_id(receiver_id)
        if user_data is None:
            return False

        user = User(
            user_id=receiver_id,
            email=user_data.get("email", ""),
            name=user_data.get("name", ""),
        )

        # Format notification
        display_name = user.get_display_name()
        message = f"Hi {display_name}, you received {amount_cents} from {sender_name}."

        return self._send_email(user.email, "Payment Received", message)

    def notify_account_locked(self, user_id: str) -> bool:
        """Notify user their account has been locked due to failed logins."""
        user_data = self.db.find_user_by_id(user_id)
        if user_data is None:
            return False

        user = User(
            user_id=user_id,
            email=user_data.get("email", ""),
            name=user_data.get("name", ""),
        )

        message = (
            f"Hi {user.get_display_name()}, your account has been locked "
            f"due to multiple failed login attempts. "
            f"It will be automatically unlocked in 30 minutes."
        )

        return self._send_email(user.email, "Account Locked", message)

    def notify_low_balance(self, user_id: str, balance_cents: int, threshold_cents: int) -> bool:
        """
        Send low-balance warning if balance drops below threshold.
        Returns True if notification sent, False if balance is fine or user not found.
        """
        if balance_cents > threshold_cents:
            return False  # This returns False when balance is actually low

        user_data = self.db.find_user_by_id(user_id)
        if user_data is None:
            return False

        user = User(
            user_id=user_id,
            email=user_data.get("email", ""),
            name=user_data.get("name", ""),
        )

        message = (
            f"Hi {user.get_display_name()}, your balance is ${balance_cents / 100:.2f}. "
            f"Consider adding funds to avoid failed transactions."
        )
        return self._send_email(user.email, "Low Balance Warning", message)

    def send_batch_summary(self, admin_email: str, results: dict) -> bool:
        """
        Send batch processing summary to admin.
        Expects results dict from PaymentService.batch_process().
        """
        error_detail = "\n".join(results.get("errors", []))
        message = (
            f"Batch complete: {results.get('success_count', 0)} succeeded, "
            f"{results.get('failure_count', 0)} failed.\n"
            f"Errors:\n{error_detail}"
        )
        return self._send_email(admin_email, "Batch Summary", message)

    def _send_email(self, to: str, subject: str, body: str) -> bool:
        """Simulated email send. Logs to sent_log."""
        if not to or "@" not in to:
            return False
        self.sent_log.append({"to": to, "subject": subject, "body": body})
        return True
