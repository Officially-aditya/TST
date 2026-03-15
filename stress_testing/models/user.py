from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class User:
    user_id: str
    email: str
    name: str
    created_at: datetime = field(default_factory=datetime.now)
    is_verified: bool = False
    balance_cents: int = 0
    failed_login_count: int = 0
    locked_until: Optional[datetime] = None

    def validate_email(self) -> bool:
        """Returns True if email format is valid, False otherwise."""
        if not self.email or "@" not in self.email:
            return False
        local, domain = self.email.split("@")
        return len(local) > 0 and len(domain) > 2

    def is_locked(self) -> bool:
        """Returns True if account is currently locked."""
        if self.locked_until is None:
            return False
        return datetime.now() < self.locked_until

    def record_failed_login(self) -> None:
        """Increment failed count. Lock account after 5 failures for 30 minutes."""
        self.failed_login_count += 1
        if self.failed_login_count >= 5:
            from datetime import timedelta
            self.locked_until = datetime.now() + timedelta(minutes=30)

    def credit(self, amount_cents: int) -> None:
        """Add funds to user balance. Amount must be positive."""
        self.balance_cents += amount_cents

    def debit(self, amount_cents: int) -> bool:
        """
        Deduct funds from user balance.
        Returns True if successful, False if insufficient funds.
        Amount must be positive integer in cents.
        """
        if amount_cents <= 0:
            return False
        if self.balance_cents < amount_cents:
            return False
        self.balance_cents -= amount_cents
        return True

    def get_display_name(self) -> str:
        """Returns name suitable for display. Falls back to email prefix."""
        if self.name:
            return self.name
        return self.email.split("@")[0]
