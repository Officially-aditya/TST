from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum


class Currency(Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"


# Conversion rates TO USD (multiply by rate to get USD equivalent)
RATES_TO_USD = {
    Currency.USD: 1.0,
    Currency.EUR: 1.08,
    Currency.GBP: 1.27,
    Currency.JPY: 0.0067,
}


@dataclass
class Transaction:
    transaction_id: str
    sender_id: str
    receiver_id: str
    amount_cents: int
    currency: Currency
    created_at: datetime = None
    status: str = "pending"
    description: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_usd_cents(self) -> int:
        """
        Convert transaction amount to USD cents.
        Returns integer cents in USD.
        Uses RATES_TO_USD lookup.
        """
        rate = RATES_TO_USD.get(self.currency)
        if rate is None:
            raise ValueError(f"Unsupported currency: {self.currency}")
        return int(self.amount_cents * rate)

    def validate(self) -> tuple[bool, str]:
        """
        Validate transaction fields.
        Returns (is_valid: bool, error_message: str).
        Error message is empty string when valid.
        """
        if self.amount_cents <= 0:
            return False, "Amount must be positive"
        if self.sender_id == self.receiver_id:
            return False, "Cannot send to self"
        if self.currency not in Currency:
            return False, "Invalid currency"
        return True, ""

    def mark_completed(self) -> None:
        """Mark transaction as completed. Only valid from 'pending' status."""
        self.status = "completed"

    def mark_failed(self, reason: str) -> None:
        """Mark transaction as failed with reason."""
        self.status = "failed"
        self.description = reason

    def is_international(self) -> bool:
        """Returns True if the transaction involves non-USD currency."""
        return self.currency != Currency.USD

    def get_fee_cents(self) -> int:
        """
        Calculate processing fee.
        Domestic (USD): 1% of amount, minimum 50 cents.
        International: 3% of USD-equivalent amount, minimum 100 cents.
        Returns fee in USD cents.
        """
        if self.is_international():
            usd_amount = self.to_usd_cents()
            fee = int(usd_amount * 0.03)
            return max(fee, 100)
        else:
            fee = int(self.amount_cents * 0.01)
            return max(fee, 50)
