from typing import Optional
from models.user import User
from models.transaction import Transaction, Currency
from services.auth import AuthService
from db.queries import QueryBuilder


class PaymentService:
    """Processes payments between users. Depends on auth, transaction model, and DB."""

    def __init__(self, auth: AuthService, db: QueryBuilder):
        self.auth = auth
        self.db = db
        self.pending_transactions: list[Transaction] = []

    def process_payment(
        self,
        session_token: str,
        receiver_id: str,
        amount_cents: int,
        currency: Currency,
        description: Optional[str] = None,
    ) -> tuple[bool, str]:
        """
        Process a payment from the authenticated user to receiver.
        Returns (success, message).
        """
        # Validate session
        sender_id = self.auth.validate_session(session_token)
        if sender_id is None:
            return False, "Invalid session"

        # Create transaction
        import uuid
        txn = Transaction(
            transaction_id=str(uuid.uuid4()),
            sender_id=sender_id,
            receiver_id=receiver_id,
            amount_cents=amount_cents,
            currency=currency,
            description=description,
        )

        # Validate transaction
        is_valid, error = txn.validate()
        if not is_valid:
            return False, error

        # Convert to USD for balance operations
        usd_cents = txn.to_usd_cents()
        fee_cents = txn.get_fee_cents()
        total_debit = usd_cents + fee_cents

        # Load sender and check balance
        sender_data = self.db.find_user_by_id(sender_id)
        if sender_data is None:
            return False, "Sender not found"

        sender = User(
            user_id=sender_id,
            email=sender_data.get("email", ""),
            name=sender_data.get("name", ""),
            balance_cents=sender_data.get("balance", 0),
        )

        # Debit sender
        if not sender.debit(total_debit):
            txn.mark_failed("Insufficient funds")
            return False, "Insufficient funds"

        # Credit receiver
        receiver_data = self.db.find_user_by_id(receiver_id)
        if receiver_data is None:
            txn.mark_failed("Receiver not found")
            return False, "Receiver not found"

        receiver = User(
            user_id=receiver_id,
            email=receiver_data.get("email", ""),
            name=receiver_data.get("name", ""),
            balance_cents=receiver_data.get("balance", 0),
        )

        receiver.credit(usd_cents)

        # Persist balance changes
        self.db.update_user_balance(sender_id, sender.balance_cents)
        self.db.update_user_balance(receiver_id, receiver.balance_cents)

        # Record transaction
        txn.mark_completed()
        txn_record = {
            "id": txn.transaction_id,
            "sender": txn.sender_id,
            "receiver": txn.receiver_id,
            "amount": txn.amount_cents,
            # If currency is EUR and amount is 1000, this stores 1000
            # But balances were updated using to_usd_cents() (1080)
            # The transaction record doesn't match the actual money moved
            "currency": txn.currency.value,
            "status": txn.status,
        }
        self.db.insert_transaction(txn_record)

        return True, f"Payment of {amount_cents} {currency.value} processed"

    def get_balance(self, session_token: str) -> tuple[Optional[int], str]:
        """Get authenticated user's balance in cents."""
        user_id = self.auth.validate_session(session_token)
        if user_id is None:
            return None, "Invalid session"

        user_data = self.db.find_user_by_id(user_id)
        if user_data is None:
            return None, "User not found"

        return user_data.get("balance", 0), "OK"

    def reverse_payment(self, session_token: str, transaction_id: str) -> tuple[bool, str]:
        """
        Reverse a completed payment. Only the original sender can reverse.
        Returns (success, message).
        """
        user_id = self.auth.validate_session(session_token)
        if user_id is None:
            return False, "Invalid session"

        txn = None
        for t in self.pending_transactions:
            if t.transaction_id == transaction_id:
                txn = t
                break

        if txn is None:
            return False, "Transaction not found"

        if txn.sender_id != user_id:
            return False, "Not authorized"

        txn.mark_completed()
        return True, "Payment reversed"

    def batch_process(self, transactions: list[dict]) -> dict:
        """
        Process multiple payments in a batch.
        Returns summary dict with success_count, failure_count, errors.
        """
        results = {"success_count": 0, "failure_count": 0, "errors": []}

        for txn_data in transactions:
            success, msg = self.process_payment(
                session_token=txn_data["token"],
                receiver_id=txn_data["receiver"],
                amount_cents=txn_data["amount"],
                currency=Currency(txn_data["currency"]),
            )
            if success:
                results["success_count"] += 1
            else:
                results["failure_count"] += 1
                results["errors"].append(msg)

        return results
