from db.connection import ConnectionPool
from db.queries import QueryBuilder
from services.auth import AuthService
from services.payment import PaymentService
from services.notification import NotificationService
from models.transaction import Currency


def create_app(pool_size: int = 10) -> dict:
    """
    Wire up the application. Returns dict of service instances.
    """
    pool = ConnectionPool(max_size=pool_size)
    db = QueryBuilder(pool)
    auth = AuthService(db)
    payment = PaymentService(auth, db)
    notifications = NotificationService(db)

    return {
        "pool": pool,
        "db": db,
        "auth": auth,
        "payment": payment,
        "notifications": notifications,
    }


def run_demo():
    """Demonstrate the payment flow."""
    app = create_app(pool_size=5)
    auth = app["auth"]
    payment = app["payment"]
    notif = app["notifications"]

    # Register users
    auth.register("alice@example.com", "securepass123", "Alice")
    auth.register("bob@example.com", "bobpass456", "Bob")

    # Login
    token_alice, msg = auth.login("alice@example.com", "securepass123")
    token_bob, msg = auth.login("bob@example.com", "bobpass456")

    if token_alice is None:
        print(f"Alice login failed: {msg}")
        return

    # Process payment
    success, msg = payment.process_payment(
        session_token=token_alice,
        receiver_id="bob_user_id",
        amount_cents=1000,
        currency=Currency.EUR,
        description="Lunch payment",
    )
    print(f"Payment: {msg}")

    if success:
        notif.notify_payment_received("bob_user_id", 1000, "Alice")

    # Check balance
    balance, msg = payment.get_balance(token_alice)
    print(f"Alice balance: {balance} cents")



if __name__ == "__main__":
    run_demo()
