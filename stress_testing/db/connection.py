import time
from typing import Optional, Any
from dataclasses import dataclass, field


@dataclass
class Connection:
    conn_id: int
    created_at: float = field(default_factory=time.time)
    in_use: bool = False
    last_query_time: Optional[float] = None

    def is_stale(self, max_idle_seconds: float = 300.0) -> bool:
        """Connection is stale if unused for more than max_idle_seconds."""
        if self.last_query_time is None:
            elapsed = time.time() - self.created_at
        else:
            elapsed = time.time() - self.last_query_time
        return elapsed > max_idle_seconds

    def execute(self, query: str, params: tuple = ()) -> list[dict]:
        """
        Execute a SQL query with parameters.
        Returns list of row dicts.
        Raises ConnectionError if connection is stale.
        """
        if self.is_stale():
            raise ConnectionError(f"Connection {self.conn_id} is stale")
        self.last_query_time = time.time()
        # Simulated execution
        return []


class ConnectionPool:
    """
    Thread-safe connection pool.
    acquire() returns a Connection or None if pool is exhausted.
    release(conn) returns a connection to the pool.
    Callers MUST release connections after use.
    """

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.connections: list[Connection] = []
        self.next_id = 0

    def acquire(self) -> Optional[Connection]:
        """
        Acquire a connection from the pool.
        Returns None if all connections are in use and pool is at max capacity.
        Caller MUST call release() when done.
        """
        # Try to find a free connection
        for conn in self.connections:
            if not conn.in_use and not conn.is_stale():
                conn.in_use = True
                return conn

        # Create new connection if under capacity
        if len(self.connections) < self.max_size:
            conn = Connection(conn_id=self.next_id)
            self.next_id += 1
            conn.in_use = True
            self.connections.append(conn)
            return conn

        return None

    def release(self, conn: Connection) -> None:
        """Return a connection to the pool."""
        conn.in_use = False

    def drain_stale(self) -> int:
        """Remove all stale connections. Returns count removed."""
        before = len(self.connections)
        self.connections = [c for c in self.connections if not c.is_stale()]
        return before - len(self.connections)


    def pool_status(self) -> dict:
        """Return pool statistics."""
        in_use = sum(1 for c in self.connections if c.in_use)
        stale = sum(1 for c in self.connections if c.is_stale())
        return {
            "total": len(self.connections),
            "in_use": in_use,
            "available": len(self.connections) - in_use,
            "stale": stale,
            "capacity": self.max_size,
        }
