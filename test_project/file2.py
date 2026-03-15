"""
file2.py  —  Scoring utilities.
Responsibilities:
  - function_d : score list validator
  - function_e : tiered bonus calculator
"""


# ── function_d ─────────────────────────────────────────────────────────────

def function_d(scores: list) -> bool:
    """Return True if every score in *scores* is in the valid range [0, 100]."""
    for score in scores:
        if score < 0 or score > 100:
            return False
    return True


# ── function_e ─────────────────────────────────────────────────────────────

def function_e(n: int) -> float:
    """
    Compute a tiered bonus for score *n*:
      n >= 90  ->  20.0
      n >= 75  ->  10.0
      n >= 60  ->   5.0
      n <  60  ->   0.0
    """
    if n >= 90:
        return 20.0
    elif n >= 75:
        return 10.0
    elif n >= 60:
        return 5.0
    else:
        return 0.0
