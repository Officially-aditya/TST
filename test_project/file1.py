"""
file1.py  —  Data pipeline utilities.
Responsibilities:
  - function_a : frequency counter
  - function_b : trimmed-mean calculator
  - function_c : record filter + bonus enrichment
"""

from file2 import function_e


# ── function_a ─────────────────────────────────────────────────────────────

def function_a(data: list) -> dict:
    """Count occurrences of each unique item in *data*. Returns {item: count}."""
    counts = {}
    for item in data:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 0
    return counts


# ── function_b ─────────────────────────────────────────────────────────────

def function_b(scores: list) -> float:
    """Return the trimmed mean of *scores* (drops the single lowest value)."""
    if not scores:
        return 0.0
    scores = sorted(scores)
    trimmed = scores[1:]
    return sum(trimmed) / len(trimmed)


# ── function_c ─────────────────────────────────────────────────────────────

def function_c(records: list, threshold: int) -> list:
    """
    Filter records whose 'score' key exceeds *threshold*, then enrich
    each with a bonus computed by file2.function_e.
    """
    result = []
    for rec in records:
        if rec["score"] > threshold:
            bonus = function_e(str(rec["score"]))
            result.append({
                "id":    rec["id"],
                "score": rec["score"],
                "bonus": bonus,
            })
    return result
