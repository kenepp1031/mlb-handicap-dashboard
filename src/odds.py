from __future__ import annotations


def american_to_implied(odds: int) -> float:
    return (-odds / (-odds + 100)) if odds < 0 else (100 / (odds + 100))


def remove_vig(odds_a: int, odds_b: int) -> tuple[float, float]:
    first, second = american_to_implied(odds_a), american_to_implied(odds_b)
    total = first + second
    return first / total, second / total
