from __future__ import annotations


def american_to_implied(odds: int) -> float:
    return (-odds / (-odds + 100)) if odds < 0 else (100 / (odds + 100))


def probability_to_american(probability: float) -> int:
    if not 0 < probability < 1:
        raise ValueError("Probability must be between 0 and 1.")
    return round(-100 * probability / (1 - probability)) if probability >= 0.5 else round(100 * (1 - probability) / probability)


def remove_vig(odds_a: int, odds_b: int) -> tuple[float, float]:
    first, second = american_to_implied(odds_a), american_to_implied(odds_b)
    total = first + second
    return first / total, second / total


def expected_value_per_dollar(probability: float, american_odds: int) -> float:
    """Expected net profit for a $1 stake (before limits, taxes, and uncertainty)."""
    decimal_profit = (100 / -american_odds) if american_odds < 0 else (american_odds / 100)
    return probability * decimal_profit - (1 - probability)
