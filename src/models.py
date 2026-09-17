from __future__ import annotations

from dataclasses import dataclass
from math import comb


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def binomial_tail(trials: int, probability: float, minimum_successes: int) -> float:
    """Probability of at least minimum_successes in a binomial distribution."""
    probability = clamp(probability, 0.001, 0.999)
    return sum(
        comb(trials, successes)
        * probability**successes
        * (1 - probability) ** (trials - successes)
        for successes in range(minimum_successes, trials + 1)
    )


@dataclass
class StrikeoutProjection:
    expected_ks: float
    over_probability: float


def project_strikeouts(
    expected_batters_faced: float,
    pitcher_k_rate: float,
    opponent_k_rate: float,
    park_k_factor: float = 1.0,
    workload_factor: float = 1.0,
    line: float = 5.5,
) -> StrikeoutProjection:
    """Simple transparent baseline; replace with calibrated negative-binomial model later."""
    matchup_rate = clamp((pitcher_k_rate * 0.60 + opponent_k_rate * 0.40) * park_k_factor, 0.08, 0.45)
    batters = max(1, round(expected_batters_faced * workload_factor))
    expected = batters * matchup_rate
    return StrikeoutProjection(expected, binomial_tail(batters, matchup_rate, int(line) + 1))

