"""Paired forecast comparisons with date clusters and moving-week blocks.

Blocks count retained forecast dates, not uninterrupted calendar weeks. All
assets at a selected date stay together. Sign-flip tests assume independent,
symmetric date-level differences; they are sensitivity checks, not randomized
experiment evidence or a remedy for adaptive reuse of a historical test set.
"""
from __future__ import annotations

from datetime import date
import math
import numpy as np

BLOCK_LENGTHS = (1, 2, 4, 8)
PRIMARY_BLOCK_LENGTH = 4


def _groups(values, dates):
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    dates = np.asarray(dates, dtype=str)
    if x.ndim != 2 or len(x) == 0 or dates.shape != (len(x),):
        raise ValueError("Expected nonempty paired values and one date per row")
    if not np.isfinite(x).all():
        raise ValueError("Paired values must be finite")
    unique, inverse = np.unique(dates, return_inverse=True)
    sums = np.zeros((len(unique), x.shape[1]))
    counts = np.bincount(inverse)
    # Canonical date/value ordering gives stable sums without quadratic scans.
    order = np.lexsort(tuple(x[:, j] for j in reversed(range(x.shape[1]))) + (dates,))
    np.add.at(sums, inverse[order], x[order])
    return x, unique, inverse, sums, counts


def _sample_dates(n_dates, rng, resamples, block_length):
    if resamples < 1 or block_length < 1:
        raise ValueError("Resamples and block length must be positive")
    length = min(block_length, n_dates)
    starts = rng.integers(0, n_dates, (resamples, math.ceil(n_dates / length)))
    return ((starts[:, :, None] + np.arange(length)) % n_dates).reshape(
        resamples, -1
    )[:, :n_dates]


def block_bootstrap_means(values, dates, *, resamples=2000, seed=0, block_length=4):
    """Circular moving blocks of consecutive, sorted observed forecast dates."""
    _, unique, _, sums, counts = _groups(values, dates)
    selected = _sample_dates(len(unique), np.random.default_rng(seed), resamples, block_length)
    return sums[selected].sum(axis=1) / counts[selected].sum(axis=1)[:, None]


def block_bootstrap_indices(dates, rng, *, resamples=2000, block_length=1):
    """Equivalent row-index samples, preserving every ticker within a date."""
    dates = np.asarray(dates, dtype=str)
    if len(dates) == 0:
        raise ValueError("No dates to bootstrap")
    unique = np.unique(dates)
    groups = [np.flatnonzero(dates == d) for d in unique]
    return [np.concatenate([groups[j] for j in selected]) for selected in
            _sample_dates(len(unique), rng, resamples, block_length)]


def percentile_ci(values):
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def exact_mcnemar(wins: int, losses: int) -> float:
    """Two-sided exact discordant-pair test; row independence is required."""
    if min(wins, losses) < 0 or int(wins) != wins or int(losses) != losses:
        raise ValueError("Discordant counts must be nonnegative integers")
    wins, losses = int(wins), int(losses)
    n = wins + losses
    if n == 0:
        return 1.0
    numerator = sum(math.comb(n, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2 * numerator / (1 << n))


def paired_week_signflip(differences, dates, *, seed=0, resamples=10000):
    """Flip whole date sums; enumerate all flips when <=20 dates differ."""
    _, _, _, sums, _ = _groups(differences, dates)
    active = sums[:, 0]
    active = active[np.abs(active) > 1e-12]
    observed = abs(active.sum())
    if len(active) <= 20:
        totals = np.zeros(1)
        for value in active:
            totals = np.concatenate((totals + value, totals - value))
        p = float((np.abs(totals) >= observed - 1e-12).mean())
        exact, draws = True, len(totals)
    else:
        rng = np.random.default_rng(seed)
        exceed = 0
        for start in range(0, resamples, 1000):
            count = min(1000, resamples - start)
            signs = 2 * rng.integers(0, 2, (count, len(active))) - 1
            exceed += int((np.abs((signs * active).sum(axis=1)) >= observed - 1e-12).sum())
        p = (exceed + 1) / (resamples + 1)
        exact, draws = False, resamples
    return {"p_two_sided": p, "exact": exact, "nonzero_date_clusters": len(active),
            "resamples": draws,
            "assumption": "Independent symmetric date-level paired differences; serial dependence is not removed by sign flips"}


def paired_inference(differences, dates, *, resamples=2000, seed=0,
                     block_lengths=BLOCK_LENGTHS, signflip_resamples=10000):
    """Positive differences mean improvement; compare paired rows on the same dates."""
    x, unique, _, _, _ = _groups(differences, dates)
    if x.shape[1] != 1:
        raise ValueError("paired_inference expects a one-dimensional difference")
    intervals = {str(length): percentile_ci(block_bootstrap_means(
        x, dates, resamples=resamples, seed=seed, block_length=length
    )[:, 0]) for length in block_lengths}
    parsed = [date.fromisoformat(d) for d in unique]
    gaps = [(b - a).days for a, b in zip(parsed[:-1], parsed[1:])]
    return {"mean_difference": float(np.sort(x[:, 0]).mean()), "n": len(x), "date_clusters": len(unique),
            "ci95_by_block_length": intervals,
            "week_signflip": paired_week_signflip(x[:, 0], dates, seed=seed, resamples=signflip_resamples),
            "bootstrap_resamples": resamples, "seed": seed,
            "bootstrap": "Circular moving blocks of consecutive retained forecast dates, all tickers together; row-weighted statistic",
            "block_unit": "retained forecast weeks; holiday gaps can span more calendar weeks",
            "first_date": str(unique[0]), "last_date": str(unique[-1]),
            "calendar_gaps_over_7_days": sum(gap > 7 for gap in gaps),
            "largest_calendar_gap_days": max(gaps, default=0)}


def probability_metrics(probabilities, labels, labels_order=("Up", "Flat", "Down")):
    """Validated multiclass accuracy, Brier score and log loss."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=str)
    order = np.asarray(labels_order, dtype=str)
    if p.shape != (len(y), len(order)) or len(y) == 0:
        raise ValueError("Probability matrix shape does not match labels")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=2e-5, rtol=0):
        raise ValueError("Expected finite nonnegative probabilities summing to one")
    if not np.isin(y, order).all() or len(np.unique(order)) != len(order):
        raise ValueError("Unknown or duplicate class labels")
    Y = (y[:, None] == order[None, :]).astype(float)
    hits = (order[p.argmax(1)] == y).astype(float)
    brier = ((p - Y) ** 2).sum(1)
    logloss = -np.log(np.clip((p * Y).sum(1), 1e-15, 1))
    return {"hit_rate": float(hits.mean()), "brier": float(brier.mean()),
            "log_loss": float(logloss.mean()), "hit_rows": hits, "brier_rows": brier,
            "log_loss_rows": logloss}


def holm_adjust(pvalues):
    """Holm step-down adjustment for an explicitly declared finite family."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Expected finite p-values in [0, 1]")
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted.tolist()


def inference_metadata():
    return {"primary_reference": "Training-selected majority class and training-fitted class frequencies",
            "primary_block_length": PRIMARY_BLOCK_LENGTH, "sensitivity_block_lengths": list(BLOCK_LENGTHS),
            "oracle_reference": "Evaluation-majority benchmark is descriptive and uses future labels; no inferential CI or success decision is based on it",
            "bss_degeneracy": "Undefined BSS resamples with zero reference loss are omitted; all-degenerate intervals are null",
            "paired_comparisons": "All forecast differences share the same labels and dates",
            "historical_status": "Exploratory: experiments 02-04 adaptively reuse the same historical test period; no untouched confirmatory test",
            "multiplicity": "Unadjusted exploratory intervals and paired tests, not familywise discovery guarantees",
            "training_uncertainty": "Intervals condition on trained predictions and reference priors; models and calibration are not refitted within resamples",
            "bootstrap_assumption": "Approximate stationarity within evaluated period; report 1/2/4/8 retained-week sensitivity",
            "signflip_assumption": "Symmetry and independence of date aggregates; exact enumeration does not remove these assumptions"}
