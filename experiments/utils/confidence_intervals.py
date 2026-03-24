from __future__ import annotations

from statistics import NormalDist
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from statsmodels.stats.proportion import proportion_confint as _proportion_confint
except Exception:  # pragma: no cover
    _proportion_confint = None


def wilson_confint(successes: int, nobs: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if nobs <= 0:
        raise ValueError("nobs must be > 0.")
    if successes < 0 or successes > nobs:
        raise ValueError("successes must be between 0 and nobs.")

    if _proportion_confint is not None:
        low, high = _proportion_confint(successes, nobs, method="wilson", alpha=alpha)
        return float(low), float(high)

    p_hat = successes / nobs
    z = NormalDist().inv_cdf(1 - alpha / 2)
    z2 = z * z
    denominator = 1 + z2 / nobs
    center = (p_hat + z2 / (2 * nobs)) / denominator
    margin = (
        z
        * np.sqrt((p_hat * (1 - p_hat) / nobs) + (z2 / (4 * nobs * nobs)))
        / denominator
    )
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def bootstrap_mean_confint(
    values: Iterable[float],
    alpha: float = 0.05,
    n_resamples: int = 10000,
    random_state: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1.")

    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("Cannot compute confidence interval on empty values.")
    if arr.size == 1:
        v = float(arr[0])
        return v, v

    rng = np.random.default_rng(random_state)
    samples = rng.choice(arr, size=(n_resamples, arr.size), replace=True)
    sample_means = samples.mean(axis=1)
    lower, upper = np.quantile(sample_means, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


def metric_mean_and_ci(
    values: Iterable[float],
    metric_name: str,
    alpha: float = 0.05,
    n_resamples: int = 10000,
    random_state: int = 42,
    em_metrics: Sequence[str] = ("em",),
) -> tuple[float, list[float]]:
    """Return mean metric value and [lower, upper] confidence interval."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("Cannot summarize empty values.")

    mean_value = float(arr.mean())
    if metric_name in em_metrics:
        rounded = np.rint(arr).astype(int)
        if not np.allclose(arr, rounded):
            raise ValueError(
                f"EM values must be binary (0/1) to apply Wilson CI, got '{metric_name}'."
            )
        low, high = wilson_confint(int(rounded.sum()), int(rounded.size), alpha=alpha)
    else:
        low, high = bootstrap_mean_confint(
            arr,
            alpha=alpha,
            n_resamples=n_resamples,
            random_state=random_state,
        )

    return mean_value, [float(low), float(high)]


def summarize_metrics_with_ci(
    df: pd.DataFrame,
    row_values: Sequence[str],
    metrics: Sequence[str],
    column_name_fn: Callable[[str, str], str],
    alpha: float = 0.05,
    n_resamples: int = 10000,
    random_state: int = 42,
    em_metrics: Sequence[str] = ("em",),
    ci_suffix: str = "_ci",
) -> pd.DataFrame:
    """
    Build a table with metric columns and adjacent CI columns.

    Output format per metric: `<metric>`, `<metric><ci_suffix>` where the CI column
    is a `[lower, upper]` list.
    """
    out: dict[str, list[object]] = {}

    for metric_idx, metric in enumerate(metrics):
        means: list[float] = []
        cis: list[list[float]] = []

        for row_idx, row_value in enumerate(row_values):
            column_name = column_name_fn(row_value, metric)
            if column_name not in df.columns:
                raise KeyError(f"Missing column '{column_name}' in input dataframe.")

            seed = random_state + metric_idx * 10_000 + row_idx
            mean_value, ci = metric_mean_and_ci(
                df[column_name].dropna().to_numpy(),
                metric_name=metric,
                alpha=alpha,
                n_resamples=n_resamples,
                random_state=seed,
                em_metrics=em_metrics,
            )
            means.append(mean_value)
            cis.append(ci)

        out[metric] = means
        out[f"{metric}{ci_suffix}"] = cis

    return pd.DataFrame(out, index=list(row_values))


def summarize_metrics_with_ci_from_values(
    row_values: Sequence[str],
    metrics: Sequence[str],
    values_fn: Callable[[str, str], Iterable[float]],
    alpha: float = 0.05,
    n_resamples: int = 10000,
    random_state: int = 42,
    em_metrics: Sequence[str] = ("em",),
    ci_suffix: str = "_ci",
) -> pd.DataFrame:
    """
    Build a metric + CI table from a custom value getter.

    `values_fn(row_value, metric)` must return the raw samples for that cell.
    """
    out: dict[str, list[object]] = {}

    for metric_idx, metric in enumerate(metrics):
        means: list[float] = []
        cis: list[list[float]] = []

        for row_idx, row_value in enumerate(row_values):
            seed = random_state + metric_idx * 10_000 + row_idx
            mean_value, ci = metric_mean_and_ci(
                values_fn(row_value, metric),
                metric_name=metric,
                alpha=alpha,
                n_resamples=n_resamples,
                random_state=seed,
                em_metrics=em_metrics,
            )
            means.append(mean_value)
            cis.append(ci)

        out[metric] = means
        out[f"{metric}{ci_suffix}"] = cis

    return pd.DataFrame(out, index=list(row_values))


def round_metrics_with_ci_table(
    table: pd.DataFrame,
    decimals: int = 3,
    ci_suffix: str = "_ci",
) -> pd.DataFrame:
    """Round metric columns and [lower, upper] CI columns for cleaner tables."""
    rounded = table.copy(deep=True)

    for col in rounded.columns:
        if col.endswith(ci_suffix):
            rounded[col] = rounded[col].apply(
                lambda ci: [
                    round(float(ci[0]), decimals),
                    round(float(ci[1]), decimals),
                ]
                if isinstance(ci, (list, tuple, np.ndarray)) and len(ci) == 2
                else ci
            )
        elif pd.api.types.is_numeric_dtype(rounded[col]):
            rounded[col] = rounded[col].round(decimals)

    return rounded
