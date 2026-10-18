"""Temporal scores by event, phase, and axis using indexed time windows.

Dependencies: NumPy and pandas. This module does not query catalogs or change
signal preprocessing. Same-name phases within an event form a single union.
To score same-name branches separately, include arrival_time in the grouping key.
"""

import numpy as np
import pandas as pd


def _to_seconds(value):
    """Accept UTCDateTime, datetime, Timestamp, UTC ISO strings, or Unix seconds."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    stamp = getattr(value, "timestamp", None)
    if stamp is not None:
        return float(stamp() if callable(stamp) else stamp)
    return pd.Timestamp(value).timestamp()


def _window_union_mean(times, activity, centers, half_width):
    """Return the mean and sample count within a union of closed windows.

    Both times and centers must be sorted. Merge index ranges before selecting
    values so overlapping windows never count a sample twice. No
    number_of_samples-by-number_of_arrivals matrix is allocated.
    """
    if len(centers) == 0:
        return np.nan, 0
    left_indices = np.searchsorted(times, centers - half_width, side="left")
    right_indices = np.searchsorted(times, centers + half_width, side="right")
    segments = []
    start = end = None
    for left, right in zip(left_indices, right_indices):
        if left == right:
            continue
        if start is None:
            start, end = left, right
        elif left <= end:
            end = max(end, right)
        else:
            segments.append(activity[start:end])
            start, end = left, right
    if start is None:
        return np.nan, 0
    segments.append(activity[start:end])
    selected = segments[0] if len(segments) == 1 else np.concatenate(segments)
    return float(selected.mean()), len(selected)


def calculate_associations_fast(
    diff_panel,
    time_data,
    sensibility=18,
    time_shift=None,
    *,
    fixed_coverage=False,
    labels=("Z", "X", "Y"),
):
    """Return event/phase/axis scores plus an 'all' row for each axis.

    diff_panel: list of tables containing time and treated_change.
    time_data: table containing event_id, phase, and arrival_time.
    sensibility: window half-width in seconds; the existing name is preserved.
    time_shift: shifts in seconds; zero is never included among the controls.

    fixed_coverage=False preserves the submitted coverage rule: the reference
    checks only the upper boundary; controls check both boundaries and may drop
    arrivals. This separates performance changes from methodological changes.
    fixed_coverage=True fixes the arrivals whose original windows are complete
    and rejects any shift that moves one of those windows outside the record.
    True is recommended for the pending methodological review.

    Both modes apply these corrections: independent counts per phase; controls
    use the same event and phase as the reference; each phase name is processed
    once per event; ties are excluded from the score numerator.

    Samples with nonfinite times or activity are excluded. This does not verify
    internal coverage or fill gaps. Missing samples or controls produce NaN and
    an explanatory status. The signal and preprocessing are not modified.
    """
    if not diff_panel or len(diff_panel) > len(labels):
        raise ValueError("Provide one or more axes and enough corresponding labels.")
    if not np.isfinite(sensibility) or sensibility < 0:
        raise ValueError("sensibility must be finite and nonnegative.")
    shifts = np.asarray(
        np.arange(-10, 10) * 18 if time_shift is None else time_shift,
        dtype=float,
    )
    if shifts.ndim != 1 or not len(shifts):
        raise ValueError("time_shift must be a nonempty one-dimensional sequence.")
    if not np.isfinite(shifts).all():
        raise ValueError("time_shift must contain only finite values.")
    shifts = np.unique(shifts)
    controls = shifts[shifts != 0]
    increments = np.diff(shifts)
    step = (
        float(increments[0])
        if len(increments) and np.all(increments == increments[0])
        else np.nan
    )

    arrivals_data = time_data[["event_id", "phase", "arrival_time"]].copy()
    if arrivals_data.isna().any().any():
        raise ValueError("Arrivals must have an event identifier, phase, and time.")
    arrivals_data["arrival_seconds"] = arrivals_data["arrival_time"].map(_to_seconds)
    if not np.isfinite(arrivals_data["arrival_seconds"].to_numpy(float)).all():
        raise ValueError("Arrival times must be finite.")
    groups = [
        (event_id, phase, np.sort(g["arrival_seconds"].to_numpy(float)))
        for (event_id, phase), g in arrivals_data.groupby(
            ["event_id", "phase"], sort=False
        )
    ]
    groups.append(
        (
            "all",
            "united_phase",
            np.sort(arrivals_data["arrival_seconds"].to_numpy(float)),
        )
    )
    results = []

    for axis, df in zip(labels, diff_panel):
        times = np.fromiter((_to_seconds(t) for t in df["time"]), dtype=float)
        activity = np.abs(df["treated_change"].to_numpy(dtype=float))
        valid = np.isfinite(times) & np.isfinite(activity)
        times, activity = times[valid], activity[valid]
        order = np.argsort(times, kind="stable")
        times, activity = times[order], activity[order]

        for event_id, phase, arrivals in groups:
            less = equal = greater = 0
            control_sample_counts = []
            if len(times):
                lower_limit, upper_limit = times[0], times[-1]
                if fixed_coverage:
                    reference_arrivals = arrivals[
                        (arrivals - sensibility > lower_limit)
                        & (arrivals + sensibility < upper_limit)
                    ]
                else:
                    reference_arrivals = arrivals[arrivals + sensibility < upper_limit]
                observed, n_observed = _window_union_mean(
                    times, activity, reference_arrivals, sensibility
                )
            else:
                reference_arrivals = np.array([], dtype=float)
                observed, n_observed = np.nan, 0

            if n_observed:
                for shift in controls:
                    if fixed_coverage:
                        centers = reference_arrivals + shift
                        if np.any(centers - sensibility <= lower_limit) or np.any(
                            centers + sensibility >= upper_limit
                        ):
                            continue
                    else:
                        centers = arrivals + shift
                        centers = centers[
                            (centers - sensibility > lower_limit)
                            & (centers + sensibility < upper_limit)
                        ]
                    mean, n_control = _window_union_mean(
                        times, activity, centers, sensibility
                    )
                    if not n_control:
                        continue
                    control_sample_counts.append(n_control)
                    if mean < observed:
                        less += 1
                    elif mean > observed:
                        greater += 1
                    else:
                        equal += 1

            total = less + equal + greater
            score = less / total if total else np.nan
            status = (
                "ok" if total else "no_valid_controls" if n_observed else "no_samples"
            )
            results.append(
                {
                    "event_id": event_id,
                    "phase": phase,
                    "axis": axis,
                    "temporal_association_score": score,
                    "shift_upper_limit": float(shifts[-1]),
                    "shift_lower_limit": float(shifts[0]),
                    "step": step,
                    "activity_mean": observed,
                    "n_samples": n_observed,
                    "n_controls": total,
                    "n_less": less,
                    "n_equal": equal,
                    "n_greater": greater,
                    "min_control_samples": min(control_sample_counts) if total else 0,
                    "max_control_samples": max(control_sample_counts) if total else 0,
                    "coverage_mode": "fixed" if fixed_coverage else "current",
                    "status": status,
                }
            )

    return pd.DataFrame(results)


def fast_association_calculator(
    diff_panel,
    time_data,
    sensibility=18,
    time_shift=None,
    *,
    cobertura_fixa=False,
    labels=("Z", "X", "Y"),
):
    """Backward-compatible wrapper for the previously supplied function name."""
    return calculate_associations_fast(
        diff_panel,
        time_data,
        sensibility=sensibility,
        time_shift=time_shift,
        fixed_coverage=cobertura_fixa,
        labels=labels,
    )
