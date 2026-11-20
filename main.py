"""Offline, count-aware forecasts for Exophase trophy rates.

The input is a sequence of periodic page snapshots. Repeated snapshots are
not independent evidence: Exophase refreshes a title's counts in batches.
This model therefore works on distinct page states, estimates tracked-player
and trophy-earner growth in log-time, and shrinks unstable trophy-specific
trends toward the cross-trophy trend.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
NORMAL_Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)
EPS_PCT = 0.01


def clamp(x, lo, hi):
    return min(hi, max(lo, x))


def logit_pct(x):
    p = clamp(float(x), EPS_PCT, 100.0 - EPS_PCT) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(x):
    # Avoid overflow even for malformed or extreme trend estimates.
    x = clamp(x, -20.0, 20.0)
    return 100.0 / (1.0 + math.exp(-x))


def weighted_line(xs, ys, weights):
    """Weighted least-squares slope/intercept, with safe zero-variance handling."""
    sw = sum(weights)
    if sw <= 0 or len(xs) < 2:
        return (0.0, ys[-1] if ys else 0.0)
    mx = sum(w * x for x, w in zip(xs, weights)) / sw
    my = sum(w * y for y, w in zip(ys, weights)) / sw
    den = sum(w * (x - mx) ** 2 for x, w in zip(xs, weights))
    if den < 1e-12:
        return (0.0, my)
    slope = sum(w * (x - mx) * (y - my)
                for x, y, w in zip(xs, ys, weights)) / den
    return slope, my - slope * mx


def snapshot_rows(case):
    """Discard duplicate page states and retain the latest timestamp for each."""
    trophies = case["trophies"]
    raw = []
    for read in case["history"]:
        rates = read.get("rates", [])
        if len(rates) != len(trophies):
            continue
        players = max(1, int(read.get("players_tracked") or 1))
        stamp = read.get("at", read.get("slot", case["at"]))
        try:
            hours = max(0.01, (parse_time(stamp) - parse_time(case["game"].get("release_date", "2026-11-19"))).total_seconds() / 3600.0)
        except (ValueError, TypeError, OverflowError):
            hours = float(len(raw) + 1)
        state = (players, tuple(None if x is None else round(float(x), 2) for x in rates))
        raw.append((hours, players, rates, state))
    # Keep changes only; this prevents a long unchanged polling interval from
    # making a flat state appear to be many independent observations.
    distinct = []
    for row in raw:
        if not distinct or row[3] != distinct[-1][3]:
            distinct.append(row)
        else:
            distinct[-1] = row
    return distinct


def parse_time(text):
    from datetime import datetime, timezone
    value = text.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def elapsed_hours(case, stamp):
    try:
        return max(0.01, (parse_time(stamp) - parse_time(case["game"].get("release_date", "2026-11-19"))).total_seconds() / 3600.0)
    except (ValueError, TypeError, OverflowError):
        return 1.0


def fit_growth(xs, ys, weights):
    """Fit recent log-value against log elapsed time, downweighting old changes."""
    if len(xs) < 2:
        return 0.0, 0.0
    # More emphasis on recent refresh-to-refresh movement; each page state is
    # still only one observation regardless of how many polls repeated it.
    w = [weights[i] * (0.65 + 0.35 * (i + 1) / len(xs)) for i in range(len(xs))]
    slope, intercept = weighted_line(xs, ys, w)
    return slope, intercept


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    if not trophies or not history:
        raise ValueError("case must contain trophies and at least one read")
    snapshots = snapshot_rows(case)
    last = history[-1]
    last_rates = last.get("rates", [])
    if len(last_rates) != len(trophies):
        raise ValueError("last read does not align with trophies")

    # Fit the growth of the tracked denominator. Use only actual changes in
    # player count and cap the extrapolation so one anomalous refresh cannot
    # imply explosive, unbounded audience growth.
    p_rows = []
    for hours, players, _, _ in snapshots:
        if not p_rows or players != p_rows[-1][1]:
            p_rows.append((hours, players))
    target_hours = elapsed_hours(case, case.get("target", case["at"]))
    last_hours = max(0.01, snapshots[-1][0]) if snapshots else 1.0
    if len(p_rows) >= 2:
        px = [math.log(max(1.0, h)) for h, _ in p_rows]
        py = [math.log(p) for _, p in p_rows]
        beta_p, _ = fit_growth(px, py, [1.0] * len(px))
        beta_p = clamp(beta_p, 0.0, 2.0)
    else:
        beta_p = 0.55
    beta_p = 0.7 * beta_p + 0.3 * 0.55
    p_last = max(1, int(last.get("players_tracked") or 1))

    # For each trophy, fit earned-count growth E ~= tracked_players * rate.
    # Counts rather than raw rates make denominator dilution explicit. Shrink
    # each noisy slope toward the broad lifecycle rate learned across trophies.
    lx = math.log(max(1.0, target_hours) / last_hours)
    per_trophy = []
    for j, trophy in enumerate(trophies):
        xs, ys, weights = [], [], []
        for i, (hours, players, rates, _) in enumerate(snapshots):
            rate = rates[j]
            if rate is None:
                continue
            # Rounding to two decimals can represent a handful of earners as
            # zero; a half-earner continuity correction stabilizes log counts.
            earned = max(0.5, players * clamp(float(rate), 0.0, 100.0) / 100.0)
            xs.append(math.log(max(1.0, hours)))
            ys.append(math.log(earned))
            weights.append(1.0 + 0.15 * i)
        if len(xs) >= 2:
            slope_e, _ = fit_growth(xs, ys, weights)
            slope_e = clamp(slope_e, -0.25, 2.5)
        else:
            slope_e = beta_p
        per_trophy.append(slope_e)

    if per_trophy:
        # A winsorized central lifecycle rate avoids trophy-specific noise
        # determining the direction of all forecasts.
        sorted_slopes = sorted(per_trophy)
        lo = sorted_slopes[int(0.15 * (len(sorted_slopes) - 1))]
        hi = sorted_slopes[int(0.85 * (len(sorted_slopes) - 1))]
        global_e = sum(clamp(s, lo, hi) for s in per_trophy) / len(per_trophy)
    else:
        global_e = beta_p

    # Calibrate uncertainty from between-trophy trend dispersion while keeping
    # the interval intentionally sharp: most uncertainty is shared across the
    # whole game and does not justify a broad random-walk band per trophy.
    deviations = sorted(abs(s - global_e) for s in per_trophy)
    median_dev = deviations[len(deviations) // 2] if deviations else 0.0
    horizon_days = max(0.0, float(case.get("horizon_minutes", 0)) / 1440.0)
    history_strength = min(1.0, max(0.0, (len(snapshots) - 1) / 5.0))
    sigma = clamp(0.012 + 0.003 * math.sqrt(horizon_days)
                  + 0.012 * median_dev * min(lx, 1.5) * history_strength,
                  0.012, 0.09)

    rows = []
    for j, trophy in enumerate(trophies):
        rate = last_rates[j]
        if rate is None:
            rate = 0.01
        rate = clamp(float(rate), 0.01, 99.99)
        if float(last_rates[j] or 0.0) >= 99.99:
            rows.append({"id": trophy["id"], "q": [99.99] * len(LEVELS)})
            continue
        observed_earned = max(0.5, p_last * rate / 100.0)
        slope_e = per_trophy[j] if j < len(per_trophy) else global_e
        # Hierarchical shrinkage is strongest for sparse histories and rare
        # achievements; the data never makes a probability exceed [0,100].
        count_strength = observed_earned / (observed_earned + 18.0)
        points = float(trophy.get("points", 15) or 15)
        tier_strength = 1.0 if points >= 90 else 0.78
        individual_weight = count_strength * tier_strength
        slope = individual_weight * slope_e + (1.0 - individual_weight) * global_e
        slope = clamp(slope, -0.15, 2.0)
        # Scale only the estimated *rate-vs-denominator* trend, not the
        # absolute earners slope. With flat/duplicate launch snapshots there
        # is no evidence to extrapolate a trophy-specific rate change.
        trend_reliability = history_strength
        median_logit = logit_pct(rate) + (slope - beta_p) * lx * trend_reliability
        # Mild ceiling-aware shrinkage stops a transient early rise from
        # projecting impossible completion rates.
        if median_logit > 0.0:
            median_logit *= 0.92
        q = [round(inv_logit_pct(median_logit + z * sigma), 4) for z in NORMAL_Z]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
