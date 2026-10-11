"""Business-layer helpers: realized profit, profit by group, and fixed-volume policy comparison.

Realized profit per loan = total payments received (principal + interest + late fees + recoveries)
minus the amount funded. No servicing fees, funding cost or discounting.
"""
import numpy as np
import pandas as pd


def add_realized_profit(df):
    """Return a copy of df with a 'profit' column (total_pymnt - funded_amnt)."""
    out = df.copy()
    out["profit"] = out["total_pymnt"] - out["funded_amnt"]
    return out


def _summary(df, mask):
    sub = df[mask]
    funded = sub["funded_amnt"].sum()
    return {
        "loans": int(len(sub)),
        "default_rate_%": float(sub["default"].mean() * 100) if len(sub) else np.nan,
        "total_profit": float(sub["profit"].sum()),
        "roi_%": float(sub["profit"].sum() / funded * 100) if funded else np.nan,
    }


def profit_by_group(df, group_col):
    """Realized profit and return on funded capital for each value of group_col."""
    rows = []
    for key, sub in df.groupby(group_col, observed=True):
        funded = sub["funded_amnt"].sum()
        rows.append({
            group_col: key,
            "loans": int(len(sub)),
            "funded_$M": funded / 1e6,
            "default_rate_%": sub["default"].mean() * 100,
            "avg_int_rate_%": sub["int_rate"].mean() if "int_rate" in sub else np.nan,
            "total_profit_$M": sub["profit"].sum() / 1e6,
            "roi_%": sub["profit"].sum() / funded * 100,
        })
    return pd.DataFrame(rows)


def fixed_volume_comparison(df, rankings, approval_rates, n_random=20, seed=42):
    """Compare ranking methods at the same approval volume.

    rankings: dict name -> array of risk scores aligned with df (higher = riskier).
    For each approval rate, the lowest-risk share of loans under each ranking is approved.
    A random policy (averaged over n_random draws) is added as the no-skill reference.
    Requires columns: profit, funded_amnt, default.
    """
    df = df.reset_index(drop=True)
    n = len(df)
    rng = np.random.default_rng(seed)
    rows = []
    for rate in approval_rates:
        k = int(round(n * rate))
        for name, score in rankings.items():
            order = np.argsort(np.asarray(score, dtype=float), kind="stable")
            mask = np.zeros(n, dtype=bool)
            mask[order[:k]] = True
            rows.append({"policy": name, "approval_rate_%": rate * 100, **_summary(df, mask)})
        draws = []
        for _ in range(n_random):
            mask = np.zeros(n, dtype=bool)
            mask[rng.permutation(n)[:k]] = True
            draws.append(_summary(df, mask))
        rows.append({"policy": "Random", "approval_rate_%": rate * 100,
                     **{key: float(np.mean([d[key] for d in draws])) for key in draws[0]}})
    return pd.DataFrame(rows)
