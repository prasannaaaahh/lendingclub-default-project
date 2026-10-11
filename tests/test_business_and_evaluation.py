import numpy as np
import pandas as pd

from business import add_realized_profit, fixed_volume_comparison, profit_by_group
from evaluation import evaluate_scores, ks_statistic


def toy_loans(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    risk = rng.random(n)
    default = (rng.random(n) < risk * 0.5).astype(int)
    funded = np.full(n, 10000.0)
    total_pymnt = np.where(default == 1, 4000.0, 11500.0)
    df = pd.DataFrame({"default": default, "funded_amnt": funded, "total_pymnt": total_pymnt,
                       "int_rate": 10 + 10 * risk})
    return add_realized_profit(df), risk


def test_profit_definition():
    df, _ = toy_loans(10)
    assert (df["profit"] == df["total_pymnt"] - df["funded_amnt"]).all()


def test_ks_and_auc_for_perfect_ranking():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert ks_statistic(y, p) == 1.0
    assert evaluate_scores(y, p)["roc_auc"] == 1.0


def test_profit_by_group_adds_up():
    df, risk = toy_loans()
    df["band"] = pd.qcut(risk, 5, labels=[1, 2, 3, 4, 5])
    tbl = profit_by_group(df, "band")
    assert tbl["loans"].sum() == len(df)
    assert np.isclose(tbl["total_profit_$M"].sum() * 1e6, df["profit"].sum())


def test_good_ranking_beats_random_at_fixed_volume():
    df, risk = toy_loans(5000)
    out = fixed_volume_comparison(df, {"Model": risk}, [0.5], n_random=10)
    model = out[out["policy"] == "Model"].iloc[0]
    rand = out[out["policy"] == "Random"].iloc[0]
    assert model["loans"] == 2500
    assert model["total_profit"] > rand["total_profit"]
    assert model["default_rate_%"] < rand["default_rate_%"]


def test_approving_everything_is_the_same_for_every_policy():
    df, risk = toy_loans(500)
    out = fixed_volume_comparison(df, {"Model": risk}, [1.0], n_random=3)
    assert np.isclose(out["total_profit"].iloc[0], out["total_profit"].iloc[1])
