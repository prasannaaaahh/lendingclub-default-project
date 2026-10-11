import numpy as np
import pandas as pd

from feature_sets import FEATURES_A_RAW, FEATURES_B_RAW
from features import (PercentileCapper, build_preprocessing_pipeline,
                      engineer_features, parse_emp_length)

# Columns that only exist after a loan is issued (see reports/leakage_audit.md)
POST_ISSUE = {
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "out_prncp", "out_prncp_inv", "last_credit_pull_d", "last_fico_range_high",
    "last_fico_range_low", "funded_amnt", "funded_amnt_inv", "pymnt_plan", "loan_status",
    "debt_settlement_flag",
}


def make_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "issue_d": pd.to_datetime("2013-06-01"),
        "loan_amnt": rng.integers(1000, 35000, n).astype(float),
        "emp_length": rng.choice(["< 1 year", "1 year", "5 years", "10+ years", None], n),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
        "annual_inc": rng.integers(20000, 200000, n).astype(float),
        "verification_status": rng.choice(["Verified", "Not Verified"], n),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "other"], n),
        "addr_state": rng.choice(["CA", "NY", "TX"], n),
        "dti": rng.uniform(0, 40, n),
        "delinq_2yrs": rng.integers(0, 3, n).astype(float),
        "earliest_cr_line": rng.choice(["Jan-1999", "Mar-2005", "Aug-2010"], n),
        "fico_range_low": rng.integers(660, 800, n).astype(float),
        "inq_last_6mths": rng.integers(0, 5, n).astype(float),
        "mths_since_last_delinq": np.where(rng.random(n) < 0.5, np.nan, rng.integers(1, 60, n)),
        "open_acc": rng.integers(2, 25, n).astype(float),
        "pub_rec": rng.integers(0, 2, n).astype(float),
        "revol_bal": rng.integers(0, 60000, n).astype(float),
        "revol_util": rng.uniform(0, 110, n),
        "total_acc": rng.integers(5, 60, n).astype(float),
        "application_type": "Individual",
        "mort_acc": rng.integers(0, 4, n).astype(float),
        "pub_rec_bankruptcies": rng.integers(0, 2, n).astype(float),
        "grade": rng.choice(list("ABCDE"), n),
        "int_rate": rng.uniform(6, 25, n),
        "installment": rng.uniform(50, 1200, n),
    })
    df["fico_range_high"] = df["fico_range_low"] + 4
    return df


def test_feature_sets_contain_no_post_issue_columns():
    assert not set(FEATURES_B_RAW) & POST_ISSUE


def test_set_a_has_no_lendingclub_risk_output():
    assert not set(FEATURES_A_RAW) & {"grade", "sub_grade", "int_rate", "installment"}


def test_parse_emp_length():
    out = parse_emp_length(pd.Series(["< 1 year", "1 year", "7 years", "10+ years", None]))
    assert out.iloc[:4].tolist() == [0, 1, 7, 10]
    assert np.isnan(out.iloc[4])


def test_credit_history_is_measured_at_issue_date():
    df = make_df(5)
    df["issue_d"] = pd.Timestamp("2012-01-01")
    df["earliest_cr_line"] = "Jan-2002"
    eng = engineer_features(df)
    assert np.allclose(eng["credit_history_years"], 10.0, atol=0.01)


def test_percentile_capper_uses_training_bounds_only():
    train = pd.DataFrame({"x": np.arange(101, dtype=float)})
    capper = PercentileCapper(rules={"x": (0.0, 0.99)}).fit(train)
    capped = capper.transform(pd.DataFrame({"x": [5.0, 1000.0]}))
    assert capped["x"].tolist() == [5.0, 99.0]


def test_pipeline_fits_on_train_and_transforms_new_data_consistently():
    train, other = make_df(300, seed=1), make_df(120, seed=2)
    for fs in ("A", "B"):
        pipe = build_preprocessing_pipeline(fs)
        cols = FEATURES_A_RAW if fs == "A" else FEATURES_B_RAW
        X_tr = pipe.fit_transform(train[["issue_d"] + cols])
        X_ot = pipe.transform(other[["issue_d"] + cols])
        assert list(X_tr.columns) == list(X_ot.columns)
        assert X_tr.isna().sum().sum() == 0 and X_ot.isna().sum().sum() == 0
