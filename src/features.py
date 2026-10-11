# Feature engineering and preprocessing for the LendingClub default project.
# Everything the model needs is built here, so every notebook uses identical steps.

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from feature_sets import FEATURES_A_RAW, FEATURES_B_RAW

GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

NUMERIC_A = [
    "loan_amnt", "log_annual_inc", "loan_to_income", "dti", "fico_mid",
    "credit_history_years", "emp_length_years", "delinq_2yrs", "inq_last_6mths",
    "mths_since_last_delinq", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "mort_acc", "pub_rec_bankruptcies",
]
NUMERIC_B_EXTRA = ["int_rate", "grade_num", "installment_to_income"]
CATEGORICAL = ["home_ownership", "verification_status", "purpose", "addr_state", "application_type"]

# (lower percentile, upper percentile) used to cap extreme values; learned on the training set only
CAP_RULES = {
    "log_annual_inc": (0.01, 0.99),
    "loan_to_income": (0.0, 0.99),
    "installment_to_income": (0.0, 0.99),
    "revol_bal": (0.0, 0.99),
    "revol_util": (0.0, 0.999),
    "total_acc": (0.0, 0.99),
    "open_acc": (0.0, 0.99),
}


def get_X(df, feature_set):
    """Return the raw input columns for feature set 'A' or 'B' (issue_d is a helper for credit history)."""
    cols = FEATURES_A_RAW if feature_set == "A" else FEATURES_B_RAW
    return df[["issue_d"] + cols].copy()


def parse_emp_length(s):
    """'< 1 year' -> 0, '1 year' -> 1, ..., '10+ years' -> 10, missing stays missing."""
    s = s.astype(object).astype(str)
    s = s.str.replace("< 1 year", "0", regex=False)
    s = s.str.replace("10+ years", "10", regex=False)
    return pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce").astype("float64")


def engineer_features(df):
    """Row-by-row feature creation. Nothing is learned from data here, so it is safe on any split."""
    out = df.copy()
    out["emp_length_years"] = parse_emp_length(out["emp_length"])

    issue = pd.to_datetime(out["issue_d"])
    earliest = pd.to_datetime(out["earliest_cr_line"], format="%b-%Y")
    out["credit_history_years"] = (issue - earliest).dt.days / 365.25

    out["fico_mid"] = (out["fico_range_low"] + out["fico_range_high"]) / 2
    out["log_annual_inc"] = np.log1p(out["annual_inc"])

    income = out["annual_inc"].replace(0, np.nan)
    out["loan_to_income"] = out["loan_amnt"] / income
    if "installment" in out.columns:
        out["installment_to_income"] = out["installment"] / (income / 12)   # monthly payment / monthly income
    if "grade" in out.columns:
        out["grade_num"] = out["grade"].map(GRADE_MAP).astype("float64")
    return out


class PercentileCapper(BaseEstimator, TransformerMixin):
    """Caps chosen columns at percentiles learned from the data it is fitted on (the training set)."""

    def __init__(self, rules=None):
        self.rules = rules

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.bounds_ = {}
        for col in self.columns_:
            if self.rules and col in self.rules:
                lo_q, hi_q = self.rules[col]
                self.bounds_[col] = (X[col].quantile(lo_q), X[col].quantile(hi_q))
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, (lo, hi) in self.bounds_.items():
            X[col] = X[col].clip(lower=lo, upper=hi)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_, dtype=object)


def build_preprocessing_pipeline(feature_set="A", scale=False):
    """
    Full preprocessing: engineer features -> cap outliers -> fill gaps (+ missing flags) -> one-hot encode.
    feature_set: 'A' (application-only) or 'B' (A + LendingClub grade, interest rate, installment).
    scale: True adds standard scaling of numeric columns (needed for logistic regression).
    """
    numeric_cols = NUMERIC_A + (NUMERIC_B_EXTRA if feature_set == "B" else [])

    numeric_steps = [
        ("cap", PercentileCapper(rules=CAP_RULES)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    categorical_steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False)),
    ]

    columns = ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_cols),
            ("cat", Pipeline(categorical_steps), CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipe = Pipeline([
        ("engineer", FunctionTransformer(engineer_features, validate=False)),
        ("columns", columns),
    ])
    pipe.set_output(transform="pandas")
    return pipe