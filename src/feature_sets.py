# Feature sets for the LendingClub default project.
# A = application-only (no LendingClub risk output).
# B = A + LendingClub's own pricing output (grade, int_rate, installment).

FEATURES_A_RAW = [
    "loan_amnt", "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "addr_state", "dti", "delinq_2yrs", "earliest_cr_line",
    "fico_range_low", "fico_range_high", "inq_last_6mths", "mths_since_last_delinq",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "application_type", "mort_acc", "pub_rec_bankruptcies",
]

LC_OUTPUT_COLS = ["grade", "int_rate", "installment"]

FEATURES_B_RAW = FEATURES_A_RAW + LC_OUTPUT_COLS