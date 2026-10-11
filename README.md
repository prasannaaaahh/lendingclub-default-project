# LendingClub Loan Default Risk Prediction

[![Tests](https://github.com/prasannaaaahh/lendingclub-default-project/actions/workflows/tests.yml/badge.svg)](https://github.com/prasannaaaahh/lendingclub-default-project/actions/workflows/tests.yml)

A machine learning project that predicts loan defaults using information available at approval time, groups borrowers into risk bands, estimates potential losses, and evaluates approval strategies.

## Key findings

- Analysed **589,488 loans** issued from 2012 to 2015.
- The best LightGBM model achieved a **test ROC-AUC of 0.687** when LendingClub's grade, interest rate, and installment were included. Using application-time features alone, it achieved **0.666**.
- The model's five risk bands showed default rates from **4.6% in the lowest-risk band to 28.0% in the highest-risk band**.
- The validation-selected approval cutoff recommended approving all loans under the project's simplified profit calculation. On the 2015 test set, only LendingClub grades F and G had negative aggregate returns.
- Post-issue information substantially inflated performance, highlighting why data leakage must be avoided.

## Problem statement

Can loan default risk be ranked using information available when a loan is approved, and can that ranking help a lender make better approval decisions?

The project examines borrower risk factors, compares application-only features with LendingClub's own risk and pricing information, estimates expected losses, and evaluates approval cutoffs.

## Data

- **Source:** [Lending Club dataset on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **Cohort:** 36-month loans issued from 2012 to 2015.
- **Target:** `Charged Off` = default; `Fully Paid` = repaid.
- **Size:** 589,488 loans after excluding loans without a final outcome.
- Raw data is not included in this repository. Follow [`data/README.md`](data/README.md) to download and prepare it.

## Approach

1. **Explore and audit data:** review the cohort and identify columns that would cause data leakage.
2. **Engineer features:** build a preprocessing pipeline for missing values, categorical variables, and derived financial features.
3. **Train models:** compare logistic regression, random forest, LightGBM, and a dummy baseline.
4. **Evaluate fairly:** use a time-based split—2012–2013 for training, 2014 for validation, and 2015 for testing.
5. **Translate scores into business results:** create risk bands, estimate expected losses, and compare approval strategies using realized profit.

## Results

### Model performance

| Model | Features | Test ROC-AUC |
|---|---|---:|
| Logistic regression | Application-time only | 0.656 |
| Random forest | Application-time only | 0.658 |
| LightGBM | Application-time only | 0.666 |
| LightGBM | Application-time + LendingClub grade, interest rate, and installment | **0.687** |
| LendingClub grade alone | Grade only | 0.667 |
| LendingClub interest rate alone | Interest rate only | 0.678 |


### Risk bands

The model separated loans into five risk bands. The observed default rate increased from **4.57% in the lowest-risk band** to **27.98% in the highest-risk band**.


### Approval strategy

Under the project's simplified profit definition, the cutoff selected using 2014 validation data approved all loans. On the 2015 test set, approving only grades A–C produced less total profit than approving all loans. This is a dataset-specific result, not a general lending recommendation.

## Tools and technologies

- Python, pandas, NumPy
- scikit-learn, LightGBM, imbalanced-learn, SHAP
- Jupyter Notebook
- matplotlib, seaborn
- pytest

## Repository structure

```text
notebooks/    EDA, feature engineering, model training, and business analysis
src/          Feature engineering, evaluation, and business logic
tests/        Unit tests
reports/      Leakage audit, result tables, and figures
data/         Data instructions and local data files (not tracked)
models/       Generated model artifacts
requirements.txt
README.md
```

## How to reproduce

### Requirements

Python 3.13 is the development environment used for this project. Install Git and create a Python virtual environment.

1. Clone the repository and install dependencies:

   ```bash
   git clone https://github.com/prasannaaaahh/lendingclub-default-project.git
   cd lendingclub-default-project
   python -m venv .venv
   ```

   On Windows:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   On macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

   Then install the packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Download and prepare the data using [`data/README.md`](data/README.md).

3. Run the notebooks in order: `01_eda.ipynb`, `02_features.ipynb`, `03_models.ipynb`, and `04_business.ipynb`.

   To execute them from PowerShell:

   ```powershell
   foreach ($n in "01_eda","02_features","03_models","04_business") {
       jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "notebooks/$n.ipynb"
   }
   ```

4. Run the tests:

   ```bash
   pip install pytest
   pytest
   ```

Some model searches, resampling comparisons, and bootstrap analyses may take several minutes.

## Limitations

- Only loans accepted by LendingClub are available; declined applicants are not represented.
- The analysis covers one historical period, and risk changed between years.
- Profit estimates omit costs such as servicing, funding, and collections.
- Results show patterns in historical data and should not be treated as a real-world lending policy.
- No fairness analysis was performed.

## License

See [`LICENSE`](LICENSE) for the MIT license.
