# Data

Data files are not stored in this repository.

1. Download the Kaggle dataset **Lending Club** (wordsforthewise) and place the accepted-loans file here:

   ```
   data/raw/accepted_2007_to_2018Q4.csv.gz
   ```

   The rejected-loans file is not used.

2. Place the LendingClub data dictionary here (notebook 01 uses it to check every kept column):

   ```
   data/raw/LCDataDictionary.xlsx
   ```

Notebooks write intermediate files to `data/processed/` (created automatically): `cohort.parquet`, `model_data.parquet`, `business_data.parquet` and the saved predictions.
