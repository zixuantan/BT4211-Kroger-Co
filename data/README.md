# Data setup

The data files are not stored in Git because several exceed GitHub's 100 MB per-file limit and redistribution remains subject to the original Dunnhumby Complete Journey data terms.

Create `data/raw/` and place these original CSV files inside it:

```text
campaign_desc.csv
campaign_table.csv
causal_data.csv
coupon.csv
coupon_redempt.csv
hh_demographic.csv
product.csv
transaction_data.csv
```

From the repository root, generate all analysis-ready files with:

```bash
.venv-course-fit/bin/python src/data_cleaning/prepare_data.py
.venv-course-fit/bin/python src/data_cleaning/build_regression_panel.py
```

The scripts create `data/processed/`, including the campaign, promotion, store-profile and regression-panel Parquet files required by `analysis.ipynb`.

