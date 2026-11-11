# Data setup

The data files are not stored in Git because several exceed GitHub's 100 MB per-file limit and redistribution remains subject to the original Dunnhumby Complete Journey data terms.

Obtain the dataset through an authorised Kaggle account:

**[Dunnhumby – The Complete Journey](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey)**

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

From the repository root, generate all analysis-ready files with one command:

```bash
.venv-course-fit/bin/python src/data_cleaning/prepare_data.py
```

The script creates `data/processed/`, including the campaign, promotion, store-profile and regression-panel Parquet files required by `analysis.ipynb`.

