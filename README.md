# Kroger coupon campaign analysis

This repository analyses the Dunnhumby Complete Journey data to evaluate Kroger coupon campaigns, behavioural customer segments, customer value, campaign lift, and display/mailer support.

## Repository structure

```text
.
├── data/
│   ├── raw/                  # Original CSV files; do not edit in place
│   └── processed/            # Analysis-ready Parquet files and derived metrics
├── src/
│   └── data_cleaning/        # Raw-to-processed preparation and validation
├── analysis.ipynb            # Main coupon-targeting analysis
```

## Recommended workflow

Run commands from the project root. The scripts also resolve the project root from their own file locations, so they do not depend on the current working directory.

1. Prepare and validate the raw data:

   ```bash
   .venv-course-fit/bin/python src/data_cleaning/prepare_data.py
   ```

2. Build the panel-regression dataset:

   ```bash
   .venv-course-fit/bin/python src/data_cleaning/build_regression_panel.py
   ```

3. Open the main analysis:

   ```text
   analysis.ipynb
   ```

   Select the **Kroger Project (.venv-course-fit)** kernel and run all cells.

   To verify it non-interactively from the project root:

   ```bash
   JUPYTER_PATH="$PWD/.venv-course-fit/share/jupyter" \
   MPLCONFIGDIR=/tmp/kroger_mpl \
   .venv-course-fit/bin/python -m jupyter nbconvert \
     --to notebook --execute --inplace analysis.ipynb \
     --ExecutePreprocessor.kernel_name=kroger-project \
     --ExecutePreprocessor.timeout=1200
   ```

## Important notes

- `data/raw/` contains source data and should be treated as immutable.
- `data/processed/` contains analysis-ready datasets and its generated data-quality report.
- `data/processed_data.zip` is the supplied packaged copy of processed outputs. The active analysis reads extracted files from `data/processed/`.

