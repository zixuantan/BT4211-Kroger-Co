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

Run commands from the project root. The scripts resolve the project root from their own file locations, so they do not depend on the current working directory.

### 1. Create the Python environment

```bash
python3 -m venv .venv-course-fit
source .venv-course-fit/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install \
  --prefix .venv-course-fit \
  --name kroger-project \
  --display-name "Kroger Project (.venv-course-fit)"
```

### 2. Add the source data

Obtain [Dunnhumby – The Complete Journey from Kaggle](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey) through your authorised Kaggle account, then place the eight CSV files listed in [`data/README.md`](data/README.md) inside `data/raw/`.

The raw and processed datasets are deliberately excluded from Git because several files exceed GitHub's 100 MB individual-file limit. They may also be subject to the original dataset's distribution terms.

### 3. Clean and prepare the data

Prepare and validate the raw data:

   ```bash
   .venv-course-fit/bin/python src/data_cleaning/prepare_data.py
   ```

Build the panel-regression dataset:

   ```bash
   .venv-course-fit/bin/python src/data_cleaning/build_regression_panel.py
   ```

### 4. Run the analysis

Open the main analysis:

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
- The optional local `data/processed_data.zip` archive is not needed: both cleaning scripts reproduce the required processed files.


