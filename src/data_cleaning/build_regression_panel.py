#!/usr/bin/env python3
"""Build the product-store-week panel used by the panel-regression section.

The output combines observed display/mailer placements with weekly transaction
sales and flags whether a product was coupon-eligible during an active campaign.
Run from anywhere with:

    python src/data_cleaning/build_regression_panel.py
"""

from __future__ import annotations

from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "regression_panel.parquet"
TEMP_PATH = OUTPUT_DIR / "regression_panel.building.parquet"


def sql_path(path: Path) -> str:
    """Return a filesystem path safely escaped for a DuckDB SQL literal."""
    return str(path.resolve()).replace("'", "''")


def validate_inputs() -> None:
    required = [
        RAW_DIR / "causal_data.csv",
        RAW_DIR / "transaction_data.csv",
        RAW_DIR / "coupon.csv",
        RAW_DIR / "campaign_desc.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing regression-panel inputs:\n" + "\n".join(missing))


def build_regression_panel(output_path: Path = OUTPUT_PATH) -> Path:
    """Create and validate the regression panel, then replace the output atomically."""
    validate_inputs()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.stem}.building{output_path.suffix}")
    if temp_path.exists():
        temp_path.unlink()

    causal_path = sql_path(RAW_DIR / "causal_data.csv")
    transaction_path = sql_path(RAW_DIR / "transaction_data.csv")
    coupon_path = sql_path(RAW_DIR / "coupon.csv")
    campaign_path = sql_path(RAW_DIR / "campaign_desc.csv")
    temp_sql_path = sql_path(temp_path)

    con = duckdb.connect()
    try:
        con.execute("SET memory_limit='4GB'")
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE clean_placements AS
            SELECT
                PRODUCT_ID,
                STORE_ID,
                WEEK_NO,
                MAX(CASE
                    WHEN TRIM(CAST(display AS VARCHAR)) NOT IN ('0', '')
                    THEN 1 ELSE 0
                END) AS has_display,
                MAX(CASE
                    WHEN TRIM(CAST(mailer AS VARCHAR)) NOT IN ('0', '')
                    THEN 1 ELSE 0
                END) AS has_mailer
            FROM read_csv_auto('{causal_path}')
            GROUP BY PRODUCT_ID, STORE_ID, WEEK_NO
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE weekly_sales AS
            SELECT
                PRODUCT_ID,
                STORE_ID,
                WEEK_NO,
                SUM(SALES_VALUE) AS sales_revenue,
                SUM(QUANTITY) AS quantity_sold
            FROM read_csv_auto('{transaction_path}')
            GROUP BY PRODUCT_ID, STORE_ID, WEEK_NO
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE promo_sales_panel AS
            SELECT
                p.PRODUCT_ID,
                p.STORE_ID,
                p.WEEK_NO,
                p.has_display,
                p.has_mailer,
                COALESCE(s.sales_revenue, 0) AS sales_revenue,
                COALESCE(s.quantity_sold, 0) AS quantity_sold
            FROM clean_placements p
            LEFT JOIN weekly_sales s
              ON p.PRODUCT_ID = s.PRODUCT_ID
             AND p.STORE_ID = s.STORE_ID
             AND p.WEEK_NO = s.WEEK_NO
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE coupon_weeks AS
            SELECT DISTINCT c.PRODUCT_ID, week_no
            FROM read_csv_auto('{coupon_path}') c
            JOIN read_csv_auto('{campaign_path}') d
              ON c.CAMPAIGN = d.CAMPAIGN
            CROSS JOIN LATERAL (
                SELECT UNNEST(
                    RANGE(
                        CAST(FLOOR((d.START_DAY - 1) / 7) + 1 AS INTEGER),
                        CAST(FLOOR((d.END_DAY - 1) / 7) + 2 AS INTEGER)
                    )
                ) AS week_no
            )
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE regression_panel AS
            SELECT
                p.*,
                CASE WHEN c.PRODUCT_ID IS NOT NULL THEN 1 ELSE 0 END AS coupon_eligible,
                CASE
                    WHEN p.has_display = 1 AND p.has_mailer = 1 THEN 'combined'
                    WHEN p.has_display = 1 AND p.has_mailer = 0 THEN 'display_only'
                    WHEN p.has_display = 0 AND p.has_mailer = 1 THEN 'mailer_only'
                    ELSE 'no_support'
                END AS promotion_type,
                CASE
                    WHEN p.has_display = 1 AND p.has_mailer = 0 THEN 1 ELSE 0
                END AS display_only,
                CASE
                    WHEN p.has_display = 1 AND p.has_mailer = 1 THEN 1 ELSE 0
                END AS combined
            FROM promo_sales_panel p
            LEFT JOIN coupon_weeks c
              ON p.PRODUCT_ID = c.PRODUCT_ID
             AND p.WEEK_NO = c.week_no
            """
        )

        checks = con.execute(
            """
            SELECT
                COUNT(*) AS rows,
                COUNT(DISTINCT (PRODUCT_ID, STORE_ID, WEEK_NO)) AS unique_keys,
                SUM(coupon_eligible) AS coupon_eligible_rows,
                SUM(display_only) AS display_only_rows,
                SUM(combined) AS combined_rows
            FROM regression_panel
            """
        ).fetchone()
        if checks[0] != checks[1]:
            raise AssertionError("Regression panel key is not unique")

        con.execute(
            f"""
            COPY regression_panel TO '{temp_sql_path}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        con.close()

    temp_path.replace(output_path)
    print(
        f"Created {output_path.relative_to(PROJECT_ROOT)}: "
        f"{checks[0]:,} rows, {checks[2]:,} coupon-eligible rows, "
        f"{checks[3]:,} display-only rows, {checks[4]:,} combined rows"
    )
    return output_path


if __name__ == "__main__":
    build_regression_panel()
