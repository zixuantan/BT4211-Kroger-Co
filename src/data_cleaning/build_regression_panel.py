#!/usr/bin/env python3
"""Build product-store-week panel used in the panel-regression section.
Run with:
    python src/data_cleaning/build_regression_panel.py
"""

import duckdb
from pathlib import Path 

#set the file paths 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "regression_panel.parquet"
TEMP_PATH = OUTPUT_DIR / "regression_panel.building.parquet"

#create processed folder if does not exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

#identify display and mailer support, as long as there is a non-zero for that particular pdt-store-week, then just take it as it has some kind of support 
con.execute("""
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

    FROM read_csv_auto('{RAW_DIR / "causal_data.csv"}')
    GROUP BY PRODUCT_ID, STORE_ID, WEEK_NO
""")


#aggregate weekly product-store sales from the transaction data
con.execute("""
    CREATE OR REPLACE TEMP TABLE weekly_sales AS
    SELECT
        PRODUCT_ID,
        STORE_ID,
        WEEK_NO,
        SUM(SALES_VALUE) AS sales_revenue,
        SUM(QUANTITY) AS quantity_sold

    FROM read_csv_auto('{RAW_DIR / "transaction_data.csv"}')
    GROUP BY PRODUCT_ID, STORE_ID, WEEK_NO
""")

#combine the promotional placements and sales from above
con.execute("""
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
""")

#identify coupon-eligible product-weeks
con.execute("""
    CREATE OR REPLACE TEMP TABLE coupon_weeks AS
    SELECT DISTINCT
        c.PRODUCT_ID,
        week_no

    FROM read_csv_auto('{RAW_DIR / "coupon.csv"}') c
    JOIN read_csv_auto('{RAW_DIR / "campaign_desc.csv"}') d
        ON c.CAMPAIGN = d.CAMPAIGN

    CROSS JOIN LATERAL (
        SELECT UNNEST(
            RANGE(
                CAST(FLOOR((d.START_DAY - 1) / 7) + 1 AS INTEGER),
                CAST(FLOOR((d.END_DAY - 1) / 7) + 2 AS INTEGER)
            )
        ) AS week_no
    )
""")

#adds coupon eligibility and classify promotional support
con.execute("""
    CREATE OR REPLACE TEMP TABLE final_panel AS
    SELECT
        p.*,

        CASE WHEN c.PRODUCT_ID IS NOT NULL
             THEN 1 ELSE 0 END AS coupon_eligible,

        CASE
            WHEN p.has_display = 1 AND p.has_mailer = 1
                THEN 'combined'
            WHEN p.has_display = 1 AND p.has_mailer = 0
                THEN 'display_only'
            WHEN p.has_display = 0 AND p.has_mailer = 1
                THEN 'mailer_only'
            ELSE 'no_support'
        END AS promotion_type

    FROM promo_sales_panel p
    LEFT JOIN coupon_weeks c
        ON p.PRODUCT_ID = c.PRODUCT_ID
        AND p.WEEK_NO = c.WEEK_NO
""")


#Create promotional support dummy variables for later part
con.execute("""
    CREATE OR REPLACE TEMP TABLE regression_panel AS
    SELECT
        *,

        CASE WHEN promotion_type = 'display_only'
             THEN 1 ELSE 0 END AS display_only,

        CASE WHEN promotion_type = 'combined'
             THEN 1 ELSE 0 END AS combined

    FROM final_panel
""")


#save completed regression dataset
con.execute("""
    COPY regression_panel
    TO '{OUTPUT_PATH.as_posix()}'
    (FORMAT PARQUET)
""")
print(f"Reg panel saved to: {OUTPUT_PATH}")

con.close()