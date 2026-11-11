#!/usr/bin/env python3
"""Prepare analysis-ready Dunnhumby data sets with validation and reporting.

Run from anywhere with:
    python3 src/data_cleaning/prepare_data.py

The script uses pandas for cleaning/aggregation and pyarrow only as the Parquet
writer.  The 36.8-million-row causal table is written in chunks so the final
panel does not need to be held fully in memory.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = OUTPUT_DIR / "data_preparation_report.md"
PANEL_CHUNK_SIZE = 1_000_000

FILES = {
    "transaction_data": "transaction_data.csv",
    "product": "product.csv",
    "causal_data": "causal_data.csv",
    "hh_demographic": "hh_demographic.csv",
    "campaign_table": "campaign_table.csv",
    "campaign_desc": "campaign_desc.csv",
    "coupon": "coupon.csv",
    "coupon_redempt": "coupon_redempt.csv",
}

KEYS = {
    "transaction_data": ["household_key", "BASKET_ID", "PRODUCT_ID"],
    "product": ["PRODUCT_ID"],
    "causal_data": ["PRODUCT_ID", "STORE_ID", "WEEK_NO"],
    "hh_demographic": ["household_key"],
    "campaign_table": ["household_key", "CAMPAIGN"],
    "campaign_desc": ["CAMPAIGN"],
    "coupon": ["COUPON_UPC", "PRODUCT_ID", "CAMPAIGN"],
    "coupon_redempt": ["household_key", "DAY", "COUPON_UPC", "CAMPAIGN"],
}


def slug(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return text or "blank"


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype("float64").div(denominator.replace(0, np.nan))
    return result.replace([np.inf, -np.inf], np.nan)


def joined_codes(values: pd.Series) -> str:
    """Preserve every distinct categorical code when a causal key repeats."""
    unique = sorted({str(v) for v in values if pd.notna(v)})
    return "|".join(unique)


def has_nonzero_code(value: object) -> bool:
    """Treat literal code 0 as no placement; do not decode other code meanings."""
    if pd.isna(value):
        return False
    return any(code != "0" for code in str(value).split("|"))


def audit_frame(name: str, frame: pd.DataFrame, key: list[str]) -> dict:
    return {
        "name": name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing": {column: int(count) for column, count in frame.isna().sum().items()},
        "exact_duplicates": int(frame.duplicated().sum()),
        "key": key,
        "key_duplicates": int(frame.duplicated(key).sum()),
    }


def write_parquet(frame: pd.DataFrame, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    frame.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    return path


def build_store_outputs(
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    demographics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    before = len(transactions)
    tx = transactions.merge(products, on="PRODUCT_ID", how="left", validate="many_to_one", indicator=True)
    assert len(tx) == before, "Product join multiplied transaction rows"
    product_match_rate = float((tx["_merge"] == "both").mean())
    assert product_match_rate == 1.0, "Some transaction products lack product metadata"
    tx = tx.drop(columns="_merge")

    tx["retail_discount_savings"] = (-tx["RETAIL_DISC"]).clip(lower=0)
    tx["has_retail_discount"] = tx["RETAIL_DISC"] < 0

    grouped = tx.groupby("STORE_ID", observed=True)
    store = grouped.agg(
        transaction_line_count=("PRODUCT_ID", "size"),
        total_observed_sales_revenue=("SALES_VALUE", "sum"),
        distinct_baskets=("BASKET_ID", "nunique"),
        distinct_households=("household_key", "nunique"),
        total_units=("QUANTITY", "sum"),
        active_weeks=("WEEK_NO", "nunique"),
        retail_discount_line_count=("has_retail_discount", "sum"),
        total_retail_discount_raw=("RETAIL_DISC", "sum"),
        total_retail_discount_savings=("retail_discount_savings", "sum"),
        sales_on_retail_discount_lines=(
            "SALES_VALUE",
            lambda s: s[tx.loc[s.index, "has_retail_discount"]].sum(),
        ),
    )

    basket = tx.groupby(["STORE_ID", "BASKET_ID"], observed=True).agg(
        basket_sales=("SALES_VALUE", "sum"),
        basket_units=("QUANTITY", "sum"),
        basket_product_lines=("PRODUCT_ID", "size"),
    )
    basket_summary = basket.groupby("STORE_ID").agg(
        average_basket_value=("basket_sales", "mean"),
        average_basket_size_units=("basket_units", "mean"),
        average_basket_product_lines=("basket_product_lines", "mean"),
    )
    store = store.join(basket_summary)

    store_hh = tx[["STORE_ID", "household_key"]].drop_duplicates()
    store_hh_week = tx[["STORE_ID", "household_key", "WEEK_NO"]].drop_duplicates()
    hh_baskets = (
        tx[["STORE_ID", "household_key", "BASKET_ID"]]
        .drop_duplicates()
        .groupby(["STORE_ID", "household_key"])
        .size()
    )
    hh_weeks = store_hh_week.groupby(["STORE_ID", "household_key"]).size()
    household_behaviour = pd.DataFrame(
        {
            "average_baskets_per_observed_household": hh_baskets.groupby("STORE_ID").mean(),
            "average_shopping_weeks_per_observed_household": hh_weeks.groupby("STORE_ID").mean(),
        }
    )
    store = store.join(household_behaviour)

    brand_sales = tx.groupby(["STORE_ID", "BRAND"], observed=True)["SALES_VALUE"].sum().unstack(fill_value=0)
    for brand_name, output_name in [("Private", "private_label_sales_share"), ("National", "national_brand_sales_share")]:
        values = brand_sales[brand_name] if brand_name in brand_sales else pd.Series(0.0, index=store.index)
        store[output_name] = safe_divide(values.reindex(store.index, fill_value=0), store["total_observed_sales_revenue"])

    department_totals = tx.groupby("DEPARTMENT", observed=True)["SALES_VALUE"].sum().sort_values(ascending=False)
    major_departments = department_totals.head(10).index.tolist()
    department_sales = tx.groupby(["STORE_ID", "DEPARTMENT"], observed=True)["SALES_VALUE"].sum().unstack(fill_value=0)
    major_sum = pd.Series(0.0, index=store.index)
    for department in major_departments:
        values = department_sales.get(department, pd.Series(0.0, index=department_sales.index)).reindex(store.index, fill_value=0)
        store[f"department_share_{slug(department)}"] = safe_divide(values, store["total_observed_sales_revenue"])
        major_sum = major_sum.add(values, fill_value=0)
    store["department_share_other"] = safe_divide(store["total_observed_sales_revenue"] - major_sum, store["total_observed_sales_revenue"])

    store["average_observed_spending_per_household"] = safe_divide(
        store["total_observed_sales_revenue"], store["distinct_households"]
    )
    store["average_weekly_observed_sales"] = safe_divide(
        store["total_observed_sales_revenue"], store["active_weeks"]
    )
    store["retail_discount_sales_share"] = safe_divide(
        store["sales_on_retail_discount_lines"], store["total_observed_sales_revenue"]
    )
    store["average_retail_discount_savings_per_line"] = safe_divide(
        store["total_retail_discount_savings"], store["transaction_line_count"]
    )
    positive_units = tx.loc[tx["QUANTITY"] > 0].groupby("STORE_ID")["QUANTITY"].sum().reindex(store.index, fill_value=0)
    store["average_retail_discount_savings_per_positive_unit"] = safe_divide(
        store["total_retail_discount_savings"], positive_units
    )

    # Transparent review flags only. No stores are excluded.
    store["sparse_baskets_flag_lt_100"] = store["distinct_baskets"] < 100
    store["sparse_households_flag_lt_20"] = store["distinct_households"] < 20
    store["sparse_active_weeks_flag_lt_26"] = store["active_weeks"] < 26
    store["potentially_sparse_observations"] = store[
        ["sparse_baskets_flag_lt_100", "sparse_households_flag_lt_20", "sparse_active_weeks_flag_lt_26"]
    ].any(axis=1)
    store = store.reset_index().sort_values("STORE_ID").reset_index(drop=True)
    assert store["STORE_ID"].is_unique

    demo_link = store_hh.merge(demographics, on="household_key", how="left", validate="many_to_one", indicator=True)
    coverage = demo_link.groupby("STORE_ID").agg(
        observed_households=("household_key", "nunique"),
        households_with_demographics=("_merge", lambda s: int((s == "both").sum())),
    )
    coverage["demographic_coverage_pct"] = safe_divide(
        coverage["households_with_demographics"], coverage["observed_households"]
    )
    covered = demo_link.loc[demo_link["_merge"] == "both"].copy()
    demo = coverage.copy()
    for column, prefix in [
        ("INCOME_DESC", "income_share"),
        ("AGE_DESC", "age_share"),
        ("HOUSEHOLD_SIZE_DESC", "household_size_share"),
    ]:
        distribution = pd.crosstab(covered["STORE_ID"], covered[column], normalize="index")
        distribution.columns = [f"{prefix}_{slug(value)}" for value in distribution.columns]
        demo = demo.join(distribution, how="left")
    demo = demo.reset_index().sort_values("STORE_ID").reset_index(drop=True)
    assert demo["STORE_ID"].is_unique

    checks = {
        "transaction_product_match_rate": product_match_rate,
        "major_departments": major_departments,
        "store_count": len(store),
        "sparse_store_count": int(store["potentially_sparse_observations"].sum()),
        "zero_sales_store_ids": store.loc[
            store["total_observed_sales_revenue"] == 0, "STORE_ID"
        ].tolist(),
        "stores_without_demographic_coverage": int(
            (demo["households_with_demographics"] == 0).sum()
        ),
        "store_transactions_summary": store["transaction_line_count"].describe().to_dict(),
        "store_households_summary": store["distinct_households"].describe().to_dict(),
        "store_active_weeks_summary": store["active_weeks"].describe().to_dict(),
    }
    return store, demo, checks


def aggregate_product_store_week(transactions: pd.DataFrame) -> pd.DataFrame:
    grouped = transactions.groupby(["PRODUCT_ID", "STORE_ID", "WEEK_NO"], observed=True)
    weekly = grouped.agg(
        sales_revenue=("SALES_VALUE", "sum"),
        quantity_sold=("QUANTITY", "sum"),
        product_basket_count=("BASKET_ID", "nunique"),
        purchasing_household_count=("household_key", "nunique"),
        transaction_line_count=("PRODUCT_ID", "size"),
        total_retail_discount=("RETAIL_DISC", "sum"),
        total_coupon_discount=("COUPON_DISC", "sum"),
        total_coupon_match_discount=("COUPON_MATCH_DISC", "sum"),
    ).reset_index()
    weekly["average_realized_selling_price"] = np.where(
        weekly["quantity_sold"] > 0,
        weekly["sales_revenue"] / weekly["quantity_sold"],
        np.nan,
    )
    assert not weekly.duplicated(["PRODUCT_ID", "STORE_ID", "WEEK_NO"]).any()
    assert np.isclose(weekly["sales_revenue"].sum(), transactions["SALES_VALUE"].sum(), rtol=0, atol=0.01)
    return weekly


def prepare_causal_duplicates() -> tuple[pd.DataFrame, pd.MultiIndex, dict]:
    causal = pd.read_csv(
        DATA_DIR / FILES["causal_data"],
        dtype={"PRODUCT_ID": "int64", "STORE_ID": "int64", "WEEK_NO": "int64", "display": "string", "mailer": "string"},
    )
    key = KEYS["causal_data"]
    duplicate_mask = causal.duplicated(key, keep=False)
    duplicate_rows = causal.loc[duplicate_mask].copy()
    duplicate_keys = pd.MultiIndex.from_frame(duplicate_rows[key].drop_duplicates())
    resolved = duplicate_rows.groupby(key, as_index=False).agg(
        display=("display", joined_codes),
        mailer=("mailer", joined_codes),
        promotion_source_rows=("PRODUCT_ID", "size"),
        display_distinct_code_count=("display", "nunique"),
        mailer_distinct_code_count=("mailer", "nunique"),
    )
    stats = audit_frame("causal_data", causal, key)
    stats.update(
        {
            "display_values": causal["display"].value_counts().to_dict(),
            "mailer_values": causal["mailer"].value_counts().to_dict(),
            "duplicate_key_groups": len(resolved),
            "duplicate_keys_with_display_conflict": int((resolved["display_distinct_code_count"] > 1).sum()),
            "duplicate_keys_with_mailer_conflict": int((resolved["mailer_distinct_code_count"] > 1).sum()),
            "min_week": int(causal["WEEK_NO"].min()),
            "max_week": int(causal["WEEK_NO"].max()),
            "unique_stores": int(causal["STORE_ID"].nunique()),
            "unique_products": int(causal["PRODUCT_ID"].nunique()),
        }
    )
    del causal, duplicate_rows
    return resolved, duplicate_keys, stats


def enrich_panel_chunk(
    causal_chunk: pd.DataFrame,
    weekly_sales: pd.DataFrame,
    products: pd.DataFrame,
    min_causal_week: int,
) -> pd.DataFrame:
    key = KEYS["causal_data"]
    before = len(causal_chunk)
    panel = causal_chunk.merge(weekly_sales, on=key, how="left", validate="one_to_one", indicator="transaction_merge")
    assert len(panel) == before, "Sales join multiplied causal rows"
    panel["transaction_match"] = panel["transaction_merge"].eq("both")
    panel = panel.drop(columns="transaction_merge")

    zero_float = ["sales_revenue", "total_retail_discount", "total_coupon_discount", "total_coupon_match_discount"]
    zero_int = ["quantity_sold", "product_basket_count", "purchasing_household_count", "transaction_line_count"]
    panel[zero_float] = panel[zero_float].fillna(0.0).astype("float64")
    panel[zero_int] = panel[zero_int].fillna(0).astype("int64")
    # average_realized_selling_price remains missing when quantity is zero or no transaction matched.

    panel = panel.merge(products, on="PRODUCT_ID", how="left", validate="many_to_one", indicator="product_merge")
    assert len(panel) == before, "Product join multiplied panel rows"
    assert panel["product_merge"].eq("both").all(), "Causal product lacks product metadata"
    panel = panel.drop(columns="product_merge")
    panel["has_display"] = panel["display"].map(has_nonzero_code).astype(bool)
    panel["has_mailer"] = panel["mailer"].map(has_nonzero_code).astype(bool)
    panel["has_any_placement"] = panel["has_display"] | panel["has_mailer"]
    panel["has_both_display_mailer"] = panel["has_display"] & panel["has_mailer"]
    panel["promotion_code_conflict"] = (
        (panel["display_distinct_code_count"] > 1) | (panel["mailer_distinct_code_count"] > 1)
    )
    panel["relative_week_index"] = panel["WEEK_NO"] - min_causal_week + 1
    return panel


def build_promo_panel(
    weekly_sales: pd.DataFrame,
    products: pd.DataFrame,
    resolved_duplicates: pd.DataFrame,
    duplicate_keys: pd.MultiIndex,
    causal_stats: dict,
) -> dict:
    path = OUTPUT_DIR / "df_promo_panel.parquet"
    writer: pq.ParquetWriter | None = None
    row_count = 0
    transaction_matches = 0
    panel_sales_sum = 0.0
    null_counts: dict[str, int] = {}
    key = KEYS["causal_data"]

    def append(frame: pd.DataFrame) -> None:
        nonlocal writer, row_count, transaction_matches, panel_sales_sum, null_counts
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, compression="snappy")
        writer.write_table(table)
        row_count += len(frame)
        transaction_matches += int(frame["transaction_match"].sum())
        panel_sales_sum += float(frame["sales_revenue"].sum())
        for column, count in frame.isna().sum().items():
            null_counts[column] = null_counts.get(column, 0) + int(count)

    reader = pd.read_csv(
        DATA_DIR / FILES["causal_data"],
        chunksize=PANEL_CHUNK_SIZE,
        dtype={"PRODUCT_ID": "int64", "STORE_ID": "int64", "WEEK_NO": "int64", "display": "string", "mailer": "string"},
    )
    for chunk in reader:
        chunk_index = pd.MultiIndex.from_frame(chunk[key])
        chunk = chunk.loc[~chunk_index.isin(duplicate_keys)].copy()
        chunk["promotion_source_rows"] = 1
        chunk["display_distinct_code_count"] = 1
        chunk["mailer_distinct_code_count"] = 1
        append(enrich_panel_chunk(chunk, weekly_sales, products, causal_stats["min_week"]))

    append(enrich_panel_chunk(resolved_duplicates, weekly_sales, products, causal_stats["min_week"]))
    if writer is not None:
        writer.close()

    expected_rows = causal_stats["rows"] - causal_stats["key_duplicates"]
    assert row_count == expected_rows, f"Panel has {row_count:,} rows; expected {expected_rows:,} unique keys"
    # Key uniqueness follows from removing all repeated source keys and appending one resolved row per key.
    return {
        "rows": row_count,
        "columns": len(pq.ParquetFile(path).schema.names),
        "transaction_matches": transaction_matches,
        "transaction_match_rate": transaction_matches / row_count,
        "unmatched_rows": row_count - transaction_matches,
        "transaction_aggregate_rows": len(weekly_sales),
        "transaction_aggregate_key_match_rate": transaction_matches / len(weekly_sales),
        "panel_sales_sum": panel_sales_sum,
        "source_sales_sum": float(weekly_sales["sales_revenue"].sum()),
        "panel_sales_coverage_rate": panel_sales_sum / float(weekly_sales["sales_revenue"].sum()),
        "null_counts": null_counts,
        "primary_key_unique": True,
        "path": path,
    }


def build_campaign_outputs(
    campaign_table: pd.DataFrame,
    campaign_desc: pd.DataFrame,
    coupon: pd.DataFrame,
    redemptions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    campaigns = campaign_table.merge(
        campaign_desc,
        on="CAMPAIGN",
        how="left",
        validate="many_to_one",
        suffixes=("_table", "_campaign"),
        indicator=True,
    )
    assert campaigns["_merge"].eq("both").all()
    assert campaigns["DESCRIPTION_table"].eq(campaigns["DESCRIPTION_campaign"]).all()
    campaigns = campaigns.rename(columns={"DESCRIPTION_campaign": "campaign_type"})[
        ["household_key", "CAMPAIGN", "campaign_type", "START_DAY", "END_DAY"]
    ]
    assert not campaigns.duplicated(["household_key", "CAMPAIGN"]).any()

    coupon_mapping = coupon.drop_duplicates().sort_values(["CAMPAIGN", "COUPON_UPC", "PRODUCT_ID"]).reset_index(drop=True)
    assert not coupon_mapping.duplicated(["COUPON_UPC", "PRODUCT_ID", "CAMPAIGN"]).any()

    redemption = redemptions.merge(
        campaign_desc.rename(columns={"DESCRIPTION": "campaign_type"}),
        on="CAMPAIGN",
        how="left",
        validate="many_to_one",
        indicator="campaign_merge",
    )
    assert redemption["campaign_merge"].eq("both").all()
    redemption = redemption.drop(columns="campaign_merge")
    coupon_pairs = coupon_mapping[["COUPON_UPC", "CAMPAIGN"]].drop_duplicates()
    redemption = redemption.merge(coupon_pairs, on=["COUPON_UPC", "CAMPAIGN"], how="left", validate="many_to_one", indicator="coupon_mapping_merge")
    redemption["coupon_mapping_match"] = redemption["coupon_mapping_merge"].eq("both")
    redemption = redemption.drop(columns="coupon_mapping_merge")
    participant_pairs = campaign_table[["household_key", "CAMPAIGN"]].drop_duplicates()
    redemption = redemption.merge(
        participant_pairs,
        on=["household_key", "CAMPAIGN"],
        how="left",
        validate="many_to_one",
        indicator="participation_merge",
    )
    redemption["campaign_participation_match"] = redemption["participation_merge"].eq("both")
    redemption = redemption.drop(columns="participation_merge")
    assert len(redemption) == len(redemptions), "Redemptions multiplied during enrichment"
    assert not redemption.duplicated(KEYS["coupon_redempt"]).any()

    checks = {
        "campaign_description_match_rate": 1.0,
        "coupon_exact_duplicates_removed": int(len(coupon) - len(coupon_mapping)),
        "coupon_pair_count": int(coupon_mapping[["COUPON_UPC", "CAMPAIGN"]].drop_duplicates().shape[0]),
        "coupon_pair_multiple_product_count": int((coupon_mapping.groupby(["COUPON_UPC", "CAMPAIGN"]).size() > 1).sum()),
        "max_products_per_coupon_campaign": int(coupon_mapping.groupby(["COUPON_UPC", "CAMPAIGN"]).size().max()),
        "redemption_coupon_match_rate": float(redemption["coupon_mapping_match"].mean()),
        "redemption_participation_match_rate": float(redemption["campaign_participation_match"].mean()),
    }
    return campaigns, redemption, coupon_mapping, checks


def dataframe_validation(name: str, frame: pd.DataFrame, unit: str, key: list[str]) -> dict:
    return {
        "name": name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "unit": unit,
        "key": key,
        "unique": not frame.duplicated(key).any(),
        "missing_pct": (frame.isna().mean() * 100).to_dict(),
    }


def fmt_number(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def render_schema(audit: dict) -> str:
    lines = [f"### `{audit['name']}.csv`", "", f"- Shape: {audit['rows']:,} rows × {audit['columns']} columns", f"- Candidate key: `{', '.join(audit['key'])}`", f"- Exact duplicate rows: {audit['exact_duplicates']:,}", f"- Duplicate candidate-key rows (beyond first): {audit['key_duplicates']:,}", "", "| Column | dtype | Missing |", "|---|---:|---:|"]
    for column, dtype in audit["dtypes"].items():
        lines.append(f"| `{column}` | `{dtype}` | {audit['missing'][column]:,} |")
    return "\n".join(lines)


def render_output_validation(item: dict) -> str:
    nonzero_missing = {key: value for key, value in item["missing_pct"].items() if value > 0}
    missing_text = ", ".join(f"`{key}` {value:.2f}%" for key, value in nonzero_missing.items()) or "None"
    return "\n".join(
        [
            f"### `{item['name']}`",
            "",
            f"- Shape: {item['rows']:,} rows × {item['columns']} columns",
            f"- Unit of observation: {item['unit']}",
            f"- Primary key: `{', '.join(item['key'])}`",
            f"- Primary key unique: **{item['unique']}**",
            f"- Columns with missing values: {missing_text}",
        ]
    )


def write_report(
    audits: list[dict],
    validations: list[dict],
    store_checks: dict,
    causal_checks: dict,
    panel_checks: dict,
    campaign_checks: dict,
    transactions: pd.DataFrame,
) -> None:
    discount_counts = {
        column: {
            "negative": int((transactions[column] < 0).sum()),
            "zero": int((transactions[column] == 0).sum()),
            "positive": int((transactions[column] > 0).sum()),
            "sum": float(transactions[column].sum()),
        }
        for column in ["RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC"]
    }
    quantity_quantiles = transactions["QUANTITY"].quantile([0.5, 0.99, 0.999, 1]).to_dict()
    schemas = "\n\n".join(render_schema(item) for item in audits)
    outputs = "\n\n".join(render_output_validation(item) for item in validations)
    major_departments = ", ".join(f"`{value}`" for value in store_checks["major_departments"])
    report = f"""# Dunnhumby data preparation report

## Scope

This pipeline prepares store profiles, a product-store-week promotion panel, and supplementary household campaign/coupon files. It does not perform clustering, regression, prediction, causal inference, or managerial recommendation. Sales metrics describe only the supplied sampled-household transactions, not total store sales.

## Source data audit

{schemas}

## Identifier, time, and categorical checks

- `product.PRODUCT_ID` is unique. It matches 100% of transaction rows and 100% of causal rows, so product joins are many-to-one and do not multiply rows.
- `BASKET_ID` identifies 276,484 baskets globally. Within each basket, household, store, day, week, and transaction time are constant. No basket-product pair repeats.
- Transaction `DAY` spans 1–711 and `WEEK_NO` spans 1–102. The exact observed mapping is `WEEK_NO = ((DAY + 1) // 7) + 1`; weeks begin at day 6 after a five-day first partial week.
- Causal weeks span {causal_checks['min_week']}–{causal_checks['max_week']} and cover {causal_checks['unique_stores']} stores, versus {store_checks['store_count']} stores in transactions. The causal table is therefore a promotion-analysis universe, not the complete transaction-store universe.
- The causal source has {causal_checks['duplicate_key_groups']:,} duplicated product-store-week keys. {causal_checks['duplicate_keys_with_display_conflict']:,} vary in display code and {causal_checks['duplicate_keys_with_mailer_conflict']:,} vary in mailer code. The panel retains one row per key and preserves all distinct codes in pipe-delimited form, plus source-row and conflict fields.
- Display values and counts: `{causal_checks['display_values']}`.
- Mailer values and counts: `{causal_checks['mailer_values']}`.
- Literal code `0` is treated as no placement for indicator construction. All nonzero codes are retained without attempting to decode their unfamiliar meanings.
- Product `BRAND` contains only `National` and `Private`. Product size has 30,607 single-space values, which are blank-like text rather than CSV nulls and are retained.
- Household demographics cover 801 unique households and contain no CSV nulls; categorical labels such as `Unknown` and `None/Unknown` are retained as source values.

## Discount and quantity checks

Source discount fields are primarily nonpositive, so negative values represent discounts. Raw signed totals are preserved. Store features also calculate positive retail savings as `max(-RETAIL_DISC, 0)`.

| Field | Negative rows | Zero rows | Positive rows | Raw sum |
|---|---:|---:|---:|---:|
| `RETAIL_DISC` | {discount_counts['RETAIL_DISC']['negative']:,} | {discount_counts['RETAIL_DISC']['zero']:,} | {discount_counts['RETAIL_DISC']['positive']:,} | {discount_counts['RETAIL_DISC']['sum']:,.2f} |
| `COUPON_DISC` | {discount_counts['COUPON_DISC']['negative']:,} | {discount_counts['COUPON_DISC']['zero']:,} | {discount_counts['COUPON_DISC']['positive']:,} | {discount_counts['COUPON_DISC']['sum']:,.2f} |
| `COUPON_MATCH_DISC` | {discount_counts['COUPON_MATCH_DISC']['negative']:,} | {discount_counts['COUPON_MATCH_DISC']['zero']:,} | {discount_counts['COUPON_MATCH_DISC']['positive']:,} | {discount_counts['COUPON_MATCH_DISC']['sum']:,.2f} |

`QUANTITY` is never negative, but it is highly skewed (median {quantity_quantiles[0.5]:,.0f}, 99th percentile {quantity_quantiles[0.99]:,.0f}, 99.9th percentile {quantity_quantiles[0.999]:,.0f}, maximum {quantity_quantiles[1.0]:,.0f}). The pipeline retains these records and documents the resulting sensitivity of unit-based basket-size and unit-price measures.

## Construction logic and join cardinalities

### Store profiles

- Transactions are left joined to unique product metadata with a many-to-one validation; match rate: {store_checks['transaction_product_match_rate']:.2%}.
- Basket value and basket size are calculated after aggregating transaction lines by store and basket. Basket size is supplied in both units and product-line counts because extreme source quantities can distort unit counts.
- Spending per household and baskets/shopping weeks per household use observed households only.
- Brand shares use source `BRAND`. Department shares use the ten departments with the most observed transaction sales: {major_departments}. Remaining departments are combined into `department_share_other`.
- Sparse-store flags are review aids only: fewer than 100 baskets, fewer than 20 households, or fewer than 26 active weeks. {store_checks['sparse_store_count']:,} stores meet at least one flag; none are excluded.

Transaction lines per store: `{store_checks['store_transactions_summary']}`.

Unique households per store: `{store_checks['store_households_summary']}`.

Active weeks per store: `{store_checks['store_active_weeks_summary']}`.

### Store demographics

- Each household is counted at most once per store before the unique household-demographic join.
- Household income, age, and size shares are calculated among covered observed households only.
- Coverage counts and percentages are retained for every store. Missing demographics are not imputed, and households that use multiple stores contribute once to each applicable store profile.
- {store_checks['stores_without_demographic_coverage']:,} stores have no observed household with demographic data; their distribution shares remain missing rather than being imputed or zero-filled.

### Product-store-week promotional panel

- Transaction sales reconcile to the source total within one cent before the causal join.
- The causal table is the starting universe. After consolidating repeated causal keys, the panel has {panel_checks['rows']:,} unique product-store-week rows.
- {panel_checks['transaction_matches']:,} rows match at least one transaction aggregation ({panel_checks['transaction_match_rate']:.2%}); {panel_checks['unmatched_rows']:,} do not.
- Transactions form {panel_checks['transaction_aggregate_rows']:,} distinct product-store-week groups. {panel_checks['transaction_aggregate_key_match_rate']:.2%} of those keys appear in the causal universe. Their sales total {panel_checks['panel_sales_sum']:,.2f}, or {panel_checks['panel_sales_coverage_rate']:.2%} of the {panel_checks['source_sales_sum']:,.2f} source transaction sales; the difference is retained in the transaction source but lies outside the causal panel universe.
- For unmatched causal rows, additive observed-transaction measures are zero-filled and `transaction_match=False`. This means no sampled transaction matched; it does not establish that a product was stocked and unsold.
- `average_realized_selling_price` is sales divided by positive quantity and remains missing otherwise.
- Product metadata joins many-to-one at 100%. `STORE_ID` is retained. No `store_segment` value is created before clustering.
- `relative_week_index` starts at 1 for causal week {causal_checks['min_week']}. Month and year are not derived because the files provide no calendar origin.

### Campaign and coupon files

- Household campaigns join campaign descriptions many-to-one at a {campaign_checks['campaign_description_match_rate']:.2%} match rate; campaign type values agree between both sources.
- The coupon source contains {campaign_checks['coupon_exact_duplicates_removed']:,} exact duplicates, removed only from the coupon-product mapping. The cleaned mapping remains one row per coupon-product-campaign.
- There are {campaign_checks['coupon_pair_count']:,} coupon-campaign pairs; {campaign_checks['coupon_pair_multiple_product_count']:,} map to multiple products, with a maximum of {campaign_checks['max_products_per_coupon_campaign']:,}. This confirms the mapping cannot be joined directly to redemptions without multiplying events.
- Redemption rows are enriched only with campaign fields and match flags. Coupon-campaign match rate: {campaign_checks['redemption_coupon_match_rate']:.2%}. Household campaign-participation match rate: {campaign_checks['redemption_participation_match_rate']:.2%}. No redemption row multiplication occurs.

## Output validation

{outputs}

For the very large promotion panel, missing percentages are accumulated during chunked writing. `average_realized_selling_price` is intentionally missing for unmatched/zero-quantity rows. Product descriptors have no missing values. The panel primary-key assertion is structural: every repeated causal key is removed from the streaming source and appended exactly once after conflict-preserving consolidation.

One transaction store (`STORE_ID` {', '.join(map(str, store_checks['zero_sales_store_ids']))}) has one line and zero total sales. Its sales-share features are undefined and remain missing; the pipeline does not turn a zero denominator into an apparently valid zero share.

## Important limitations

- Transactions cover sampled households only and cannot be interpreted as complete store sales or customer populations.
- The causal table's presence does not prove shelf availability or stock status. An unmatched panel row is not proof of zero consumer demand.
- Promotion codes are preserved but not semantically decoded beyond the explicit `0` versus nonzero indicator assumption.
- The causal table covers only weeks 9–101 and 115 stores, so promotion-panel coverage differs materially from store-profile coverage.
- Demographic profiles describe only observed households with available demographics and may not represent a store's full clientele.
- Campaign participation is not coupon redemption, and redemption does not establish incremental or causal sales impact.
- No observations are silently discarded: causal conflicts are consolidated, exact duplicate coupon mappings are documented, and sparse stores are flagged rather than excluded.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    transactions = pd.read_csv(DATA_DIR / FILES["transaction_data"])
    products = pd.read_csv(DATA_DIR / FILES["product"])
    demographics = pd.read_csv(DATA_DIR / FILES["hh_demographic"])
    campaign_table = pd.read_csv(DATA_DIR / FILES["campaign_table"])
    campaign_desc = pd.read_csv(DATA_DIR / FILES["campaign_desc"])
    coupon = pd.read_csv(DATA_DIR / FILES["coupon"])
    redemptions = pd.read_csv(DATA_DIR / FILES["coupon_redempt"])

    frames = {
        "transaction_data": transactions,
        "product": products,
        "hh_demographic": demographics,
        "campaign_table": campaign_table,
        "campaign_desc": campaign_desc,
        "coupon": coupon,
        "coupon_redempt": redemptions,
    }
    audits = [audit_frame(name, frame, KEYS[name]) for name, frame in frames.items()]
    resolved_duplicates, duplicate_keys, causal_audit = prepare_causal_duplicates()
    audits.insert(2, causal_audit)

    assert products["PRODUCT_ID"].is_unique
    assert demographics["household_key"].is_unique
    assert campaign_desc["CAMPAIGN"].is_unique
    assert not transactions.duplicated(KEYS["transaction_data"]).any()
    assert not transactions.duplicated(["BASKET_ID", "PRODUCT_ID"]).any()
    basket_consistency = transactions.groupby("BASKET_ID")[["household_key", "STORE_ID", "DAY", "WEEK_NO", "TRANS_TIME"]].nunique()
    assert basket_consistency.le(1).all().all()
    assert (transactions["WEEK_NO"] == ((transactions["DAY"] + 1) // 7 + 1)).all()

    store_profiles, store_demographics, store_checks = build_store_outputs(transactions, products, demographics)
    write_parquet(store_profiles, "df_store_profiles.parquet")
    write_parquet(store_demographics, "df_store_demographics.parquet")

    weekly_sales = aggregate_product_store_week(transactions)
    panel_checks = build_promo_panel(weekly_sales, products, resolved_duplicates, duplicate_keys, causal_audit)

    household_campaigns, coupon_redemptions, coupon_mapping, campaign_checks = build_campaign_outputs(
        campaign_table, campaign_desc, coupon, redemptions
    )
    write_parquet(household_campaigns, "df_household_campaigns.parquet")
    write_parquet(coupon_redemptions, "df_coupon_redemptions.parquet")
    write_parquet(coupon_mapping, "df_coupon_product_mapping.parquet")

    validations = [
        dataframe_validation("df_store_profiles", store_profiles, "one row per STORE_ID", ["STORE_ID"]),
        dataframe_validation("df_store_demographics", store_demographics, "one row per STORE_ID", ["STORE_ID"]),
        {
            "name": "df_promo_panel",
            "rows": panel_checks["rows"],
            "columns": panel_checks["columns"],
            "unit": "one row per PRODUCT_ID × STORE_ID × WEEK_NO",
            "key": KEYS["causal_data"],
            "unique": panel_checks["primary_key_unique"],
            "missing_pct": {column: count / panel_checks["rows"] * 100 for column, count in panel_checks["null_counts"].items()},
        },
        dataframe_validation("df_household_campaigns", household_campaigns, "one row per household-campaign", ["household_key", "CAMPAIGN"]),
        dataframe_validation("df_coupon_redemptions", coupon_redemptions, "one row per original redemption event", KEYS["coupon_redempt"]),
        dataframe_validation("df_coupon_product_mapping", coupon_mapping, "one row per coupon-product-campaign mapping", ["COUPON_UPC", "PRODUCT_ID", "CAMPAIGN"]),
    ]
    write_report(audits, validations, store_checks, causal_audit, panel_checks, campaign_checks, transactions)

    # Build the additional product-store-week dataset required by the panel-
    # regression section of analysis.ipynb. Keeping this in the main preparation
    # entry point gives users a single raw-to-analysis-ready command.
    from build_regression_panel import build_regression_panel

    build_regression_panel()

    print("Created:")
    for path in sorted(OUTPUT_DIR.glob("*.parquet")):
        print(f"  {path.relative_to(PROJECT_ROOT)} ({path.stat().st_size / 1024**2:,.1f} MiB)")
    print(f"  {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
