
import pandas as pd
from pathlib import Path

from config import REPORT_DIR

# Load results from both optimization versions
v1 = pd.read_csv(
    REPORT_DIR / "supplier_summary.csv"
)

v2 = pd.read_csv(
    REPORT_DIR / "supplier_summary_v2.csv"
)
print("\nV1 Columns:")
print(v1.columns.tolist())

print("\nV2 Columns:")
print(v2.columns.tolist())

# Compare supplier utilization

comparison = v1.merge(
    v2,
    on="supplier_id",
    suffixes=("_v1", "_v2")
)

print("\n========== SUPPLIER COMPARISON ==========")

print(comparison.to_string(index=False))

# Compare product-level allocations
allocation_v1 = pd.read_csv(
    REPORT_DIR / "supplier_allocation.csv"
)

allocation_v2 = pd.read_csv(
    REPORT_DIR / "supplier_allocation_v2.csv"
)

product_v1 = (
    allocation_v1.groupby("product_id")
    ["allocated_quantity"].sum()
    .rename("allocation_v1")
)

product_v2 = (
    allocation_v2.groupby("product_id")
    ["allocated_quantity"].sum()
    .rename("allocation_v2")
)

product_comparison = pd.concat(
    [product_v1, product_v2],
    axis=1
).fillna(0)

product_comparison["allocation_change"] = (
    product_comparison["allocation_v2"]
    - product_comparison["allocation_v1"]
)

product_comparison = product_comparison.sort_values(
    "allocation_change",
    ascending=False
)

print("\n========== TOP 10 ALLOCATION CHANGES ==========")

print(product_comparison.head(10).to_string())

# Save comparison
product_comparison.to_csv(
    REPORT_DIR / "supplier_allocation_comparison.csv"
)

print("\nComparison report saved successfully!")
