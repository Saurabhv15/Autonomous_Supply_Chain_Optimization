
import pandas as pd

from config import REPORT_DIR

# Load V3 results
requirements = pd.read_csv(
    REPORT_DIR / "replenishment_requirements_v3.csv"
)

shortages = pd.read_csv(
    REPORT_DIR / "procurement_shortage_v2.csv"
)

allocations = pd.read_csv(
    REPORT_DIR / "supplier_allocation_v2.csv"
)

# Product-wise allocation
allocated = (
    allocations.groupby("product_id")["allocated_quantity"]
    .sum()
    .reset_index()
)

# Merge requirement and allocation
audit = requirements.merge(
    allocated,
    on="product_id",
    how="left"
)

audit["allocated_quantity"] = (
    audit["allocated_quantity"].fillna(0)
)

audit["unfulfilled_quantity"] = (
    audit["planned_quantity"]
    - audit["allocated_quantity"]
)

audit["fill_rate_percent"] = (
    audit["allocated_quantity"]
    / audit["planned_quantity"].replace(0, 1)
) * 100

# Sort by highest replenishment requirement
audit = audit.sort_values(
    "planned_quantity",
    ascending=False
)

print("\n========== V3 REPLENISHMENT AUDIT ==========")

print(
    audit[
        [
            "product_id",
            "planned_quantity",
            "allocated_quantity",
            "unfulfilled_quantity",
            "fill_rate_percent"
        ]
    ].head(20).to_string(index=False)
)

print("\n========== SUMMARY ==========")

print("Total required:",
      int(audit["planned_quantity"].sum()))

print("Total allocated:",
      int(audit["allocated_quantity"].sum()))

print("Total unfulfilled:",
      int(audit["unfulfilled_quantity"].sum()))

print("Overall fill rate:",
      round(
          audit["allocated_quantity"].sum()
          / audit["planned_quantity"].sum() * 100,
          2
      ), "%")

audit.to_csv(
    REPORT_DIR / "replenishment_audit_v3.csv",
    index=False
)

print("\nAudit report saved successfully!")


# ==========================================
# ALLOCATION DIAGNOSTICS
# ==========================================

# Products that received allocation
allocated_products = audit[
    audit["allocated_quantity"] > 0
].sort_values(
    "allocated_quantity",
    ascending=False
)

# Products that received no allocation
unallocated_products = audit[
    audit["allocated_quantity"] == 0
].sort_values(
    "planned_quantity",
    ascending=False
)

print("\n========== PRODUCTS WITH ALLOCATION ==========")

print(
    allocated_products[
        [
            "product_id",
            "planned_quantity",
            "allocated_quantity",
            "unfulfilled_quantity",
            "fill_rate_percent"
        ]
    ].to_string(index=False)
)

print("\nNumber of products receiving allocation:",
      len(allocated_products))

print("\nNumber of products receiving zero allocation:",
      len(unallocated_products))

print("\n========== ALLOCATION SUMMARY ==========")

print("Highest allocated product:")
print(allocated_products.head(1).to_string(index=False))

print("\nHighest unallocated requirement:")
print(unallocated_products.head(1).to_string(index=False))
