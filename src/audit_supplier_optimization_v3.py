import pandas as pd
import numpy as np

from config import REPORT_DIR


# ============================================================
# STEP 1: LOAD V3 OUTPUT FILES
# ============================================================

print("\nLoading V3 optimization reports...")

requirements = pd.read_csv(
    REPORT_DIR / "replenishment_requirements_v3.csv"
)

allocation = pd.read_csv(
    REPORT_DIR / "supplier_allocation_v3.csv"
)

shortage = pd.read_csv(
    REPORT_DIR / "procurement_shortage_v3.csv"
)

supplier_summary = pd.read_csv(
    REPORT_DIR / "supplier_summary_v3.csv"
)

print("All V3 reports loaded successfully!")


# ============================================================
# STEP 2: VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = {
    "requirements": [
        "product_id",
        "planned_quantity"
    ],
    "allocation": [
        "product_id",
        "supplier_id",
        "allocated_quantity"
    ],
    "shortage": [
        "product_id",
        "planned_quantity",
        "allocated_quantity",
        "unfulfilled_quantity",
        "product_fill_rate",
        "priority_score"
    ],
    "supplier_summary": [
        "supplier_id",
        "capacity_per_cycle",
        "allocated_quantity",
        "unused_capacity",
        "utilization_percent"
    ]
}

datasets = {
    "requirements": requirements,
    "allocation": allocation,
    "shortage": shortage,
    "supplier_summary": supplier_summary
}

for dataset_name, columns in required_columns.items():

    for column in columns:

        if column not in datasets[dataset_name].columns:

            raise ValueError(
                f"Missing column '{column}' "
                f"in {dataset_name} report."
            )

print("All required columns verified!")


# ============================================================
# STEP 3: BASIC DATA VALIDATION
# ============================================================

print("\n========== BASIC DATA VALIDATION ==========")

checks = []

def add_check(name, passed, details):

    checks.append({
        "check_name": name,
        "status": "PASS" if passed else "FAIL",
        "details": details
    })

# Check duplicate product IDs
add_check(
    "Unique replenishment product IDs",
    not requirements["product_id"].duplicated().any(),
    "Each product should appear once."
)

# Check duplicate product IDs in shortage report
add_check(
    "Unique shortage product IDs",
    not shortage["product_id"].duplicated().any(),
    "Each product should appear once."
)

# Check duplicate supplier IDs
add_check(
    "Unique supplier IDs",
    not supplier_summary["supplier_id"].duplicated().any(),
    "Each supplier should appear once."
)

# Check negative allocation
add_check(
    "Non-negative allocations",
    (allocation["allocated_quantity"] >= 0).all(),
    "Allocated quantities must not be negative."
)

# Check negative shortage
add_check(
    "Non-negative shortage",
    (shortage["unfulfilled_quantity"] >= 0).all(),
    "Unfulfilled quantities must not be negative."
)


# ============================================================
# STEP 4: PRODUCT-WISE ALLOCATION AUDIT
# ============================================================

print("\n========== PRODUCT ALLOCATION AUDIT ==========")

# Aggregate supplier allocations for every product
allocated_by_product = (
    allocation.groupby("product_id")["allocated_quantity"]
    .sum()
    .rename("calculated_allocated")
)

# Combine planned requirements and shortage report
product_audit = requirements[
    ["product_id", "planned_quantity"]
].merge(
    shortage[
        [
            "product_id",
            "allocated_quantity",
            "unfulfilled_quantity",
            "product_fill_rate",
            "priority_score"
        ]
    ],
    on="product_id",
    how="left",
    validate="one_to_one"
)

# Compare allocation file with shortage file
product_audit = product_audit.merge(
    allocated_by_product,
    on="product_id",
    how="left"
)

product_audit["calculated_allocated"] = (
    product_audit["calculated_allocated"]
    .fillna(0)
)

product_audit["allocated_quantity"] = (
    product_audit["allocated_quantity"].fillna(0)
)

product_audit["unfulfilled_quantity"] = (
    product_audit["unfulfilled_quantity"].fillna(0)
)

# Check product-level quantity balance
product_audit["quantity_balance"] = (
    product_audit["allocated_quantity"]
    + product_audit["unfulfilled_quantity"]
)

product_audit["balance_difference"] = (
    product_audit["planned_quantity"]
    - product_audit["quantity_balance"]
)

balance_valid = np.isclose(
    product_audit["planned_quantity"],
    product_audit["quantity_balance"],
    atol=1e-6
).all()

add_check(
    "Product demand balance",
    balance_valid,
    "Planned quantity = Allocated quantity + Unfulfilled quantity."
)

# Check allocation consistency between two reports
allocation_valid = np.isclose(
    product_audit["allocated_quantity"],
    product_audit["calculated_allocated"],
    atol=1e-6
).all()

add_check(
    "Allocation report consistency",
    allocation_valid,
    "Allocation CSV totals must match shortage CSV."
)

# Calculate fill rate independently
product_audit["calculated_fill_rate"] = np.where(
    product_audit["planned_quantity"] > 0,
    (
        product_audit["allocated_quantity"]
        / product_audit["planned_quantity"]
    ) * 100,
    100
)

fill_rate_valid = np.isclose(
    product_audit["product_fill_rate"],
    product_audit["calculated_fill_rate"],
    atol=0.01
).all()

add_check(
    "Product fill rate calculation",
    fill_rate_valid,
    "Fill rate is independently recalculated."
)

print(
    product_audit[
        [
            "product_id",
            "planned_quantity",
            "allocated_quantity",
            "unfulfilled_quantity",
            "calculated_fill_rate"
        ]
    ].head(15).to_string(index=False)
)


# ============================================================
# STEP 5: SUPPLIER CAPACITY AUDIT
# ============================================================

print("\n========== SUPPLIER CAPACITY AUDIT ==========")

allocated_by_supplier = (
    allocation.groupby("supplier_id")["allocated_quantity"]
    .sum()
    .rename("calculated_allocation")
)

supplier_audit = supplier_summary.merge(
    allocated_by_supplier,
    on="supplier_id",
    how="left"
)

supplier_audit["calculated_allocation"] = (
    supplier_audit["calculated_allocation"].fillna(0)
)

# Capacity validation
supplier_audit["capacity_valid"] = (
    supplier_audit["calculated_allocation"]
    <= supplier_audit["capacity_per_cycle"]
)

capacity_valid = supplier_audit["capacity_valid"].all()

add_check(
    "Supplier capacity constraints",
    capacity_valid,
    "Allocation must not exceed supplier capacity."
)

# Allocation report consistency
supplier_allocation_valid = np.isclose(
    supplier_audit["allocated_quantity"],
    supplier_audit["calculated_allocation"],
    atol=1e-6
).all()

add_check(
    "Supplier allocation consistency",
    supplier_allocation_valid,
    "Supplier summary must match allocation CSV."
)

# Recalculate unused capacity
supplier_audit["calculated_unused_capacity"] = (
    supplier_audit["capacity_per_cycle"]
    - supplier_audit["calculated_allocation"]
)

unused_valid = np.isclose(
    supplier_audit["unused_capacity"],
    supplier_audit["calculated_unused_capacity"],
    atol=1e-6
).all()

add_check(
    "Unused supplier capacity",
    unused_valid,
    "Unused capacity is independently recalculated."
)

# Recalculate utilization
supplier_audit["calculated_utilization"] = np.where(
    supplier_audit["capacity_per_cycle"] > 0,
    (
        supplier_audit["calculated_allocation"]
        / supplier_audit["capacity_per_cycle"]
    ) * 100,
    0
)

utilization_valid = np.isclose(
    supplier_audit["utilization_percent"],
    supplier_audit["calculated_utilization"],
    atol=0.01
).all()

add_check(
    "Supplier utilization calculation",
    utilization_valid,
    "Supplier utilization is independently recalculated."
)

print(
    supplier_audit[
        [
            "supplier_id",
            "capacity_per_cycle",
            "calculated_allocation",
            "calculated_unused_capacity",
            "calculated_utilization"
        ]
    ].to_string(index=False)
)


# ============================================================
# STEP 6: PRIORITY ANALYSIS
# ============================================================

print("\n========== PRIORITY ANALYSIS ==========")

product_audit = product_audit.sort_values(
    "priority_score",
    ascending=False
)

product_audit["allocation_status"] = np.select(
    [
        product_audit["allocated_quantity"] == 0,
        product_audit["unfulfilled_quantity"] == 0
    ],
    [
        "No Allocation",
        "Fully Fulfilled"
    ],
    default="Partially Fulfilled"
)

print("\nTop 15 priority products:")

print(
    product_audit[
        [
            "product_id",
            "priority_score",
            "planned_quantity",
            "allocated_quantity",
            "allocation_status"
        ]
    ].head(15).to_string(index=False)
)

# Descriptive diagnostic, not a mathematical constraint
top_priority_count = min(10, len(product_audit))

top_priority_fulfilled = (
    product_audit.head(top_priority_count)[
        "allocated_quantity"
    ].sum()
)

add_check(
    "Priority allocation diagnostic",
    top_priority_fulfilled > 0,
    (
        f"Top {top_priority_count} priority products received "
        f"{top_priority_fulfilled:.0f} units. "
        "This is a diagnostic, not a solver constraint."
    )
)


# ============================================================
# STEP 7: NETWORK PERFORMANCE METRICS
# ============================================================

print("\n========== NETWORK PERFORMANCE ==========")

total_planned = product_audit["planned_quantity"].sum()

total_allocated = product_audit["allocated_quantity"].sum()

total_shortage = product_audit["unfulfilled_quantity"].sum()

total_capacity = supplier_audit["capacity_per_cycle"].sum()

network_fill_rate = (
    total_allocated / total_planned * 100
    if total_planned > 0 else 100
)

maximum_capacity_fill_rate = (
    total_capacity / total_planned * 100
    if total_planned > 0 else 0
)

network_balance_valid = np.isclose(
    total_planned,
    total_allocated + total_shortage,
    atol=1e-6
)

add_check(
    "Network demand balance",
    network_balance_valid,
    "Total planned = Total allocated + Total shortage."
)

add_check(
    "Allocation within total capacity",
    total_allocated <= total_capacity + 1e-6,
    "Total allocation cannot exceed total supplier capacity."
)

print("Total planned quantity:", total_planned)
print("Total allocated quantity:", total_allocated)
print("Total shortage:", total_shortage)
print("Total supplier capacity:", total_capacity)

print(f"Network fill rate: {network_fill_rate:.2f}%")

print(
    f"Maximum theoretical capacity coverage: "
    f"{maximum_capacity_fill_rate:.2f}%"
)


# ============================================================
# STEP 8: FINAL AUDIT REPORT
# ============================================================

audit_results = pd.DataFrame(checks)

passed = (audit_results["status"] == "PASS").sum()

failed = (audit_results["status"] == "FAIL").sum()

print("\n========== FINAL AUDIT RESULTS ==========")

print(audit_results.to_string(index=False))

print("\nTotal checks:", len(audit_results))
print("Passed:", passed)
print("Failed:", failed)

if failed == 0:
    print("\nALL STRUCTURAL AND MATHEMATICAL AUDIT CHECKS PASSED!")
else:
    print("\nAUDIT ISSUES FOUND. Review failed checks.")


# ============================================================
# STEP 9: SAVE AUDIT REPORTS
# ============================================================

audit_results.to_csv(
    REPORT_DIR / "audit_checks_v3.csv",
    index=False
)

product_audit.to_csv(
    REPORT_DIR / "audit_product_checks_v3.csv",
    index=False
)

supplier_audit.to_csv(
    REPORT_DIR / "audit_supplier_checks_v3.csv",
    index=False
)

print("\n========== AUDIT REPORTS SAVED ==========")

print("audit_checks_v3.csv")
print("audit_product_checks_v3.csv")
print("audit_supplier_checks_v3.csv")

print("\nV3 AUDIT COMPLETED!")
