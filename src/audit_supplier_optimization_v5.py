import pandas as pd
import numpy as np

from config import REPORT_DIR


# ============================================================
# STEP 1: AUDIT CONFIGURATION
# ============================================================

BASELINE_CAPACITY_SHARE = 0.50

TIER_CAPACITY_SHARES = {
    "Tier 1": 0.70,
    "Tier 2": 0.20,
    "Tier 3": 0.10
}

TOLERANCE = 1e-6

checks = []


# ============================================================
# STEP 2: AUDIT HELPER FUNCTIONS
# ============================================================

def add_check(name, passed, details=""):

    checks.append({
        "check_name": name,
        "status": "PASS" if passed else "FAIL",
        "details": str(details)
    })


def proportional_integer_allocation(quantities, budget):
    """
    Independently reproduce largest remainder allocation.
    """

    quantities = quantities.astype(int)

    total = int(quantities.sum())

    if budget < 0 or budget > total:
        raise ValueError(
            "Invalid budget in independent allocation."
        )

    if total == 0 or budget == 0:

        return pd.Series(
            0,
            index=quantities.index,
            dtype=int
        )

    exact = (
        quantities / total
    ) * budget

    allocation = np.floor(exact).astype(int)

    remaining = int(
        budget - allocation.sum()
    )

    fractional = exact - allocation

    order = fractional.sort_values(
        ascending=False,
        kind="stable"
    ).index

    for idx in order:

        if remaining <= 0:
            break

        if allocation.loc[idx] < quantities.loc[idx]:

            allocation.loc[idx] += 1
            remaining -= 1

    if remaining != 0:
        raise RuntimeError(
            "Independent largest remainder allocation failed."
        )

    return allocation.astype(int)


def independently_recalculate_policy(
    data,
    total_capacity
):
    """
    Recalculate baseline and tier allocations
    without using the saved V5 allocation columns.
    """

    result = data.copy()

    total_requirement = int(
        result["planned_quantity"].sum()
    )

    baseline_capacity = min(
        int(
            np.floor(
                total_capacity * BASELINE_CAPACITY_SHARE
            )
        ),
        total_requirement
    )

    result["expected_baseline"] = (
        proportional_integer_allocation(
            result["planned_quantity"],
            baseline_capacity
        )
    )

    remaining_capacity = (
        total_capacity - baseline_capacity
    )

    tier_names = list(TIER_CAPACITY_SHARES.keys())

    shares = pd.Series(
        [TIER_CAPACITY_SHARES[t] for t in tier_names],
        index=tier_names,
        dtype=float
    )

    if not np.isclose(shares.sum(), 1.0):
        raise ValueError(
            "Tier shares do not sum to one."
        )

    raw_budgets = (
        shares * remaining_capacity
    )

    tier_budgets = np.floor(
        raw_budgets
    ).astype(int)

    leftover = int(
        remaining_capacity - tier_budgets.sum()
    )

    fractional = raw_budgets - tier_budgets

    for tier in fractional.sort_values(
        ascending=False,
        kind="stable"
    ).index:

        if leftover <= 0:
            break

        tier_budgets.loc[tier] += 1
        leftover -= 1

    result["expected_incremental"] = 0

    carry_forward = 0

    expected_tier_rows = []

    for tier in tier_names:

        tier_indices = result.index[
            result["expected_tier"] == tier
        ]

        remaining_requirement = (
            result.loc[tier_indices, "planned_quantity"]
            - result.loc[tier_indices, "expected_baseline"]
        ).clip(lower=0).astype(int)

        available_requirement = int(
            remaining_requirement.sum()
        )

        nominal_budget = int(
            tier_budgets.loc[tier]
        )

        budget_for_tier = (
            nominal_budget + carry_forward
        )

        actual_budget = min(
            budget_for_tier,
            available_requirement
        )

        received_carry = (
            budget_for_tier - nominal_budget
        )

        carry_forward = (
            budget_for_tier - actual_budget
        )

        if actual_budget > 0:

            extra = proportional_integer_allocation(
                remaining_requirement,
                actual_budget
            )

            result.loc[
                tier_indices,
                "expected_incremental"
            ] = extra

        expected_tier_rows.append({

            "priority_tier": tier,

            "expected_nominal_budget": nominal_budget,

            "expected_carried_capacity_received": (
                received_carry
            ),

            "expected_available_requirement": (
                available_requirement
            ),

            "expected_incremental_allocation": (
                actual_budget
            ),

            "expected_unused_capacity_carried": (
                carry_forward
            )

        })

    expected_tier_df = pd.DataFrame(
        expected_tier_rows
    )

    result["expected_policy_allocation"] = (
        result["expected_baseline"]
        + result["expected_incremental"]
    )

    return (
        result,
        expected_tier_df,
        baseline_capacity,
        remaining_capacity,
        carry_forward
    )


# ============================================================
# STEP 3: LOAD V5 REPORTS
# ============================================================

print("\n========== V5 INDEPENDENT AUDIT ==========")

replenishment = pd.read_csv(
    REPORT_DIR / "replenishment_requirements_v5.csv"
)

shortage = pd.read_csv(
    REPORT_DIR / "procurement_shortage_v5.csv"
)

allocation = pd.read_csv(
    REPORT_DIR / "supplier_allocation_v5.csv"
)

supplier_summary = pd.read_csv(
    REPORT_DIR / "supplier_summary_v5.csv"
)

tier_budget = pd.read_csv(
    REPORT_DIR / "tier_capacity_budget_v5.csv"
)

tier_performance = pd.read_csv(
    REPORT_DIR / "tier_performance_v5.csv"
)

network_summary = pd.read_csv(
    REPORT_DIR / "optimization_summary_v5.csv"
)

print("All V5 reports loaded successfully.")


# ============================================================
# STEP 4: CHECK REQUIRED COLUMNS
# ============================================================

required_columns = {

    "replenishment": [
        "product_id",
        "planned_quantity",
        "priority_score"
    ],

    "shortage": [
        "product_id",
        "priority_rank",
        "priority_tier",
        "priority_score",
        "planned_quantity",
        "baseline_allocation",
        "incremental_allocation",
        "policy_allocation",
        "allocated_quantity",
        "unfulfilled_quantity",
        "product_fill_rate"
    ],

    "allocation": [
        "product_id",
        "supplier_id",
        "allocated_quantity",
        "unit_cost",
        "risk_multiplier",
        "risk_adjusted_unit_cost"
    ],

    "supplier_summary": [
        "supplier_id",
        "capacity_per_cycle",
        "allocated_quantity",
        "unused_capacity",
        "utilization_percent",
        "risk_multiplier"
    ],

    "tier_budget": [
        "priority_tier",
        "nominal_capacity_budget",
        "carried_capacity_received",
        "available_requirement",
        "incremental_allocation",
        "unused_capacity_carried_forward"
    ],

    "tier_performance": [
        "priority_tier",
        "planned_quantity",
        "baseline_allocation",
        "incremental_allocation",
        "allocated_quantity",
        "shortage",
        "fill_rate_percent"
    ]

}

datasets = {
    "replenishment": replenishment,
    "shortage": shortage,
    "allocation": allocation,
    "supplier_summary": supplier_summary,
    "tier_budget": tier_budget,
    "tier_performance": tier_performance
}

for name, columns in required_columns.items():

    df = datasets[name]

    missing = [
        column for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {name}: {missing}"
        )

print("Required columns validated.")


# ============================================================
# STEP 5: BASIC DATA VALIDATION
# ============================================================

add_check(
    "Unique product IDs in replenishment",
    replenishment["product_id"].is_unique
)

add_check(
    "Unique product IDs in shortage",
    shortage["product_id"].is_unique
)

add_check(
    "Unique supplier IDs",
    supplier_summary["supplier_id"].is_unique
)

add_check(
    "No duplicate product-supplier allocation pairs",
    not allocation.duplicated(
        ["product_id", "supplier_id"]
    ).any()
)

add_check(
    "Nonnegative product requirements",
    (replenishment["planned_quantity"] >= 0).all()
)

add_check(
    "Nonnegative allocation quantities",
    (allocation["allocated_quantity"] >= 0).all()
)

add_check(
    "Nonnegative shortages",
    (shortage["unfulfilled_quantity"] >= 0).all()
)


# ============================================================
# STEP 6: CHECK PRODUCT SET CONSISTENCY
# ============================================================

replenishment_products = set(
    replenishment["product_id"]
)

shortage_products = set(
    shortage["product_id"]
)

allocation_products = set(
    allocation["product_id"]
)

add_check(
    "Replenishment and shortage product IDs match",
    replenishment_products == shortage_products
)

add_check(
    "Allocation products are valid",
    allocation_products.issubset(replenishment_products)
)

# Compare planned quantities
planned_check = replenishment[
    ["product_id", "planned_quantity"]
].merge(
    shortage[
        ["product_id", "planned_quantity"]
],
    on="product_id",
    suffixes=("_source", "_shortage"),
    validate="one_to_one"
)

planned_match = np.allclose(
    planned_check["planned_quantity_source"],
    planned_check["planned_quantity_shortage"]
)

add_check(
    "Planned quantities match across reports",
    planned_match
)


# ============================================================
# STEP 7: INDEPENDENTLY RECREATE PRIORITY RANKS
# ============================================================

independent = replenishment.copy()

independent = independent[
    independent["planned_quantity"] > 0
].copy()

independent = independent.sort_values(
    "priority_score",
    ascending=False,
    kind="stable"
).reset_index(drop=True)

n = len(independent)

tier1_end = int(
    np.ceil(n * 0.20)
)

tier2_end = int(
    np.ceil(n * 0.50)
)

independent["expected_rank"] = (
    np.arange(1, n + 1)
)

independent["expected_tier"] = "Tier 3"

independent.loc[
    independent.index < tier1_end,
    "expected_tier"
] = "Tier 1"

independent.loc[
    (independent.index >= tier1_end)
    & (independent.index < tier2_end),
    "expected_tier"
] = "Tier 2"

rank_check = independent.merge(
    shortage[
        [
            "product_id",
            "priority_rank",
            "priority_tier",
            "priority_score"
        ]
    ],
    on="product_id",
    suffixes=("_expected", "_actual"),
    validate="one_to_one"
)

rank_correct = (
    rank_check["expected_rank"]
    == rank_check["priority_rank_actual"]
).all()

tier_correct = (
    rank_check["expected_tier"]
    == rank_check["priority_tier_actual"]
).all()

priority_score_correct = np.allclose(
    rank_check["priority_score_expected"],
    rank_check["priority_score_actual"],
    atol=TOLERANCE
)

add_check(
    "Priority ranking independently verified",
    rank_correct
)

add_check(
    "Priority tier assignment independently verified",
    tier_correct
)

add_check(
    "Priority scores consistent across reports",
    priority_score_correct
)


# ============================================================
# STEP 8: INDEPENDENTLY RECREATE CAPACITY POLICY
# ============================================================

total_capacity = int(
    supplier_summary["capacity_per_cycle"].sum()
)

(
    expected_policy,
    expected_tier_budget,
    expected_baseline_capacity,
    expected_incremental_capacity,
    expected_unused_capacity
) = independently_recalculate_policy(
    independent,
    total_capacity
)


# ------------------------------------------------------------
# 8.1 Compare baseline allocations
# ------------------------------------------------------------

baseline_comparison = expected_policy[
    ["product_id", "expected_baseline"]
].merge(
    shortage[
        ["product_id", "baseline_allocation"]
    ],
    on="product_id",
    validate="one_to_one"
)

baseline_correct = (
    baseline_comparison["expected_baseline"]
    == baseline_comparison["baseline_allocation"]
).all()

add_check(
    "Baseline allocation independently recalculated",
    baseline_correct
)


# ------------------------------------------------------------
# 8.2 Compare incremental allocations
# ------------------------------------------------------------

incremental_comparison = expected_policy[
    ["product_id", "expected_incremental"]
].merge(
    shortage[
        ["product_id", "incremental_allocation"]
    ],
    on="product_id",
    validate="one_to_one"
)

incremental_correct = (
    incremental_comparison["expected_incremental"]
    == incremental_comparison["incremental_allocation"]
).all()

add_check(
    "Tier incremental allocation independently recalculated",
    incremental_correct
)


# ------------------------------------------------------------
# 8.3 Compare final policy allocations
# ------------------------------------------------------------

policy_comparison = expected_policy[
    ["product_id", "expected_policy_allocation"]
].merge(
    shortage[
        ["product_id", "policy_allocation"]
    ],
    on="product_id",
    validate="one_to_one"
)

policy_correct = (
    policy_comparison["expected_policy_allocation"]
    == policy_comparison["policy_allocation"]
).all()

add_check(
    "Final policy allocation independently verified",
    policy_correct
)

print("\nCapacity policy independently recalculated.")

# ============================================================
# STEP 9: TIER BUDGET VALIDATION
# ============================================================

budget_comparison = expected_tier_budget.merge(
    tier_budget,
    on="priority_tier",
    suffixes=("_expected", "_actual"),
    validate="one_to_one"
)

budget_columns = [
    (
        "expected_nominal_budget",
        "nominal_capacity_budget"
    ),
    (
        "expected_carried_capacity_received",
        "carried_capacity_received"
    ),
    (
        "expected_available_requirement",
        "available_requirement"
    ),
    (
        "expected_incremental_allocation",
        "incremental_allocation"
    ),
    (
        "expected_unused_capacity_carried",
        "unused_capacity_carried_forward"
    )
]

for expected_column, actual_column in budget_columns:

    passed = (
        budget_comparison[expected_column]
        == budget_comparison[actual_column]
    ).all()

    add_check(
        f"Tier budget: {actual_column}",
        passed
    )


# ============================================================
# STEP 10: FAIRNESS VALIDATION
# ============================================================

# Fairness means proportional allocation to remaining
# requirement within each tier, subject to integer rounding.

fairness_correct = (
    incremental_correct
)

add_check(
    "Within-tier proportional fairness",
    fairness_correct,
    "Expected incremental allocations independently reproduced."
)

# Compare incremental fill rates inside each tier
fairness_rows = []

for tier in ["Tier 1", "Tier 2", "Tier 3"]:

    tier_data = shortage[
        shortage["priority_tier"] == tier
    ].copy()

    tier_data["remaining_requirement"] = (
        tier_data["planned_quantity"]
        - tier_data["baseline_allocation"]
    )

    tier_data["incremental_fill_rate"] = np.where(
        tier_data["remaining_requirement"] > 0,
        tier_data["incremental_allocation"]
        / tier_data["remaining_requirement"],
        0
    )

    fairness_rows.append({

        "priority_tier": tier,

        "minimum_incremental_fill_rate": (
            tier_data["incremental_fill_rate"].min()
        ),

        "maximum_incremental_fill_rate": (
            tier_data["incremental_fill_rate"].max()
        ),

        "fill_rate_difference": (
            tier_data["incremental_fill_rate"].max()
            - tier_data["incremental_fill_rate"].min()
        )

    })

fairness_report = pd.DataFrame(
    fairness_rows
)

fairness_report.to_csv(
    REPORT_DIR / "audit_v5_fairness.csv",
    index=False
)


# ============================================================
# STEP 11: PRODUCT ALLOCATION AUDIT
# ============================================================

allocation_by_product = (
    allocation.groupby("product_id")[
        "allocated_quantity"
    ]
    .sum()
    .reset_index()
    .rename(
        columns={
            "allocated_quantity": "independent_allocated_quantity"
        }
    )
)

product_audit = shortage.merge(
    allocation_by_product,
    on="product_id",
    how="left"
)

product_audit[
    "independent_allocated_quantity"
] = product_audit[
    "independent_allocated_quantity"
].fillna(0).astype(int)

product_audit["demand_balance"] = (
    product_audit["allocated_quantity"]
    + product_audit["unfulfilled_quantity"]
)

product_audit["independent_shortage"] = (
    product_audit["planned_quantity"]
    - product_audit["independent_allocated_quantity"]
)

product_audit["independent_fill_rate"] = np.where(
    product_audit["planned_quantity"] > 0,
    product_audit["independent_allocated_quantity"]
    / product_audit["planned_quantity"] * 100,
    100
)

product_allocation_match = (
    product_audit["allocated_quantity"]
    == product_audit["independent_allocated_quantity"]
).all()

product_shortage_match = (
    product_audit["unfulfilled_quantity"]
    == product_audit["independent_shortage"]
).all()

product_balance_match = (
    product_audit["demand_balance"]
    == product_audit["planned_quantity"]
).all()

product_fill_match = np.allclose(
    product_audit["product_fill_rate"],
    product_audit["independent_fill_rate"],
    atol=TOLERANCE
)

add_check(
    "Product allocation matches supplier allocation",
    product_allocation_match
)

add_check(
    "Product shortage independently verified",
    product_shortage_match
)

add_check(
    "Product demand balance",
    product_balance_match
)

add_check(
    "Product fill rates independently verified",
    product_fill_match
)


# ============================================================
# STEP 12: SUPPLIER CAPACITY AUDIT
# ============================================================

allocation_by_supplier = (
    allocation.groupby("supplier_id")[
        "allocated_quantity"
    ]
    .sum()
    .reset_index()
    .rename(
        columns={
            "allocated_quantity": "independent_allocated_quantity"
        }
    )
)

supplier_audit = supplier_summary.merge(
    allocation_by_supplier,
    on="supplier_id",
    how="left"
)

supplier_audit[
    "independent_allocated_quantity"
] = supplier_audit[
    "independent_allocated_quantity"
].fillna(0).astype(int)

supplier_audit["independent_unused_capacity"] = (
    supplier_audit["capacity_per_cycle"]
    - supplier_audit["independent_allocated_quantity"]
)

supplier_audit["independent_utilization"] = np.where(
    supplier_audit["capacity_per_cycle"] > 0,
    supplier_audit["independent_allocated_quantity"]
    / supplier_audit["capacity_per_cycle"] * 100,
    0
)

supplier_allocation_match = (
    supplier_audit["allocated_quantity"]
    == supplier_audit["independent_allocated_quantity"]
).all()

supplier_capacity_pass = (
    supplier_audit["independent_allocated_quantity"]
    <= supplier_audit["capacity_per_cycle"]
).all()

supplier_unused_match = (
    supplier_audit["unused_capacity"]
    == supplier_audit["independent_unused_capacity"]
).all()

supplier_utilization_match = np.allclose(
    supplier_audit["utilization_percent"],
    supplier_audit["independent_utilization"],
    atol=TOLERANCE
)

add_check(
    "Supplier allocation matches detailed report",
    supplier_allocation_match
)

add_check(
    "Supplier capacity constraints",
    supplier_capacity_pass
)

add_check(
    "Unused supplier capacity verified",
    supplier_unused_match
)

add_check(
    "Supplier utilization independently verified",
    supplier_utilization_match
)


# ============================================================
# STEP 13: PROCUREMENT COST AUDIT
# ============================================================

independent_base_cost = (
    allocation["allocated_quantity"]
    * allocation["unit_cost"]
).sum()

independent_risk_cost = (
    allocation["allocated_quantity"]
    * allocation["risk_adjusted_unit_cost"]
).sum()

independent_risk_premium = (
    independent_risk_cost - independent_base_cost
)

# Check risk-adjusted unit cost calculation
expected_risk_adjusted_unit_cost = (
    allocation["unit_cost"]
    * allocation["risk_multiplier"]
)

risk_unit_cost_match = np.allclose(
    allocation["risk_adjusted_unit_cost"],
    expected_risk_adjusted_unit_cost,
    atol=TOLERANCE
)

add_check(
    "Risk-adjusted unit cost calculation",
    risk_unit_cost_match
)

# Check supplier risk multipliers against supplier summary
risk_comparison = allocation.merge(
    supplier_summary[
        [
            "supplier_id",
            "risk_multiplier"
        ]
    ],
    on="supplier_id",
    suffixes=("_allocation", "_supplier"),
    validate="many_to_one"
)

supplier_risk_match = np.allclose(
    risk_comparison["risk_multiplier_allocation"],
    risk_comparison["risk_multiplier_supplier"],
    atol=TOLERANCE
)

add_check(
    "Supplier risk multipliers match",
    supplier_risk_match
)

# Compare costs with network summary
reported_summary = network_summary.iloc[0]

base_cost_match = np.isclose(
    independent_base_cost,
    reported_summary["base_procurement_cost"],
    atol=0.01
)

risk_cost_match = np.isclose(
    independent_risk_cost,
    reported_summary["risk_adjusted_procurement_cost"],
    atol=0.01
)

risk_premium_match = np.isclose(
    independent_risk_premium,
    reported_summary["risk_premium"],
    atol=0.01
)

add_check(
    "Base procurement cost independently verified",
    base_cost_match
)

add_check(
    "Risk-adjusted procurement cost independently verified",
    risk_cost_match
)

add_check(
    "Supplier risk premium independently verified",
    risk_premium_match
)


# ============================================================
# STEP 14: NETWORK-LEVEL AUDIT
# ============================================================

total_planned = int(
    shortage["planned_quantity"].sum()
)

total_allocated = int(
    shortage["allocated_quantity"].sum()
)

total_shortage = int(
    shortage["unfulfilled_quantity"].sum()
)

total_supplier_capacity = int(
    supplier_summary["capacity_per_cycle"].sum()
)

total_allocation_detail = int(
    allocation["allocated_quantity"].sum()
)

total_supplier_allocation = int(
    supplier_summary["allocated_quantity"].sum()
)

network_fill_rate = (
    total_allocated / total_planned * 100
    if total_planned > 0
    else 100
)

maximum_capacity_coverage = (
    min(total_supplier_capacity, total_planned)
    / total_planned * 100
    if total_planned > 0
    else 100
)

add_check(
    "Network demand balance",
    total_allocated + total_shortage == total_planned
)

add_check(
    "Detailed allocation total",
    total_allocation_detail == total_allocated
)

add_check(
    "Supplier summary total",
    total_supplier_allocation == total_allocated
)

add_check(
    "Total allocation within total capacity",
    total_allocated <= total_supplier_capacity
)

add_check(
    "Network summary planned quantity",
    total_planned == reported_summary["total_planned_quantity"]
)

add_check(
    "Network summary allocated quantity",
    total_allocated == reported_summary["total_allocated_quantity"]
)

add_check(
    "Network summary shortage",
    total_shortage == reported_summary["total_shortage"]
)

add_check(
    "Network fill rate",
    np.isclose(
        network_fill_rate,
        reported_summary["network_fill_rate_percent"],
        atol=TOLERANCE
    )
)

add_check(
    "Maximum capacity coverage",
    np.isclose(
        maximum_capacity_coverage,
        reported_summary["maximum_capacity_coverage_percent"],
        atol=TOLERANCE
    )
)


# ============================================================
# STEP 15: SAVE INDEPENDENT AUDIT REPORTS
# ============================================================

audit_results = pd.DataFrame(checks)

product_audit.to_csv(
    REPORT_DIR / "audit_product_checks_v5.csv",
    index=False
)

supplier_audit.to_csv(
    REPORT_DIR / "audit_supplier_checks_v5.csv",
    index=False
)

audit_results.to_csv(
    REPORT_DIR / "audit_checks_v5.csv",
    index=False
)


# ============================================================
# STEP 16: FINAL AUDIT SUMMARY
# ============================================================

passed = int(
    (audit_results["status"] == "PASS").sum()
)

failed = int(
    (audit_results["status"] == "FAIL").sum()
)

total_checks = len(audit_results)

print("\n========== AUDIT CHECK RESULTS ==========")

for _, row in audit_results.iterrows():

    symbol = (
        "PASS"
        if row["status"] == "PASS"
        else "FAIL"
    )

    print(
        f"{symbol}: {row['check_name']}"
    )

print("\n========== AUDIT SUMMARY ==========")

print("Total checks:", total_checks)
print("Passed:", passed)
print("Failed:", failed)

print(
    f"Independent network fill rate: "
    f"{network_fill_rate:.2f}%"
)

print(
    f"Independent base procurement cost: "
    f"Rs. {independent_base_cost:,.2f}"
)

print(
    f"Independent risk premium: "
    f"Rs. {independent_risk_premium:,.2f}"
)

print(
    f"Independent risk-adjusted cost: "
    f"Rs. {independent_risk_cost:,.2f}"
)

print("\n========== FAIRNESS SUMMARY ==========")

print(
    fairness_report.to_string(index=False)
)

if failed == 0:

    print("\nALL INDEPENDENT AUDIT CHECKS PASSED!")

else:

    print(
        f"\nAUDIT COMPLETED WITH {failed} FAILED CHECK(S)."
    )

print("\nSaved reports:")
print("audit_checks_v5.csv")
print("audit_product_checks_v5.csv")
print("audit_supplier_checks_v5.csv")
print("audit_v5_fairness.csv")

print("\nV5 INDEPENDENT AUDIT COMPLETED!")
