import pandas as pd
import numpy as np
import pyomo.environ as pyo

from config import RAW_DATA_DIR, REPORT_DIR


# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

BASELINE_CAPACITY_SHARE = 0.50

# Split of remaining capacity across priority tiers.
# These are configurable policy assumptions.
TIER_CAPACITY_SHARES = {
    "Tier 1": 0.70,
    "Tier 2": 0.20,
    "Tier 3": 0.10
}

SOLVER_NAME = "appsi_highs"


# ============================================================
# STEP 2: HELPER FUNCTIONS
# ============================================================

def proportional_integer_allocation(quantities, budget):
    """
    Distribute an integer budget proportionally to quantities.

    Uses the largest remainder method.
    Never allocates more than the available quantity.
    """

    quantities = quantities.astype(int)

    total_quantity = int(quantities.sum())

    if budget < 0:
        raise ValueError("Budget cannot be negative.")

    if budget > total_quantity:
        raise ValueError(
            "Budget cannot exceed total available quantity."
        )

    if total_quantity == 0 or budget == 0:
        return pd.Series(
            0,
            index=quantities.index,
            dtype=int
        )

    exact = (
        quantities / total_quantity
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

        if remaining == 0:
            break

        if allocation.loc[idx] < quantities.loc[idx]:

            allocation.loc[idx] += 1
            remaining -= 1

    if remaining != 0:
        raise RuntimeError(
            "Largest remainder allocation failed."
        )

    return allocation.astype(int)


def allocate_by_priority_tier(
    data,
    incremental_capacity,
    tier_shares
):
    """
    Allocate incremental capacity tier by tier.

    Within each tier, allocation is proportional to
    remaining product requirement.

    Unused capacity from a tier is passed to subsequent tiers.
    """

    data = data.copy()

    data["incremental_allocation"] = 0

    tier_budget_rows = []

    # Initial integer budgets for all tiers
    tier_names = list(tier_shares.keys())

    share_values = pd.Series(
        [tier_shares[t] for t in tier_names],
        index=tier_names,
        dtype=float
    )

    if not np.isclose(share_values.sum(), 1.0):
        raise ValueError(
            "Tier capacity shares must sum to 1."
        )

    raw_budgets = (
        share_values * incremental_capacity
    )

    tier_budgets = np.floor(
        raw_budgets
    ).astype(int)

    remaining_budget = int(
        incremental_capacity - tier_budgets.sum()
    )

    fractional = raw_budgets - tier_budgets

    for tier in fractional.sort_values(
        ascending=False,
        kind="stable"
    ).index:

        if remaining_budget == 0:
            break

        tier_budgets.loc[tier] += 1
        remaining_budget -= 1

    # Distribute capacity in priority order
    carry_forward = 0

    for tier in tier_names:

        tier_mask = (
            data["priority_tier"] == tier
        )

        tier_indices = data.index[tier_mask]

        tier_remaining = (
            data.loc[tier_indices, "planned_quantity"]
            - data.loc[tier_indices, "baseline_allocation"]
        ).clip(lower=0).astype(int)

        available_requirement = int(
            tier_remaining.sum()
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

        carry_forward = (
            budget_for_tier - actual_budget
        )

        if actual_budget > 0:

            extra = proportional_integer_allocation(
                tier_remaining,
                actual_budget
            )

            data.loc[
                tier_indices,
                "incremental_allocation"
            ] = extra

        tier_budget_rows.append({

            "priority_tier": tier,

            "nominal_capacity_budget": nominal_budget,

            "carried_capacity_received": (
                budget_for_tier - nominal_budget
            ),

            "available_requirement": (
                available_requirement
            ),

            "incremental_allocation": (
                actual_budget
            ),

            "unused_capacity_carried_forward": (
                carry_forward
            )

        })

    # If all tiers are fulfilled, unused capacity remains unused.
    tier_budget_df = pd.DataFrame(
        tier_budget_rows
    )

    return data, tier_budget_df, carry_forward


# ============================================================
# STEP 3: LOAD DATA
# ============================================================

print("\n========== V5: HYBRID PRIORITY ALLOCATION ==========")

products = pd.read_csv(
    RAW_DATA_DIR / "products.csv"
)

suppliers = pd.read_csv(
    RAW_DATA_DIR / "suppliers.csv"
)

inventory = pd.read_csv(
    REPORT_DIR / "inventory_policy_results.csv"
)

print("Datasets loaded successfully!")


# ============================================================
# STEP 4: INPUT VALIDATION
# ============================================================

required_product_columns = [
    "product_id",
    "unit_cost",
    "selling_price"
]

required_supplier_columns = [
    "supplier_id",
    "reliability",
    "lead_time_days",
    "capacity_per_cycle"
]

required_inventory_columns = [
    "product_id",
    "average_daily_demand",
    "current_network_stock",
    "reorder_point",
    "EOQ"
]

datasets = {
    "products": (products, required_product_columns),
    "suppliers": (suppliers, required_supplier_columns),
    "inventory": (inventory, required_inventory_columns)
}

for name, (df, columns) in datasets.items():

    for column in columns:

        if column not in df.columns:
            raise ValueError(
                f"Missing column '{column}' in {name}."
            )

    if df[columns].isna().any().any():
        raise ValueError(
            f"Missing values found in {name}."
        )

if products["product_id"].duplicated().any():
    raise ValueError("Duplicate product IDs.")

if suppliers["supplier_id"].duplicated().any():
    raise ValueError("Duplicate supplier IDs.")

if inventory["product_id"].duplicated().any():
    raise ValueError("Duplicate inventory product IDs.")

numeric_columns = {
    "products": [
        "unit_cost",
        "selling_price"
    ],
    "suppliers": [
        "reliability",
        "lead_time_days",
        "capacity_per_cycle"
    ],
    "inventory": [
        "average_daily_demand",
        "current_network_stock",
        "reorder_point",
        "EOQ"
    ]
}

for name, columns in numeric_columns.items():

    df = datasets[name][0]

    for column in columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        )

    if (df[columns] < 0).any().any():
        raise ValueError(
            f"Negative values found in {name}."
        )

if (
    (suppliers["reliability"] <= 0).any()
    or (suppliers["reliability"] > 1).any()
):
    raise ValueError(
        "Supplier reliability must be in (0, 1]."
    )

if suppliers["capacity_per_cycle"].sum() <= 0:
    raise ValueError("Supplier capacity must be positive.")

if not set(inventory["product_id"]).issubset(
    set(products["product_id"])
):
    raise ValueError(
        "Inventory contains unknown product IDs."
    )

print("Input validation passed!")


# ============================================================
# STEP 5: MERGE PRODUCT AND INVENTORY DATA
# ============================================================

data = inventory.merge(
    products[
        [
            "product_id",
            "unit_cost",
            "selling_price"
        ]
    ],
    on="product_id",
    how="left",
    validate="one_to_one"
)

if data[
    ["unit_cost", "selling_price"]
].isna().any().any():

    raise ValueError(
        "Product prices or costs are missing."
    )


# ============================================================
# STEP 6: CALCULATE REPLENISHMENT REQUIREMENT
# ============================================================

data["days_of_inventory"] = (
    data["current_network_stock"]
    / data["average_daily_demand"].replace(0, 1)
)

data["target_stock"] = (
    data["reorder_point"] + data["EOQ"]
)

data["inventory_position"] = (
    data["current_network_stock"]
)

data["planned_quantity"] = (
    data["target_stock"]
    - data["inventory_position"]
).clip(lower=0)

data["planned_quantity"] = (
    data["planned_quantity"].round().astype(int)
)

data["stock_shortfall"] = (
    data["reorder_point"]
    - data["current_network_stock"]
).clip(lower=0)

data["shortage_ratio"] = (
    data["stock_shortfall"]
    / data["reorder_point"].replace(0, 1)
)


# ============================================================
# STEP 7: CALCULATE PRIORITY SCORE
# ============================================================

def min_max_normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:

        return pd.Series(
            0.0,
            index=series.index
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


data["demand_score"] = min_max_normalize(
    data["average_daily_demand"]
)

data["shortage_score"] = min_max_normalize(
    data["shortage_ratio"]
)

data["priority_score"] = (
    1
    + 0.5 * data["demand_score"]
    + 0.5 * data["shortage_score"]
)


# ============================================================
# STEP 8: REMOVE ZERO-REQUIREMENT PRODUCTS
# ============================================================

data = data[
    data["planned_quantity"] > 0
].copy()

data = data.reset_index(drop=True)

if data.empty:
    raise ValueError(
        "No positive replenishment requirements."
    )


# ============================================================
# STEP 9: CREATE PRIORITY TIERS
# ============================================================

# Sort products by descending priority score.
data = data.sort_values(
    "priority_score",
    ascending=False,
    kind="stable"
).reset_index(drop=True)

number_of_products = len(data)

tier1_end = int(
    np.ceil(number_of_products * 0.20)
)

tier2_end = int(
    np.ceil(number_of_products * 0.50)
)

data["priority_rank"] = (
    np.arange(1, number_of_products + 1)
)

data["priority_tier"] = "Tier 3"

data.loc[
    data.index < tier1_end,
    "priority_tier"
] = "Tier 1"

data.loc[
    (data.index >= tier1_end)
    & (data.index < tier2_end),
    "priority_tier"
] = "Tier 2"

print("\n========== PRIORITY TIER SUMMARY ==========")

print(
    data.groupby("priority_tier").agg(
        product_count=("product_id", "count"),
        total_requirement=("planned_quantity", "sum"),
        average_priority=("priority_score", "mean")
    ).to_string()
)

print("\nTop priority products:")

print(
    data[
        [
            "product_id",
            "priority_rank",
            "priority_tier",
            "priority_score",
            "planned_quantity"
        ]
    ].head(15).to_string(index=False)
)


# ============================================================
# STEP 10: CALCULATE BASELINE CAPACITY
# ============================================================

total_capacity = int(
    suppliers["capacity_per_cycle"].sum()
)

total_requirement = int(
    data["planned_quantity"].sum()
)

baseline_capacity = min(
    int(
        np.floor(
            total_capacity * BASELINE_CAPACITY_SHARE
        )
    ),
    total_requirement
)

incremental_capacity = (
    total_capacity - baseline_capacity
)

print("\n========== CAPACITY DISTRIBUTION ==========")

print("Total capacity:", total_capacity)
print("Total requirement:", total_requirement)
print("Baseline capacity:", baseline_capacity)
print("Incremental capacity:", incremental_capacity)


# ============================================================
# STEP 11: BASELINE PROPORTIONAL ALLOCATION
# ============================================================

data["baseline_allocation"] = (
    proportional_integer_allocation(
        data["planned_quantity"],
        baseline_capacity
    )
)

print(
    "Baseline allocation:",
    data["baseline_allocation"].sum()
)


# ============================================================
# STEP 12: PRIORITY-TIER FAIR ALLOCATION
# ============================================================

data, tier_budget_df, unused_capacity = (
    allocate_by_priority_tier(
        data=data,
        incremental_capacity=incremental_capacity,
        tier_shares=TIER_CAPACITY_SHARES
    )
)

print("\n========== TIER CAPACITY ALLOCATION ==========")

print(
    tier_budget_df.to_string(index=False)
)

data["policy_allocation"] = (
    data["baseline_allocation"]
    + data["incremental_allocation"]
)

if (
    data["policy_allocation"]
    > data["planned_quantity"]
).any():
    raise ValueError(
        "Policy allocation exceeds product requirement."
    )

total_policy_allocation = int(
    data["policy_allocation"].sum()
)

print(
    "\nTotal policy allocation:",
    total_policy_allocation
)

print(
    "Unused capacity:",
    unused_capacity
)


# ============================================================
# STEP 13: SUPPLIER RISK CALCULATION
# ============================================================

suppliers["risk_multiplier"] = (
    1
    + 0.5 * (1 - suppliers["reliability"])
    + 0.02 * suppliers["lead_time_days"]
)

print("\n========== SUPPLIER RISK PROFILE ==========")

print(
    suppliers[
        [
            "supplier_id",
            "reliability",
            "lead_time_days",
            "capacity_per_cycle",
            "risk_multiplier"
        ]
    ].to_string(index=False)
)


# ============================================================
# STEP 14: PREPARE PYOMO MODEL
# ============================================================

PRODUCTS = data["product_id"].tolist()

SUPPLIER_IDS = suppliers["supplier_id"].tolist()

policy_quantity = dict(
    zip(
        data["product_id"],
        data["policy_allocation"]
    )
)

unit_cost = dict(
    zip(
        data["product_id"],
        data["unit_cost"]
    )
)

risk = dict(
    zip(
        suppliers["supplier_id"],
        suppliers["risk_multiplier"]
    )
)

capacity = dict(
    zip(
        suppliers["supplier_id"],
        suppliers["capacity_per_cycle"]
    )
)


# ============================================================
# STEP 15: BUILD SUPPLIER ASSIGNMENT MODEL
# ============================================================

model = pyo.ConcreteModel()

model.PRODUCTS = pyo.Set(
    initialize=PRODUCTS
)

model.SUPPLIERS = pyo.Set(
    initialize=SUPPLIER_IDS
)

# Integer units allocated from each supplier to each product.
model.x = pyo.Var(
    model.PRODUCTS,
    model.SUPPLIERS,
    domain=pyo.NonNegativeIntegers
)


# ------------------------------------------------------------
# Constraint 1: Exact product allocation
# ------------------------------------------------------------

def product_allocation_rule(model, i):

    return (
        sum(
            model.x[i, s]
            for s in model.SUPPLIERS
        )
        == policy_quantity[i]
    )


model.product_allocation_constraint = pyo.Constraint(
    model.PRODUCTS,
    rule=product_allocation_rule
)


# ------------------------------------------------------------
# Constraint 2: Supplier capacity
# ------------------------------------------------------------

def supplier_capacity_rule(model, s):

    return (
        sum(
            model.x[i, s]
            for i in model.PRODUCTS
        )
        <= capacity[s]
    )


model.supplier_capacity_constraint = pyo.Constraint(
    model.SUPPLIERS,
    rule=supplier_capacity_rule
)


# ============================================================
# STEP 16: MINIMIZE RISK-ADJUSTED PROCUREMENT COST
# ============================================================

def procurement_cost_rule(model):

    return sum(
        unit_cost[i]
        * risk[s]
        * model.x[i, s]

        for i in model.PRODUCTS
        for s in model.SUPPLIERS
    )


model.objective = pyo.Objective(
    rule=procurement_cost_rule,
    sense=pyo.minimize
)


# ============================================================
# STEP 17: SOLVE MODEL
# ============================================================

solver = pyo.SolverFactory(SOLVER_NAME)

if not solver.available(exception_flag=False):

    raise RuntimeError(
        f"Solver {SOLVER_NAME} is not available."
    )

print("\n========== SUPPLIER ASSIGNMENT OPTIMIZATION ==========")

results = solver.solve(
    model,
    tee=False
)

if (
    results.solver.termination_condition
    != pyo.TerminationCondition.optimal
):
    raise RuntimeError(
        "Supplier assignment model did not reach optimality."
    )

print("Optimal supplier assignment found!")


# ============================================================
# STEP 18: EXTRACT SUPPLIER ALLOCATION
# ============================================================

allocation_rows = []

product_allocated = {
    i: 0 for i in PRODUCTS
}

supplier_allocated = {
    s: 0 for s in SUPPLIER_IDS
}

for i in PRODUCTS:

    for s in SUPPLIER_IDS:

        quantity = int(
            round(
                pyo.value(model.x[i, s])
            )
        )

        if quantity > 0:

            allocation_rows.append({

                "product_id": i,

                "supplier_id": s,

                "allocated_quantity": quantity,

                "unit_cost": unit_cost[i],

                "risk_multiplier": risk[s],

                "risk_adjusted_unit_cost": (
                    unit_cost[i] * risk[s]
                )

            })

        product_allocated[i] += quantity

        supplier_allocated[s] += quantity


allocation_df = pd.DataFrame(
    allocation_rows,
    columns=[
        "product_id",
        "supplier_id",
        "allocated_quantity",
        "unit_cost",
        "risk_multiplier",
        "risk_adjusted_unit_cost"
    ]
)


# ============================================================
# STEP 19: PRODUCT-WISE SHORTAGE REPORT
# ============================================================

shortage_rows = []

for _, row in data.iterrows():

    product_id = row["product_id"]

    allocated = product_allocated[product_id]

    shortage = (
        int(row["planned_quantity"]) - allocated
    )

    fill_rate = (
        allocated / row["planned_quantity"] * 100
        if row["planned_quantity"] > 0
        else 100
    )

    shortage_rows.append({

        "product_id": product_id,

        "priority_rank": int(row["priority_rank"]),

        "priority_tier": row["priority_tier"],

        "priority_score": row["priority_score"],

        "planned_quantity": int(
            row["planned_quantity"]
        ),

        "baseline_allocation": int(
            row["baseline_allocation"]
        ),

        "incremental_allocation": int(
            row["incremental_allocation"]
        ),

        "policy_allocation": int(
            row["policy_allocation"]
        ),

        "allocated_quantity": allocated,

        "unfulfilled_quantity": shortage,

        "product_fill_rate": fill_rate

    })


shortage_df = pd.DataFrame(
    shortage_rows
)


# ============================================================
# STEP 20: SUPPLIER SUMMARY
# ============================================================

supplier_rows = []

for _, row in suppliers.iterrows():

    supplier_id = row["supplier_id"]

    allocated = supplier_allocated[supplier_id]

    supplier_capacity = int(
        row["capacity_per_cycle"]
    )

    unused = (
        supplier_capacity - allocated
    )

    utilization = (
        allocated / supplier_capacity * 100
        if supplier_capacity > 0
        else 0
    )

    supplier_rows.append({

        "supplier_id": supplier_id,

        "capacity_per_cycle": supplier_capacity,

        "allocated_quantity": allocated,

        "unused_capacity": unused,

        "utilization_percent": utilization,

        "reliability": row["reliability"],

        "lead_time_days": row["lead_time_days"],

        "risk_multiplier": row["risk_multiplier"]

    })


supplier_summary = pd.DataFrame(
    supplier_rows
)


# ============================================================
# STEP 21: TIER-WISE PERFORMANCE
# ============================================================

tier_summary = (
    shortage_df.groupby("priority_tier")
    .agg(
        product_count=("product_id", "count"),
        planned_quantity=("planned_quantity", "sum"),
        baseline_allocation=("baseline_allocation", "sum"),
        incremental_allocation=("incremental_allocation", "sum"),
        allocated_quantity=("allocated_quantity", "sum"),
        shortage=("unfulfilled_quantity", "sum"),
        average_priority=("priority_score", "mean")
    )
    .reset_index()
)

tier_summary["fill_rate_percent"] = (
    tier_summary["allocated_quantity"]
    / tier_summary["planned_quantity"]
    * 100
)


# ============================================================
# STEP 22: NETWORK PERFORMANCE METRICS
# ============================================================

total_planned = int(
    shortage_df["planned_quantity"].sum()
)

total_allocated = int(
    shortage_df["allocated_quantity"].sum()
)

total_shortage = int(
    shortage_df["unfulfilled_quantity"].sum()
)

total_baseline = int(
    shortage_df["baseline_allocation"].sum()
)

total_incremental = int(
    shortage_df["incremental_allocation"].sum()
)

network_fill_rate = (
    total_allocated / total_planned * 100
    if total_planned > 0
    else 100
)

base_procurement_cost = sum(
    unit_cost[i] * product_allocated[i]
    for i in PRODUCTS
)

risk_adjusted_cost = sum(
    unit_cost[i]
    * risk[s]
    * pyo.value(model.x[i, s])

    for i in PRODUCTS
    for s in SUPPLIER_IDS
)

risk_premium = (
    risk_adjusted_cost - base_procurement_cost
)

maximum_capacity_coverage = (
    total_capacity / total_planned * 100
    if total_planned > 0
    else 0
)


# ============================================================
# STEP 23: INDEPENDENT VALIDATION
# ============================================================

print("\n========== V5 VALIDATION ==========")

assert (
    total_allocated + total_shortage
    == total_planned
), "Demand balance failed!"

assert (
    total_baseline + total_incremental
    == total_policy_allocation
), "Policy allocation balance failed!"

assert (
    total_allocated == total_policy_allocation
), "Solver allocation does not match policy allocation!"

assert (
    shortage_df["allocated_quantity"]
    <= shortage_df["planned_quantity"]
).all(), "Product requirement constraint failed!"

assert (
    shortage_df["allocated_quantity"]
    >= shortage_df["baseline_allocation"]
).all(), "Baseline allocation constraint failed!"

assert (
    supplier_summary["allocated_quantity"]
    <= supplier_summary["capacity_per_cycle"]
).all(), "Supplier capacity constraint failed!"

assert (
    shortage_df["incremental_allocation"]
    <= (
        shortage_df["planned_quantity"]
        - shortage_df["baseline_allocation"]
    )
).all(), "Incremental allocation exceeds remaining demand!"

assert (
    supplier_summary["allocated_quantity"].sum()
    == total_allocated
), "Supplier and product totals do not match!"

print("Demand balance: PASS")
print("Baseline allocation: PASS")
print("Tier allocation: PASS")
print("Product requirement limits: PASS")
print("Supplier capacity: PASS")
print("Supplier allocation consistency: PASS")


# ============================================================
# STEP 24: SAVE REPORTS
# ============================================================

data.to_csv(
    REPORT_DIR / "replenishment_requirements_v5.csv",
    index=False
)

allocation_df.to_csv(
    REPORT_DIR / "supplier_allocation_v5.csv",
    index=False
)

shortage_df.to_csv(
    REPORT_DIR / "procurement_shortage_v5.csv",
    index=False
)

supplier_summary.to_csv(
    REPORT_DIR / "supplier_summary_v5.csv",
    index=False
)

tier_budget_df.to_csv(
    REPORT_DIR / "tier_capacity_budget_v5.csv",
    index=False
)

tier_summary.to_csv(
    REPORT_DIR / "tier_performance_v5.csv",
    index=False
)


# ============================================================
# STEP 25: SAVE NETWORK SUMMARY
# ============================================================

summary = pd.DataFrame([{

    "total_planned_quantity": total_planned,

    "total_supplier_capacity": total_capacity,

    "baseline_allocation": total_baseline,

    "incremental_allocation": total_incremental,

    "total_allocated_quantity": total_allocated,

    "total_shortage": total_shortage,

    "network_fill_rate_percent": network_fill_rate,

    "maximum_capacity_coverage_percent": (
        maximum_capacity_coverage
    ),

    "base_procurement_cost": base_procurement_cost,

    "risk_premium": risk_premium,

    "risk_adjusted_procurement_cost": (
        risk_adjusted_cost
    )

}])

summary.to_csv(
    REPORT_DIR / "optimization_summary_v5.csv",
    index=False
)


# ============================================================
# STEP 26: DISPLAY FINAL RESULTS
# ============================================================

print("\n========== V5 NETWORK RESULTS ==========")

print("Total planned requirement:", total_planned)

print("Total supplier capacity:", total_capacity)

print("Baseline allocation:", total_baseline)

print("Incremental allocation:", total_incremental)

print("Total allocated quantity:", total_allocated)

print("Total shortage:", total_shortage)

print(
    f"Network fill rate: {network_fill_rate:.2f}%"
)

print(
    f"Maximum capacity coverage: "
    f"{maximum_capacity_coverage:.2f}%"
)

print(
    f"Base procurement cost: Rs. "
    f"{base_procurement_cost:,.2f}"
)

print(
    f"Risk premium: Rs. "
    f"{risk_premium:,.2f}"
)

print(
    f"Risk-adjusted procurement cost: Rs. "
    f"{risk_adjusted_cost:,.2f}"
)


# ============================================================
# STEP 27: DISPLAY TIER-WISE PERFORMANCE
# ============================================================

print("\n========== TIER-WISE PERFORMANCE ==========")

print(
    tier_summary.to_string(index=False)
)


# ============================================================
# STEP 28: DISPLAY TOP PRIORITY PRODUCTS
# ============================================================

print("\n========== TOP PRIORITY PRODUCT ALLOCATION ==========")

print(
    shortage_df.sort_values(
        "priority_rank"
    )[
        [
            "product_id",
            "priority_rank",
            "priority_tier",
            "priority_score",
            "planned_quantity",
            "baseline_allocation",
            "incremental_allocation",
            "allocated_quantity",
            "unfulfilled_quantity",
            "product_fill_rate"
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# STEP 29: DISPLAY SUPPLIER PERFORMANCE
# ============================================================

print("\n========== SUPPLIER PERFORMANCE ==========")

print(
    supplier_summary.to_string(index=False)
)


# ============================================================
# STEP 30: COMPLETION
# ============================================================

print("\n========== SAVED V5 REPORTS ==========")

print("replenishment_requirements_v5.csv")
print("supplier_allocation_v5.csv")
print("procurement_shortage_v5.csv")
print("supplier_summary_v5.csv")
print("tier_capacity_budget_v5.csv")
print("tier_performance_v5.csv")
print("optimization_summary_v5.csv")

print("\nV5 HYBRID OPTIMIZATION COMPLETED SUCCESSFULLY!")
