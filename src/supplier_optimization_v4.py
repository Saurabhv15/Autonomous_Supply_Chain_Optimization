import pandas as pd
import numpy as np
import pyomo.environ as pyo

from config import RAW_DATA_DIR, REPORT_DIR


# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

# Share of total supplier capacity reserved for
# proportional baseline allocation.
BASELINE_CAPACITY_SHARE = 0.50

# Numerical tolerance for preserving the optimal
# priority-weighted fulfillment score in Stage 2.
OPTIMALITY_TOLERANCE = 1e-7

SOLVER_NAME = "appsi_highs"


# ============================================================
# STEP 2: LOAD DATASETS
# ============================================================

print("\n========== V4: HYBRID SUPPLIER OPTIMIZATION ==========")

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
# STEP 3: VALIDATE INPUT DATA
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
                f"Missing column '{column}' in {name} dataset."
            )

    if df[columns].isna().any().any():
        raise ValueError(
            f"Missing values found in {name} dataset."
        )

if products["product_id"].duplicated().any():
    raise ValueError("Duplicate product IDs found.")

if suppliers["supplier_id"].duplicated().any():
    raise ValueError("Duplicate supplier IDs found.")

if inventory["product_id"].duplicated().any():
    raise ValueError("Duplicate inventory product IDs found.")

numeric_product_columns = [
    "unit_cost",
    "selling_price"
]

numeric_supplier_columns = [
    "reliability",
    "lead_time_days",
    "capacity_per_cycle"
]

numeric_inventory_columns = [
    "average_daily_demand",
    "current_network_stock",
    "reorder_point",
    "EOQ"
]

for column in numeric_product_columns:
    products[column] = pd.to_numeric(
        products[column], errors="raise"
    )

for column in numeric_supplier_columns:
    suppliers[column] = pd.to_numeric(
        suppliers[column], errors="raise"
    )

for column in numeric_inventory_columns:
    inventory[column] = pd.to_numeric(
        inventory[column], errors="raise"
    )

for df, columns in [
    (products, numeric_product_columns),
    (suppliers, numeric_supplier_columns),
    (inventory, numeric_inventory_columns)
]:
    if (df[columns] < 0).any().any():
        raise ValueError("Negative numeric values found.")

if (
    (suppliers["reliability"] <= 0).any()
    or (suppliers["reliability"] > 1).any()
):
    raise ValueError(
        "Supplier reliability must be greater than 0 and at most 1."
    )

if not 0 < BASELINE_CAPACITY_SHARE < 1:
    raise ValueError(
        "BASELINE_CAPACITY_SHARE must be between 0 and 1."
    )

if suppliers["capacity_per_cycle"].sum() <= 0:
    raise ValueError("Total supplier capacity must be positive.")

if not set(inventory["product_id"]).issubset(
    set(products["product_id"])
):
    raise ValueError(
        "Some inventory products are missing from products.csv."
    )

print("All input validations passed!")


# ============================================================
# STEP 4: PREPARE PRODUCT DATA
# ============================================================

data = inventory.merge(
    products[
        ["product_id", "unit_cost", "selling_price"]
    ],
    on="product_id",
    how="left",
    validate="one_to_one"
)

if data[["unit_cost", "selling_price"]].isna().any().any():
    raise ValueError(
        "Product cost or selling price missing after merge."
    )

# Inventory coverage in days
data["days_of_inventory"] = (
    data["current_network_stock"]
    / data["average_daily_demand"].replace(0, 1)
)

# Target stock policy
data["target_stock"] = (
    data["reorder_point"] + data["EOQ"]
)

# Current inventory position
data["inventory_position"] = (
    data["current_network_stock"]
)

# Planned replenishment quantity
data["planned_quantity"] = (
    data["target_stock"] - data["inventory_position"]
).clip(lower=0)

data["planned_quantity"] = (
    data["planned_quantity"].round().astype(int)
)

# Stock shortfall
data["stock_shortfall"] = (
    data["reorder_point"]
    - data["current_network_stock"]
).clip(lower=0)

# Shortage ratio
data["shortage_ratio"] = (
    data["stock_shortfall"]
    / data["reorder_point"].replace(0, 1)
)


# ============================================================
# STEP 5: CALCULATE PRODUCT PRIORITY SCORE
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

# Priority score
data["priority_score"] = (
    1
    + 0.5 * data["demand_score"]
    + 0.5 * data["shortage_score"]
)

if (data["priority_score"] <= 0).any():
    raise ValueError("Priority scores must be positive.")

# Remove products having zero replenishment requirement
data = data[
    data["planned_quantity"] > 0
].copy()

data = data.reset_index(drop=True)

if data.empty:
    raise ValueError("No positive replenishment requirement found.")

print("\n========== PRODUCT PRIORITIES ==========")

print(
    data[
        [
            "product_id",
            "planned_quantity",
            "days_of_inventory",
            "priority_score"
        ]
    ]
    .sort_values(
        "priority_score",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


# ============================================================
# STEP 6: CALCULATE SUPPLIER RISK
# ============================================================

suppliers["risk_multiplier"] = (
    1
    + 0.5 * (1 - suppliers["reliability"])
    + 0.02 * suppliers["lead_time_days"]
)

suppliers["risk_adjusted_unit_cost_factor"] = (
    suppliers["risk_multiplier"]
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
# STEP 7: CALCULATE HYBRID BASELINE ALLOCATION
# ============================================================

total_capacity = int(
    suppliers["capacity_per_cycle"].sum()
)

total_requirement = int(
    data["planned_quantity"].sum()
)

baseline_capacity = int(
    np.floor(
        total_capacity * BASELINE_CAPACITY_SHARE
    )
)

priority_capacity = (
    total_capacity - baseline_capacity
)

if baseline_capacity > total_requirement:
    baseline_capacity = total_requirement

    priority_capacity = (
        total_capacity - baseline_capacity
    )

print("\n========== HYBRID CAPACITY DISTRIBUTION ==========")

print("Total supplier capacity:", total_capacity)
print("Total planned requirement:", total_requirement)

print("Baseline capacity:", baseline_capacity)
print("Priority capacity:", priority_capacity)


# ------------------------------------------------------------
# Largest remainder method
# ------------------------------------------------------------

def proportional_integer_allocation(
    quantities,
    allocation_budget
):

    total = int(quantities.sum())

    if allocation_budget < 0 or allocation_budget > total:
        raise ValueError("Invalid baseline allocation budget.")

    if total == 0:
        return pd.Series(
            0,
            index=quantities.index,
            dtype=int
        )

    exact_allocations = (
        quantities / total
    ) * allocation_budget

    allocations = np.floor(
        exact_allocations
    ).astype(int)

    remaining = (
        allocation_budget - allocations.sum()
    )

    fractional_parts = (
        exact_allocations - allocations
    )

    order = fractional_parts.sort_values(
        ascending=False,
        kind="stable"
    ).index

    for idx in order:

        if remaining <= 0:
            break

        if allocations.loc[idx] < quantities.loc[idx]:

            allocations.loc[idx] += 1
            remaining -= 1

    if remaining != 0:
        raise ValueError(
            "Could not distribute baseline capacity exactly."
        )

    return allocations


data["baseline_allocation"] = (
    proportional_integer_allocation(
        data["planned_quantity"],
        baseline_capacity
    )
)

data["baseline_allocation"] = (
    data["baseline_allocation"].astype(int)
)

if (
    data["baseline_allocation"]
    > data["planned_quantity"]
).any():
    raise ValueError(
        "Baseline allocation exceeds product requirement."
    )

print(
    "Baseline allocation distributed:",
    data["baseline_allocation"].sum()
)


# ============================================================
# STEP 8: PREPARE PYOMO MODEL
# ============================================================

PRODUCTS = data["product_id"].tolist()

SUPPLIER_IDS = suppliers["supplier_id"].tolist()

planned = dict(
    zip(
        data["product_id"],
        data["planned_quantity"]
    )
)

baseline = dict(
    zip(
        data["product_id"],
        data["baseline_allocation"]
    )
)

priority = dict(
    zip(
        data["product_id"],
        data["priority_score"]
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

# Each unit's contribution to weighted fulfillment score
priority_weight = {
    i: priority[i] / planned[i]
    for i in PRODUCTS
}


# ============================================================
# STEP 9: BUILD OPTIMIZATION MODEL
# ============================================================

model = pyo.ConcreteModel()

model.PRODUCTS = pyo.Set(
    initialize=PRODUCTS
)

model.SUPPLIERS = pyo.Set(
    initialize=SUPPLIER_IDS
)

# Integer allocation variable
model.x = pyo.Var(
    model.PRODUCTS,
    model.SUPPLIERS,
    domain=pyo.NonNegativeIntegers
)


# ------------------------------------------------------------
# Constraint 1: Product demand limit
# ------------------------------------------------------------

def product_limit_rule(model, i):

    return (
        sum(
            model.x[i, s]
            for s in model.SUPPLIERS
        )
        <= planned[i]
    )


model.product_limit = pyo.Constraint(
    model.PRODUCTS,
    rule=product_limit_rule
)


# ------------------------------------------------------------
# Constraint 2: Baseline allocation
# ------------------------------------------------------------

def baseline_rule(model, i):

    return (
        sum(
            model.x[i, s]
            for s in model.SUPPLIERS
        )
        >= baseline[i]
    )


model.baseline_constraint = pyo.Constraint(
    model.PRODUCTS,
    rule=baseline_rule
)


# ------------------------------------------------------------
# Constraint 3: Supplier capacity
# ------------------------------------------------------------

def supplier_capacity_rule(model, s):

    return (
        sum(
            model.x[i, s]
            for i in model.PRODUCTS
        )
        <= capacity[s]
    )


model.supplier_capacity = pyo.Constraint(
    model.SUPPLIERS,
    rule=supplier_capacity_rule
)


# ============================================================
# STEP 10: STAGE 1 - PRIORITY-WEIGHTED FULFILLMENT
# ============================================================

def weighted_fulfillment_expression(model):

    return sum(
        priority_weight[i]
        * sum(
            model.x[i, s]
            for s in model.SUPPLIERS
        )
        for i in model.PRODUCTS
    )


model.priority_objective = pyo.Objective(
    rule=weighted_fulfillment_expression,
    sense=pyo.maximize
)


# ============================================================
# STEP 11: SOLVE STAGE 1
# ============================================================

solver = pyo.SolverFactory(SOLVER_NAME)

if not solver.available(exception_flag=False):
    raise RuntimeError(
        f"Solver {SOLVER_NAME} is not available."
    )

print("\n========== STAGE 1 ==========")
print("Maximizing priority-weighted fulfillment...")

stage1_results = solver.solve(
    model,
    tee=False
)

if (
    stage1_results.solver.termination_condition
    != pyo.TerminationCondition.optimal
):
    raise RuntimeError(
        "Stage 1 did not reach an optimal solution."
    )

best_priority_score = pyo.value(
    model.priority_objective
)

print("Stage 1 optimal score:", best_priority_score)


# ============================================================
# STEP 12: STAGE 2 - MINIMIZE PROCUREMENT RISK COST
# ============================================================

# Preserve the optimal Stage 1 priority score.
model.priority_objective.deactivate()

model.priority_score_floor = pyo.Constraint(
    expr=(
        weighted_fulfillment_expression(model)
        >= best_priority_score - OPTIMALITY_TOLERANCE
    )
)


def procurement_cost_expression(model):

    return sum(
        unit_cost[i]
        * risk[s]
        * model.x[i, s]
        for i in model.PRODUCTS
        for s in model.SUPPLIERS
    )


model.cost_objective = pyo.Objective(
    rule=procurement_cost_expression,
    sense=pyo.minimize
)

print("\n========== STAGE 2 ==========")
print("Minimizing risk-adjusted procurement cost...")

stage2_results = solver.solve(
    model,
    tee=False
)

if (
    stage2_results.solver.termination_condition
    != pyo.TerminationCondition.optimal
):
    raise RuntimeError(
        "Stage 2 did not reach an optimal solution."
    )

print("Stage 2 optimal solution found!")


# ============================================================
# STEP 13: EXTRACT OPTIMIZATION RESULTS
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

                "priority_score": priority[i],

                "supplier_reliability": float(
                    suppliers.loc[
                        suppliers["supplier_id"] == s,
                        "reliability"
                    ].iloc[0]
                ),

                "lead_time_days": int(
                    suppliers.loc[
                        suppliers["supplier_id"] == s,
                        "lead_time_days"
                    ].iloc[0]
                ),

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
        "priority_score",
        "supplier_reliability",
        "lead_time_days",
        "risk_multiplier",
        "risk_adjusted_unit_cost"
    ]
)


# ============================================================
# STEP 14: PRODUCT-WISE SHORTAGE REPORT
# ============================================================

shortage_rows = []

for _, row in data.iterrows():

    product_id = row["product_id"]

    allocated = product_allocated[product_id]

    unfulfilled = (
        int(row["planned_quantity"]) - allocated
    )

    fill_rate = (
        allocated / row["planned_quantity"] * 100
        if row["planned_quantity"] > 0
        else 100
    )

    priority_extra = (
        allocated - int(row["baseline_allocation"])
    )

    shortage_rows.append({

        "product_id": product_id,

        "planned_quantity": int(
            row["planned_quantity"]
        ),

        "baseline_allocation": int(
            row["baseline_allocation"]
        ),

        "priority_incremental_allocation": (
            priority_extra
        ),

        "allocated_quantity": allocated,

        "unfulfilled_quantity": unfulfilled,

        "product_fill_rate": fill_rate,

        "priority_score": row["priority_score"],

        "days_of_inventory": row["days_of_inventory"],

        "shortage_penalty_per_unit": (
            row["selling_price"] * row["priority_score"]
        )

    })


shortage_df = pd.DataFrame(shortage_rows)


# ============================================================
# STEP 15: SUPPLIER SUMMARY
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
# STEP 16: CALCULATE PERFORMANCE METRICS
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

network_fill_rate = (
    total_allocated / total_planned * 100
    if total_planned > 0
    else 100
)

baseline_total = int(
    shortage_df["baseline_allocation"].sum()
)

priority_incremental_total = int(
    shortage_df["priority_incremental_allocation"].sum()
)

base_procurement_cost = sum(
    unit_cost[i] * product_allocated[i]
    for i in PRODUCTS
)

risk_adjusted_cost = sum(
    unit_cost[i] * risk[s] * pyo.value(model.x[i, s])
    for i in PRODUCTS
    for s in SUPPLIER_IDS
)

risk_premium = (
    risk_adjusted_cost - base_procurement_cost
)

final_priority_score = sum(
    priority_weight[i] * product_allocated[i]
    for i in PRODUCTS
)

supplier_capacity_used = sum(
    supplier_allocated[s]
    for s in SUPPLIER_IDS
)

maximum_capacity_coverage = (
    total_capacity / total_planned * 100
    if total_planned > 0
    else 0
)


# ============================================================
# STEP 17: MATHEMATICAL VALIDATION
# ============================================================

print("\n========== V4 CONSTRAINT VALIDATION ==========")

# Product demand balance
assert (
    total_allocated + total_shortage
    == total_planned
), "Product demand balance failed!"

# Baseline allocation
assert (
    (shortage_df["allocated_quantity"]
     >= shortage_df["baseline_allocation"]).all()
), "Baseline allocation constraint failed!"

# Product quantity limit
assert (
    (shortage_df["allocated_quantity"]
     <= shortage_df["planned_quantity"]).all()
), "Product allocation limit failed!"

# Supplier capacity
assert (
    (supplier_summary["allocated_quantity"]
     <= supplier_summary["capacity_per_cycle"]).all()
), "Supplier capacity constraint failed!"

# Baseline allocation total
assert baseline_total == baseline_capacity, (
    "Baseline allocation total is incorrect!"
)

# Total supplier allocation
assert total_allocated == supplier_capacity_used, (
    "Supplier allocation total does not match network allocation!"
)

# Priority score preservation
assert (
    final_priority_score
    >= best_priority_score - OPTIMALITY_TOLERANCE - 1e-6
), "Priority score preservation failed!"

print("Product demand balance: PASS")
print("Baseline allocation: PASS")
print("Product quantity limits: PASS")
print("Supplier capacity: PASS")
print("Baseline allocation total: PASS")
print("Network allocation consistency: PASS")
print("Priority score preservation: PASS")


# ============================================================
# STEP 18: SAVE V4 REPORTS
# ============================================================

# Full replenishment requirement
replenishment_report = data.copy()

replenishment_report.to_csv(
    REPORT_DIR / "replenishment_requirements_v4.csv",
    index=False
)

allocation_df.to_csv(
    REPORT_DIR / "supplier_allocation_v4.csv",
    index=False
)

shortage_df.to_csv(
    REPORT_DIR / "procurement_shortage_v4.csv",
    index=False
)

supplier_summary.to_csv(
    REPORT_DIR / "supplier_summary_v4.csv",
    index=False
)


# ============================================================
# STEP 19: SAVE OPTIMIZATION SUMMARY
# ============================================================

summary = pd.DataFrame([{

    "total_planned_quantity": total_planned,

    "total_supplier_capacity": total_capacity,

    "baseline_capacity": baseline_capacity,

    "priority_capacity": priority_capacity,

    "baseline_allocation": baseline_total,

    "priority_incremental_allocation": (
        priority_incremental_total
    ),

    "total_allocated_quantity": total_allocated,

    "total_shortage": total_shortage,

    "network_fill_rate_percent": network_fill_rate,

    "maximum_capacity_coverage_percent": (
        maximum_capacity_coverage
    ),

    "stage1_optimal_priority_score": (
        best_priority_score
    ),

    "final_priority_score": final_priority_score,

    "base_procurement_cost": base_procurement_cost,

    "risk_premium": risk_premium,

    "risk_adjusted_procurement_cost": (
        risk_adjusted_cost
    )

}])


summary.to_csv(
    REPORT_DIR / "optimization_summary_v4.csv",
    index=False
)


# ============================================================
# STEP 20: DISPLAY FINAL RESULTS
# ============================================================

print("\n========== V4 HYBRID OPTIMIZATION RESULTS ==========")

print("Total planned requirement:", total_planned)

print("Total supplier capacity:", total_capacity)

print("Baseline allocation:", baseline_total)

print(
    "Priority incremental allocation:",
    priority_incremental_total
)

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
    f"Base procurement cost: "
    f"Rs. {base_procurement_cost:,.2f}"
)

print(
    f"Supplier risk premium: "
    f"Rs. {risk_premium:,.2f}"
)

print(
    f"Risk-adjusted procurement cost: "
    f"Rs. {risk_adjusted_cost:,.2f}"
)


# ============================================================
# STEP 21: DISPLAY TOP PRIORITY ALLOCATIONS
# ============================================================

print("\n========== TOP PRIORITY PRODUCT ALLOCATIONS ==========")

print(
    shortage_df.sort_values(
        "priority_score",
        ascending=False
    )[
        [
            "product_id",
            "priority_score",
            "planned_quantity",
            "baseline_allocation",
            "priority_incremental_allocation",
            "allocated_quantity",
            "unfulfilled_quantity",
            "product_fill_rate"
        ]
    ].head(15).to_string(index=False)
)


# ============================================================
# STEP 22: DISPLAY SUPPLIER PERFORMANCE
# ============================================================

print("\n========== SUPPLIER PERFORMANCE ==========")

print(
    supplier_summary.to_string(index=False)
)


# ============================================================
# STEP 23: REPORT FILES
# ============================================================

print("\n========== SAVED V4 REPORTS ==========")

print("replenishment_requirements_v4.csv")
print("supplier_allocation_v4.csv")
print("procurement_shortage_v4.csv")
print("supplier_summary_v4.csv")
print("optimization_summary_v4.csv")

print("\nV4 HYBRID SUPPLIER OPTIMIZATION COMPLETED!")
