
import pandas as pd
import pyomo.environ as pyo

from config import RAW_DATA_DIR, REPORT_DIR


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

products = pd.read_csv(RAW_DATA_DIR / "products.csv")
suppliers = pd.read_csv(RAW_DATA_DIR / "suppliers.csv")
inventory = pd.read_csv(REPORT_DIR / "inventory_policy_results.csv")

print("Datasets loaded successfully!")


# ============================================================
# STEP 2: VALIDATE INPUT DATA
# ============================================================

# Required columns in products.csv
required_product_columns = [
    "product_id",
    "unit_cost",
    "selling_price"
]

# Required columns in suppliers.csv
required_supplier_columns = [
    "supplier_id",
    "reliability",
    "lead_time_days",
    "capacity_per_cycle"
]

# Required columns in inventory_policy_results.csv
required_inventory_columns = [
    "product_id",
    "average_daily_demand",
    "current_network_stock",
    "reorder_point",
    "EOQ"
]


# ------------------------------------------------------------
# 1. Validate product columns
# ------------------------------------------------------------

for column in required_product_columns:
    if column not in products.columns:
        raise ValueError(
            f"Missing product column: {column}"
        )


# ------------------------------------------------------------
# 2. Validate supplier columns
# ------------------------------------------------------------

for column in required_supplier_columns:
    if column not in suppliers.columns:
        raise ValueError(
            f"Missing supplier column: {column}"
        )


# ------------------------------------------------------------
# 3. Validate inventory columns
# ------------------------------------------------------------

for column in required_inventory_columns:
    if column not in inventory.columns:
        raise ValueError(
            f"Missing inventory column: {column}"
        )


# ------------------------------------------------------------
# 4. Check duplicate IDs
# ------------------------------------------------------------

if products["product_id"].duplicated().any():
    raise ValueError("Duplicate product IDs found.")

if suppliers["supplier_id"].duplicated().any():
    raise ValueError("Duplicate supplier IDs found.")

if inventory["product_id"].duplicated().any():
    raise ValueError(
        "Duplicate product IDs in inventory policy."
    )


# ------------------------------------------------------------
# 5. Validate supplier capacity
# ------------------------------------------------------------

if (suppliers["capacity_per_cycle"] < 0).any():
    raise ValueError(
        "Supplier capacity cannot be negative."
    )


# ------------------------------------------------------------
# 6. Validate supplier reliability
# ------------------------------------------------------------

if (
    (suppliers["reliability"] <= 0).any()
    or (suppliers["reliability"] > 1).any()
):
    raise ValueError(
        "Supplier reliability must be between 0 and 1."
    )


# ------------------------------------------------------------
# 7. Validate inventory values
# ------------------------------------------------------------

numeric_columns = [
    "average_daily_demand",
    "current_network_stock",
    "reorder_point",
    "EOQ"
]

for column in numeric_columns:
    if inventory[column].isna().any():
        raise ValueError(
            f"Missing values found in inventory column: {column}"
        )

    if (inventory[column] < 0).any():
        raise ValueError(
            f"Negative values found in inventory column: {column}"
        )


# ------------------------------------------------------------
# 8. Validation completed
# ------------------------------------------------------------

print("\n========== INPUT DATA VALIDATION ==========")

print("Product records:", len(products))
print("Supplier records:", len(suppliers))
print("Inventory policy records:", len(inventory))

print("Product columns: Verified")
print("Supplier columns: Verified")
print("Inventory columns: Verified")
print("Duplicate IDs: None")
print("Supplier capacity: Valid")
print("Supplier reliability: Valid")

print("\nAll input data validations passed successfully!")

# ============================================================
# STEP 3: PREPARE PRODUCT DATA
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
    raise ValueError("Product cost information is missing.")

print("\nProduct data prepared successfully!")


# ============================================================
# STEP 4: CALCULATE REPLENISHMENT REQUIREMENT
# ============================================================
# Calculate current inventory coverage in days
data["days_of_inventory"] = (
    data["current_network_stock"]
    / data["average_daily_demand"].replace(0, 1)
)
# Target stock policy:
# Target stock = Reorder Point + Economic Order Quantity

data["target_stock"] = (
    data["reorder_point"] + data["EOQ"]
)

# No outstanding orders or backorders are available
# in the current synthetic dataset.
data["inventory_position"] = data["current_network_stock"]

data["planned_quantity"] = (
    data["target_stock"] - data["inventory_position"]
).clip(lower=0)

data["planned_quantity"] = (
    data["planned_quantity"].round().astype(int)
)

print("\n========== V3 REPLENISHMENT SUMMARY ==========")

print("Total target stock:", data["target_stock"].sum())

print(
    "Current inventory:",
    data["inventory_position"].sum()
)

print(
    "Total replenishment requirement:",
    data["planned_quantity"].sum()
)


# Save replenishment requirements separately

replenishment_columns = [
    "product_id",
    "EOQ",
    "reorder_point",
    "current_network_stock",
    "target_stock",
    "inventory_position",
    "planned_quantity",
    "average_daily_demand",
    "days_of_inventory"
]

data[replenishment_columns].to_csv(
    REPORT_DIR / "replenishment_requirements_v3.csv",
    index=False
)


# ============================================================
# STEP 5: CALCULATE PRODUCT PRIORITY
# ============================================================

# Stock shortfall relative to reorder point

data["stock_shortfall"] = (
    data["reorder_point"] - data["current_network_stock"]
).clip(lower=0)

data["shortage_ratio"] = (
    data["stock_shortfall"]
    / data["reorder_point"].replace(0, 1)
)


# Min-max normalization

def normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            0.0,
            index=series.index
        )

    return (series - minimum) / (maximum - minimum)


data["demand_score"] = normalize(
    data["average_daily_demand"]
)

data["shortage_score"] = normalize(
    data["shortage_ratio"]
)


# Higher score means higher replenishment priority

data["priority_score"] = (
    1
    + 0.5 * data["demand_score"]
    + 0.5 * data["shortage_score"]
)


# Assumed shortage penalty per unfulfilled unit
# This is a synthetic business assumption, not observed data.

data["shortage_penalty"] = (
    3 * data["selling_price"] * data["priority_score"]
)


print("\n========== PRODUCT PRIORITIES ==========")

print(
    data[
        [
            "product_id",
            "average_daily_demand",
            "days_of_inventory",
            "stock_shortfall",
            "priority_score",
            "planned_quantity"
        ]
    ]
    .sort_values("priority_score", ascending=False)
    .head(10)
    .to_string(index=False)
)


# ============================================================
# STEP 6: PREPARE SUPPLIER RISK
# ============================================================

# Risk-adjusted cost multiplier.
# The coefficients are assumed proxy weights.

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
# STEP 7: CREATE OPTIMIZATION DICTIONARIES
# ============================================================

product_ids = data["product_id"].tolist()

supplier_ids = suppliers["supplier_id"].tolist()

quantity = dict(
    zip(
        data["product_id"],
        data["planned_quantity"]
    )
)

unit_cost = dict(
    zip(
        data["product_id"],
        data["unit_cost"]
    )
)

shortage_penalty = dict(
    zip(
        data["product_id"],
        data["shortage_penalty"]
    )
)

priority = dict(
    zip(
        data["product_id"],
        data["priority_score"]
    )
)

capacity = dict(
    zip(
        suppliers["supplier_id"],
        suppliers["capacity_per_cycle"]
    )
)

reliability = dict(
    zip(
        suppliers["supplier_id"],
        suppliers["reliability"]
    )
)

lead_time = dict(
    zip(
        suppliers["supplier_id"],
        suppliers["lead_time_days"]
    )
)

risk_multiplier = dict(
    zip(
        suppliers["supplier_id"],
        suppliers["risk_multiplier"]
    )
)


# ============================================================
# STEP 8: CHECK TOTAL PROCUREMENT CAPACITY
# ============================================================

total_planned = sum(quantity.values())

total_capacity = sum(capacity.values())

print("\n========== CAPACITY CHECK ==========")

print("Total planned requirement:", total_planned)
print("Total supplier capacity:", total_capacity)

print(
    "Capacity gap:",
    max(0, total_planned - total_capacity)
)

if total_planned > 0:

    print(
        "Maximum possible fill rate:",
        round(
            min(total_capacity / total_planned, 1) * 100,
            2
        ),
        "%"
    )


# ============================================================
# STEP 9: CREATE PYOMO MODEL
# ============================================================

model = pyo.ConcreteModel()

model.PRODUCTS = pyo.Set(
    initialize=product_ids
)

model.SUPPLIERS = pyo.Set(
    initialize=supplier_ids
)


# Quantity allocated from supplier s to product i

model.x = pyo.Var(
    model.PRODUCTS,
    model.SUPPLIERS,
    domain=pyo.NonNegativeIntegers
)


# Unfulfilled replenishment quantity

model.shortage = pyo.Var(
    model.PRODUCTS,
    domain=pyo.NonNegativeIntegers
)


# ============================================================
# STEP 10: OBJECTIVE FUNCTION
# ============================================================

def objective_rule(model):

    # Procurement cost adjusted for supplier risk

    procurement_risk_cost = sum(
        unit_cost[i]
        * risk_multiplier[s]
        * model.x[i, s]

        for i in model.PRODUCTS
        for s in model.SUPPLIERS
    )

    # Penalty for unfulfilled replenishment

    shortage_cost = sum(
        shortage_penalty[i] * model.shortage[i]

        for i in model.PRODUCTS
    )

    return procurement_risk_cost + shortage_cost


model.objective = pyo.Objective(
    rule=objective_rule,
    sense=pyo.minimize
)


# ============================================================
# STEP 11: PRODUCT DEMAND CONSTRAINT
# ============================================================

def demand_rule(model, i):

    return (
        sum(
            model.x[i, s]
            for s in model.SUPPLIERS
        )
        + model.shortage[i]
        == quantity[i]
    )


model.demand_constraint = pyo.Constraint(
    model.PRODUCTS,
    rule=demand_rule
)


# ============================================================
# STEP 12: SUPPLIER CAPACITY CONSTRAINT
# ============================================================

def capacity_rule(model, s):

    return (
        sum(
            model.x[i, s]
            for i in model.PRODUCTS
        )
        <= capacity[s]
    )


model.capacity_constraint = pyo.Constraint(
    model.SUPPLIERS,
    rule=capacity_rule
)


# ============================================================
# STEP 13: SOLVE OPTIMIZATION MODEL
# ============================================================

solver = pyo.SolverFactory("appsi_highs")

if not solver.available(exception_flag=False):
    raise RuntimeError(
        "HiGHS solver is unavailable. Check highspy installation."
    )

print("\nSolving V3 optimization model...")

result = solver.solve(model)

print("\n========== OPTIMIZATION STATUS ==========")

print("Status:", result.solver.status)

print(
    "Termination:",
    result.solver.termination_condition
)


# ============================================================
# STEP 14: EXTRACT OPTIMAL RESULTS
# ============================================================

if (
    result.solver.termination_condition
    != pyo.TerminationCondition.optimal
):
    raise RuntimeError(
        "Optimization did not reach an optimal solution."
    )


allocation = []

for i in model.PRODUCTS:

    for s in model.SUPPLIERS:

        allocated = pyo.value(model.x[i, s])

        if allocated > 0:

            allocation.append({

                "product_id": i,

                "supplier_id": s,

                "allocated_quantity": int(round(allocated)),

                "unit_cost": unit_cost[i],

                "priority_score": priority[i],

                "supplier_reliability": reliability[s],

                "lead_time_days": lead_time[s],

                "risk_adjusted_unit_cost": (
                    unit_cost[i] * risk_multiplier[s]
                )

            })


allocation_columns = [
    "product_id",
    "supplier_id",
    "allocated_quantity",
    "unit_cost",
    "priority_score",
    "supplier_reliability",
    "lead_time_days",
    "risk_adjusted_unit_cost"
]

allocation_df = pd.DataFrame(
    allocation,
    columns=allocation_columns
)


# ============================================================
# STEP 15: PRODUCT SHORTAGE REPORT
# ============================================================

shortage_results = []

for i in model.PRODUCTS:

    allocated = sum(
        pyo.value(model.x[i, s])
        for s in model.SUPPLIERS
    )

    unfulfilled = pyo.value(
        model.shortage[i]
    )

    planned = quantity[i]

    product_fill_rate = (
        allocated / planned * 100
        if planned > 0
        else 100
    )

    shortage_results.append({

        "product_id": i,

        "planned_quantity": planned,

        "allocated_quantity": int(round(allocated)),

        "unfulfilled_quantity": int(round(unfulfilled)),

        "product_fill_rate": product_fill_rate,

        "priority_score": priority[i],

        "shortage_penalty_per_unit": shortage_penalty[i]

    })


shortage_df = pd.DataFrame(shortage_results)


# ============================================================
# STEP 16: SUPPLIER SUMMARY
# ============================================================

supplier_results = []

for s in model.SUPPLIERS:

    allocated = sum(
        pyo.value(model.x[i, s])
        for i in model.PRODUCTS
    )

    unused = capacity[s] - allocated

    utilization = (
        allocated / capacity[s] * 100
        if capacity[s] > 0
        else 0
    )

    supplier_results.append({

        "supplier_id": s,

        "capacity_per_cycle": capacity[s],

        "allocated_quantity": int(round(allocated)),

        "unused_capacity": int(round(unused)),

        "utilization_percent": utilization,

        "reliability": reliability[s],

        "lead_time_days": lead_time[s],

        "risk_multiplier": risk_multiplier[s]

    })


supplier_summary = pd.DataFrame(supplier_results)


# ============================================================
# STEP 17: PERFORMANCE METRICS
# ============================================================

total_allocated = sum(
    pyo.value(model.x[i, s])
    for i in model.PRODUCTS
    for s in model.SUPPLIERS
)

total_shortage = sum(
    pyo.value(model.shortage[i])
    for i in model.PRODUCTS
)

fill_rate = (
    total_allocated / total_planned * 100
    if total_planned > 0
    else 100
)


# Base procurement cost

procurement_cost = sum(
    unit_cost[i] * pyo.value(model.x[i, s])

    for i in model.PRODUCTS
    for s in model.SUPPLIERS
)


# Additional cost due to supplier risk multiplier

risk_cost = sum(
    unit_cost[i]
    * (risk_multiplier[s] - 1)
    * pyo.value(model.x[i, s])

    for i in model.PRODUCTS
    for s in model.SUPPLIERS
)


# Shortage penalty

shortage_cost = sum(
    shortage_penalty[i] * pyo.value(model.shortage[i])

    for i in model.PRODUCTS
)


objective_value = pyo.value(model.objective)


# ============================================================
# STEP 18: VALIDATE OPTIMIZATION RESULTS
# ============================================================

assert round(total_allocated + total_shortage) == total_planned, (
    "Product demand balance validation failed."
)

assert (
    supplier_summary["allocated_quantity"]
    <= supplier_summary["capacity_per_cycle"]
).all(), "Supplier capacity constraint violated."

print("\nAll optimization constraints verified successfully!")


# ============================================================
# STEP 19: SAVE V3 OUTPUT FILES
# ============================================================

allocation_df.to_csv(
    REPORT_DIR / "supplier_allocation_v3.csv",
    index=False
)

shortage_df.to_csv(
    REPORT_DIR / "procurement_shortage_v3.csv",
    index=False
)

supplier_summary.to_csv(
    REPORT_DIR / "supplier_summary_v3.csv",
    index=False
)


# ============================================================
# STEP 20: DISPLAY RESULTS
# ============================================================

print("\n========== SUPPLIER ALLOCATION V3 ==========")

if allocation_df.empty:
    print("No supplier allocation was made.")
else:
    print(
        allocation_df.head(20).to_string(index=False)
    )


print("\n========== SUPPLIER SUMMARY V3 ==========")

print(
    supplier_summary.to_string(index=False)
)


print("\n========== PRODUCT SHORTAGE V3 ==========")

print(
    shortage_df.sort_values(
        "priority_score",
        ascending=False
    ).head(20).to_string(index=False)
)


print("\n========== V3 PERFORMANCE METRICS ==========")

print("Total planned quantity:", total_planned)

print("Total allocated quantity:", int(round(total_allocated)))

print("Total shortage:", int(round(total_shortage)))

print("Fill rate:", round(fill_rate, 2), "%")

print("Base procurement cost:", round(procurement_cost, 2))

print("Supplier risk premium:", round(risk_cost, 2))

print("Shortage penalty cost:", round(shortage_cost, 2))

print("Total objective value:", round(objective_value, 2))


print("\n========== SAVED REPORTS ==========")

print("replenishment_requirements_v3.csv")
print("supplier_allocation_v3.csv")
print("procurement_shortage_v3.csv")
print("supplier_summary_v3.csv")

print("\nVersion 3 optimization completed successfully!")
