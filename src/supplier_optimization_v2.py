
import pandas as pd
import pyomo.environ as pyo

from config import RAW_DATA_DIR, REPORT_DIR


# ==========================================
# STEP 1: LOAD DATA
# ==========================================

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



# ==========================================
# STEP 2: PREPARE PRODUCT DATA
# ==========================================

data = inventory.merge(
    products[
        ["product_id", "unit_cost", "selling_price"]
    ],
    on="product_id",
    how="left"
)


# ==========================================
# V3: REPLENISHMENT-BASED QUANTITY
# ==========================================

# Target stock = Reorder Point + EOQ
data["target_stock"] = (
    data["reorder_point"] + data["EOQ"]
)

# Current network inventory
data["inventory_position"] = (
    data["current_network_stock"]
)

# Actual procurement requirement
data["planned_quantity"] = (
    data["target_stock"] - data["inventory_position"]
).clip(lower=0)

# Replenishment audit report
replenishment_report = data[
    [
        "product_id",
        "reorder_point",
        "EOQ",
        "current_network_stock",
        "target_stock",
        "inventory_position",
        "planned_quantity"
    ]
].copy()

replenishment_report.to_csv(
    REPORT_DIR / "replenishment_requirements_v3.csv",
    index=False
)

print("\n========== V3 REPLENISHMENT REQUIREMENTS ==========")

print(
    replenishment_report.head(10).to_string(index=False)
)

print("\nTotal Replenishment Requirement:",
      int(data["planned_quantity"].sum()))

print("Products Requiring Replenishment:",
      int((data["planned_quantity"] > 0).sum()))


# Recalculate inventory coverage
data["days_of_inventory"] = (
    data["current_network_stock"]
    / data["average_daily_demand"].replace(0, 1)
)

# Recalculate stock shortfall
data["stock_shortfall"] = (
    data["reorder_point"] - data["current_network_stock"]
).clip(lower=0)

# Shortage ratio
data["shortage_ratio"] = (
    data["stock_shortfall"]
    / data["reorder_point"].replace(0, 1)
)

# Normalize daily demand
demand_min = data["average_daily_demand"].min()
demand_max = data["average_daily_demand"].max()

if demand_max == demand_min:
    data["demand_score"] = 0
else:
    data["demand_score"] = (
        (data["average_daily_demand"] - demand_min)
        / (demand_max - demand_min)
    )

# Normalize shortage ratio
shortage_min = data["shortage_ratio"].min()
shortage_max = data["shortage_ratio"].max()

if shortage_max == shortage_min:
    data["shortage_score"] = 0
else:
    data["shortage_score"] = (
        (data["shortage_ratio"] - shortage_min)
        / (shortage_max - shortage_min)
    )

# Product priority score
data["priority_score"] = (
    1
    + 0.5 * data["demand_score"]
    + 0.5 * data["shortage_score"]
)

# Shortage penalty assumption
data["shortage_penalty"] = (
    3 * data["selling_price"] * data["priority_score"]
)

print("\nProduct preparation completed!")
print(
    data[
        [
            "product_id",
            "days_of_inventory",
            "stock_shortfall",
            "priority_score"
        ]
    ].head(10).to_string(index=False)
)


# ==========================================
# STEP 3: PREPARE SUPPLIER DATA
# ==========================================

# Risk-adjusted procurement cost multiplier
suppliers["risk_multiplier"] = (
    1
    + 0.5 * (1 - suppliers["reliability"])
    + 0.02 * suppliers["lead_time_days"]
)

print("\n========== PRODUCT PRIORITIES ==========")

print(
    data[
        [
            "product_id",
            "average_daily_demand",
            "days_of_inventory",
            "priority_score"
        ]
    ].head(10).to_string(index=False)
)


# ==========================================
# STEP 4: CREATE DICTIONARIES
# ==========================================

product_ids = data["product_id"].tolist()

supplier_ids = suppliers["supplier_id"].tolist()

quantity = dict(
    zip(data["product_id"], data["planned_quantity"])
)

unit_cost = dict(
    zip(data["product_id"], data["unit_cost"])
)

shortage_penalty = dict(
    zip(data["product_id"], data["shortage_penalty"])
)

priority = dict(
    zip(data["product_id"], data["priority_score"])
)

selling_price = dict(
    zip(data["product_id"], data["selling_price"])
)

capacity = dict(
    zip(
        suppliers["supplier_id"],
        suppliers["capacity_per_cycle"]
    )
)

reliability = dict(
    zip(suppliers["supplier_id"], suppliers["reliability"])
)

lead_time = dict(
    zip(suppliers["supplier_id"], suppliers["lead_time_days"])
)

risk_multiplier = dict(
    zip(suppliers["supplier_id"], suppliers["risk_multiplier"])
)


# ==========================================
# STEP 5: CREATE OPTIMIZATION MODEL
# ==========================================

model = pyo.ConcreteModel()

model.PRODUCTS = pyo.Set(initialize=product_ids)

model.SUPPLIERS = pyo.Set(initialize=supplier_ids)


# Integer procurement quantities
model.x = pyo.Var(
    model.PRODUCTS,
    model.SUPPLIERS,
    domain=pyo.NonNegativeIntegers
)

# Integer shortage quantities
model.shortage = pyo.Var(
    model.PRODUCTS,
    domain=pyo.NonNegativeIntegers
)


# ==========================================
# STEP 6: OBJECTIVE FUNCTION
# ==========================================

def objective_rule(model):

    procurement_risk_cost = sum(
        unit_cost[i]
        * risk_multiplier[s]
        * model.x[i, s]
        for i in model.PRODUCTS
        for s in model.SUPPLIERS
    )

    shortage_cost = sum(
        shortage_penalty[i] * model.shortage[i]
        for i in model.PRODUCTS
    )

    return procurement_risk_cost + shortage_cost


model.objective = pyo.Objective(
    rule=objective_rule,
    sense=pyo.minimize
)


# ==========================================
# STEP 7: DEMAND CONSTRAINT
# ==========================================

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


# ==========================================
# STEP 8: SUPPLIER CAPACITY
# ==========================================

def capacity_rule(model, s):

    return sum(
        model.x[i, s]
        for i in model.PRODUCTS
    ) <= capacity[s]


model.capacity_constraint = pyo.Constraint(
    model.SUPPLIERS,
    rule=capacity_rule
)


# ==========================================
# STEP 9: SOLVE
# ==========================================

solver = pyo.SolverFactory("appsi_highs")

result = solver.solve(model)

print("\n========== OPTIMIZATION STATUS ==========")

print("Status:", result.solver.status)

print(
    "Termination:",
    result.solver.termination_condition
)


# ==========================================
# STEP 10: EXTRACT RESULTS
# ==========================================

if (
    result.solver.termination_condition
    == pyo.TerminationCondition.optimal
):

    allocation = []

    for i in model.PRODUCTS:

        for s in model.SUPPLIERS:

            allocated = pyo.value(model.x[i, s])

            if allocated > 0:

                allocation.append({

                    "product_id": i,

                    "supplier_id": s,

                    "allocated_quantity": allocated,

                    "unit_cost": unit_cost[i],

                    "priority_score": priority[i],

                    "supplier_reliability": reliability[s],

                    "lead_time_days": lead_time[s],

                    "risk_adjusted_unit_cost":
                        unit_cost[i] * risk_multiplier[s]

                })

    allocation_df = pd.DataFrame(allocation)

    shortage_results = []

    for i in model.PRODUCTS:

        allocated = sum(
            pyo.value(model.x[i, s])
            for s in model.SUPPLIERS
        )

        shortage_results.append({

            "product_id": i,

            "planned_quantity": quantity[i],

            "allocated_quantity": allocated,

            "unfulfilled_quantity": pyo.value(
                model.shortage[i]
            ),

            "priority_score": priority[i],

            "days_of_inventory": float(
                data.loc[
                    data["product_id"] == i,
                    "days_of_inventory"
                ].iloc[0]
            )

        })

    shortage_df = pd.DataFrame(shortage_results)

    # Supplier summary
    supplier_results = []

    for s in model.SUPPLIERS:

        allocated = sum(
            pyo.value(model.x[i, s])
            for i in model.PRODUCTS
        )

        supplier_results.append({

            "supplier_id": s,

            "capacity": capacity[s],

            "allocated_quantity": allocated,

            "unused_capacity": capacity[s] - allocated,

            "utilization_percent":
                allocated / capacity[s] * 100,

            "reliability": reliability[s],

            "lead_time_days": lead_time[s]

        })

    supplier_summary = pd.DataFrame(supplier_results)

    # ======================================
    # STEP 11: PERFORMANCE METRICS
    # ======================================

    total_planned = sum(quantity.values())

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
    )

    procurement_cost = sum(
        unit_cost[i] * pyo.value(model.x[i, s])
        for i in model.PRODUCTS
        for s in model.SUPPLIERS
    )

    risk_cost = sum(
        unit_cost[i]
        * (risk_multiplier[s] - 1)
        * pyo.value(model.x[i, s])
        for i in model.PRODUCTS
        for s in model.SUPPLIERS
    )

    shortage_cost = sum(
        shortage_penalty[i] * pyo.value(model.shortage[i])
        for i in model.PRODUCTS
    )

    # ======================================
    # STEP 12: SAVE OUTPUTS
    # ======================================

    allocation_df.to_csv(
        REPORT_DIR / "supplier_allocation_v2.csv",
        index=False
    )

    shortage_df.to_csv(
        REPORT_DIR / "procurement_shortage_v2.csv",
        index=False
    )

    supplier_summary.to_csv(
        REPORT_DIR / "supplier_summary_v2.csv",
        index=False
    )

    # ======================================
    # STEP 13: DISPLAY RESULTS
    # ======================================

    print("\n========== SUPPLIER ALLOCATION V2 ==========")

    print(allocation_df.head(20).to_string(index=False))

    print("\n========== SUPPLIER SUMMARY V2 ==========")

    print(supplier_summary.to_string(index=False))

    print("\n========== PRODUCT SHORTAGE V2 ==========")

    print(
        shortage_df.sort_values(
            "priority_score",
            ascending=False
        ).head(20).to_string(index=False)
    )

    print("\n========== PERFORMANCE METRICS ==========")

    print("Total planned quantity:", total_planned)

    print("Total allocated quantity:", total_allocated)

    print("Total shortage:", total_shortage)

    print("Fill rate:", round(fill_rate, 2), "%")

    print("Base procurement cost:", round(procurement_cost, 2))

    print("Supplier risk premium:", round(risk_cost, 2))

    print("Shortage penalty cost:", round(shortage_cost, 2))

    print(
        "Total objective value:",
        round(pyo.value(model.objective), 2)
    )

    print("\nVersion 2 optimization completed!")

else:

    print("Optimization did not reach an optimal solution.")
