
import pandas as pd
import pyomo.environ as pyo

from config import (
    RAW_DATA_DIR,
    REPORT_DIR
)

# ==========================================
# STEP 1: LOAD DATA
# ==========================================

products = pd.read_csv(
    RAW_DATA_DIR / "products.csv"
)

suppliers = pd.read_csv(
    RAW_DATA_DIR / "suppliers.csv"
)

inventory_policy = pd.read_csv(
    REPORT_DIR / "inventory_policy_results.csv"
)

print("All datasets loaded successfully!")


# ==========================================
# STEP 2: PREPARE PROCUREMENT REQUIREMENTS
# ==========================================

# For this first model, use EOQ as the
# planned procurement quantity.
# This is an initial planning assumption.

inventory_policy["planned_quantity"] = (
    inventory_policy["EOQ"]
)

# Merge product costs
procurement = inventory_policy.merge(
    products[["product_id", "unit_cost"]],
    on="product_id",
    how="left"
)

# Shortage penalty assumption:
# 3 times the unit procurement cost
procurement["shortage_penalty"] = (
    3 * procurement["unit_cost"]
)

product_ids = procurement["product_id"].tolist()
supplier_ids = suppliers["supplier_id"].tolist()

# Dictionaries for optimization
quantity = dict(
    zip(
        procurement["product_id"],
        procurement["planned_quantity"]
    )
)

unit_cost = dict(
    zip(
        procurement["product_id"],
        procurement["unit_cost"]
    )
)

shortage_penalty = dict(
    zip(
        procurement["product_id"],
        procurement["shortage_penalty"]
    )
)

capacity = dict(
    zip(
        suppliers["supplier_id"],
        suppliers["capacity"]
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


# ==========================================
# STEP 3: CREATE PYOMO MODEL
# ==========================================

model = pyo.ConcreteModel()

model.PRODUCTS = pyo.Set(
    initialize=product_ids
)

model.SUPPLIERS = pyo.Set(
    initialize=supplier_ids
)


# ==========================================
# STEP 4: DECISION VARIABLES
# ==========================================

# Quantity purchased from each supplier
model.x = pyo.Var(
    model.PRODUCTS,
    model.SUPPLIERS,
    domain=pyo.NonNegativeReals
)

# Unfulfilled product requirement
model.shortage = pyo.Var(
    model.PRODUCTS,
    domain=pyo.NonNegativeReals
)


# ==========================================
# STEP 5: OBJECTIVE FUNCTION
# ==========================================

def objective_rule(model):

    procurement_cost = sum(
        unit_cost[i] * model.x[i, s]
        for i in model.PRODUCTS
        for s in model.SUPPLIERS
    )

    shortage_cost = sum(
        shortage_penalty[i] * model.shortage[i]
        for i in model.PRODUCTS
    )

    return procurement_cost + shortage_cost


model.objective = pyo.Objective(
    rule=objective_rule,
    sense=pyo.minimize
)


# ==========================================
# STEP 6: PRODUCT REQUIREMENT CONSTRAINT
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
# STEP 7: SUPPLIER CAPACITY CONSTRAINT
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
# STEP 8: SOLVE MODEL
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
# STEP 9: EXTRACT RESULTS
# ==========================================

if (
    result.solver.termination_condition
    == pyo.TerminationCondition.optimal
):

    allocation = []

    for i in model.PRODUCTS:

        for s in model.SUPPLIERS:

            allocated = pyo.value(
                model.x[i, s]
            )

            if allocated > 0.000001:

                allocation.append({
                    "product_id": i,
                    "supplier_id": s,
                    "allocated_quantity": allocated,
                    "unit_cost": unit_cost[i],
                    "reliability": reliability[s],
                    "lead_time_days": lead_time[s]
                })

    allocation_df = pd.DataFrame(allocation)

    shortage_results = []

    for i in model.PRODUCTS:

        shortage_results.append({
            "product_id": i,
            "planned_quantity": quantity[i],
            "allocated_quantity": sum(
                pyo.value(model.x[i, s])
                for s in model.SUPPLIERS
            ),
            "unfulfilled_quantity": pyo.value(
                model.shortage[i]
            )
        })

    shortage_df = pd.DataFrame(shortage_results)

    # Supplier-wise utilization
    supplier_summary = []

    for s in model.SUPPLIERS:

        allocated = sum(
            pyo.value(model.x[i, s])
            for i in model.PRODUCTS
        )

        supplier_summary.append({
            "supplier_id": s,
            "capacity": capacity[s],
            "allocated_quantity": allocated,
            "unused_capacity": capacity[s] - allocated,
            "capacity_utilization_percent":
                allocated / capacity[s] * 100
        })

    supplier_summary_df = pd.DataFrame(
        supplier_summary
    )

    # Save outputs
    allocation_df.to_csv(
        REPORT_DIR / "supplier_allocation.csv",
        index=False
    )

    shortage_df.to_csv(
        REPORT_DIR / "procurement_shortage.csv",
        index=False
    )

    supplier_summary_df.to_csv(
        REPORT_DIR / "supplier_summary.csv",
        index=False
    )

    print("\n========== SUPPLIER ALLOCATION ==========")
    print(allocation_df.head(15).to_string(index=False))

    print("\n========== SUPPLIER SUMMARY ==========")
    print(supplier_summary_df.to_string(index=False))

    print("\n========== SHORTAGE SUMMARY ==========")
    print(shortage_df.head(15).to_string(index=False))

    print(
        "\nTotal procurement cost:",
        round(
            sum(
                unit_cost[i] * pyo.value(model.x[i, s])
                for i in model.PRODUCTS
                for s in model.SUPPLIERS
            ), 2
        )
    )

    print(
        "Total shortage:",
        round(
            sum(
                pyo.value(model.shortage[i])
                for i in model.PRODUCTS
            ), 2
        )
    )

    print(
        "Total objective value:",
        round(pyo.value(model.objective), 2)
    )

    print("\nOptimization completed successfully!")

else:
    print("Optimization did not reach an optimal solution.")
