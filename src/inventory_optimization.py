
import pandas as pd
import numpy as np

from config import RAW_DATA_DIR, REPORT_DIR

# ==========================================
# STEP 1: LOAD DATA
# ==========================================

products = pd.read_csv(
    RAW_DATA_DIR / "products.csv"
)

sales = pd.read_csv(
    RAW_DATA_DIR / "sales_history.csv"
)

print("Datasets loaded successfully.")

# ==========================================
# STEP 2: CALCULATE PRODUCT-WISE DEMAND
# ==========================================

product_demand = (
    sales.groupby("product_id")["demand"]
    .sum()
    .reset_index()
)

product_demand.rename(
    columns={"demand": "demand_180_days"},
    inplace=True
)

# ==========================================
# STEP 3: CALCULATE ANNUAL DEMAND
# ==========================================

product_demand["annual_demand"] = (
    product_demand["demand_180_days"] / 180
) * 365

# ==========================================
# STEP 4: MERGE PRODUCT INFORMATION
# ==========================================

inventory_data = pd.merge(
    products,
    product_demand,
    on="product_id",
    how="inner"
)

# ==========================================
# STEP 5: CALCULATE HOLDING COST
# ==========================================

inventory_data["annual_holding_cost"] = (
    inventory_data["unit_cost"] *
    inventory_data["holding_cost_rate"]
)

# ==========================================
# STEP 6: DEFINE ORDERING COST
# ==========================================

ORDERING_COST = 500

inventory_data["ordering_cost"] = ORDERING_COST

# ==========================================
# STEP 7: CALCULATE EOQ
# ==========================================

inventory_data["EOQ"] = np.sqrt(
    (
        2 *
        inventory_data["annual_demand"] *
        inventory_data["ordering_cost"]
    ) /
    inventory_data["annual_holding_cost"]
)

inventory_data["EOQ"] = np.ceil(
    inventory_data["EOQ"]
).astype(int)

# ==========================================
# STEP 8: CALCULATE ANNUAL ORDER FREQUENCY
# ==========================================

inventory_data["orders_per_year"] = (
    inventory_data["annual_demand"] /
    inventory_data["EOQ"]
)

# ==========================================
# STEP 9: CALCULATE ANNUAL INVENTORY COST
# ==========================================

inventory_data["annual_ordering_cost"] = (
    inventory_data["annual_demand"] /
    inventory_data["EOQ"]
) * ORDERING_COST

inventory_data["annual_holding_cost_total"] = (
    inventory_data["EOQ"] / 2
) * inventory_data["annual_holding_cost"]

inventory_data["total_relevant_cost"] = (
    inventory_data["annual_ordering_cost"] +
    inventory_data["annual_holding_cost_total"]
)

# ==========================================
# STEP 10: SAVE RESULTS
# ==========================================

output_columns = [
    "product_id",
    "unit_cost",
    "annual_demand",
    "holding_cost_rate",
    "annual_holding_cost",
    "ordering_cost",
    "EOQ",
    "orders_per_year",
    "annual_ordering_cost",
    "annual_holding_cost_total",
    "total_relevant_cost"
]

eoq_results = inventory_data[output_columns]

eoq_results.to_csv(
    REPORT_DIR / "eoq_results.csv",
    index=False
)

# ==========================================
# STEP 11: DISPLAY RESULTS
# ==========================================

print("\n========== EOQ RESULTS ==========")

print(
    eoq_results.head(10).to_string(index=False)
)

print("\n========== SUMMARY ==========")

print("Total products:", len(eoq_results))

print(
    "Average EOQ:",
    round(eoq_results["EOQ"].mean(), 2)
)

print(
    "Total annual relevant inventory cost: ₹",
    round(eoq_results["total_relevant_cost"].sum(), 2)
)

print("\nEOQ optimization completed successfully!")
print("Results saved in reports/eoq_results.csv")

suppliers = pd.read_csv(
    RAW_DATA_DIR / "suppliers.csv"
)

inventory = pd.read_csv(
    RAW_DATA_DIR / "inventory.csv"
)

print("\n========== SUPPLIER DATA ==========")
print(suppliers.columns.tolist())
print(suppliers.head())

print("\n========== INVENTORY DATA ==========")
print(inventory.columns.tolist())
print(inventory.head())


# ==========================================
# STEP 12: SAFETY STOCK AND REORDER POINT
# ==========================================

# Service level assumption: 95%
SERVICE_LEVEL_Z = 1.645

# Conservative lead time assumption
LEAD_TIME = 6

# Calculate daily network demand for each product
daily_product_demand = (
    sales.groupby(["date", "product_id"])["demand"]
    .sum()
    .reset_index()
)

# Calculate average and standard deviation of daily demand
demand_statistics = (
    daily_product_demand.groupby("product_id")["demand"]
    .agg(
        average_daily_demand="mean",
        daily_demand_std="std"
    )
    .reset_index()
)

# Merge demand statistics with EOQ results
inventory_policy = pd.merge(
    eoq_results,
    demand_statistics,
    on="product_id",
    how="inner"
)

# Calculate Safety Stock
inventory_policy["safety_stock"] = (
    SERVICE_LEVEL_Z *
    inventory_policy["daily_demand_std"] *
    np.sqrt(LEAD_TIME)
)

inventory_policy["safety_stock"] = np.ceil(
    inventory_policy["safety_stock"]
).astype(int)

# Calculate Reorder Point
inventory_policy["reorder_point"] = np.ceil(
    inventory_policy["average_daily_demand"] * LEAD_TIME
    + inventory_policy["safety_stock"]
).astype(int)

# ==========================================
# STEP 13: AGGREGATE CURRENT INVENTORY
# ==========================================

current_stock = (
    inventory.groupby("product_id")["stock_quantity"]
    .sum()
    .reset_index()
)

current_stock.rename(
    columns={"stock_quantity": "current_network_stock"},
    inplace=True
)

inventory_policy = pd.merge(
    inventory_policy,
    current_stock,
    on="product_id",
    how="left"
)

# ==========================================
# STEP 14: CHECK REORDER STATUS
# ==========================================

inventory_policy["inventory_status"] = np.where(
    inventory_policy["current_network_stock"]
    <= inventory_policy["reorder_point"],
    "REORDER REQUIRED",
    "STOCK SUFFICIENT"
)

# ==========================================
# STEP 15: SAVE INVENTORY POLICY
# ==========================================

policy_columns = [
    "product_id",
    "annual_demand",
    "average_daily_demand",
    "daily_demand_std",
    "EOQ",
    "safety_stock",
    "reorder_point",
    "current_network_stock",
    "inventory_status"
]

inventory_policy[policy_columns].to_csv(
    REPORT_DIR / "inventory_policy_results.csv",
    index=False
)

# ==========================================
# STEP 16: DISPLAY RESULTS
# ==========================================

print("\n========== SAFETY STOCK AND ROP ==========")

print(
    inventory_policy[policy_columns]
    .head(10)
    .to_string(index=False)
)

print("\n========== REORDER SUMMARY ==========")

print(
    inventory_policy["inventory_status"]
    .value_counts()
)

print("\nSafety Stock and ROP calculations completed!")
print(
    "Results saved at:",
    REPORT_DIR / "inventory_policy_results.csv"
)


# ==========================================
# STEP 17: INVENTORY COVERAGE ANALYSIS
# ==========================================

inventory_policy["days_of_inventory"] = (
    inventory_policy["current_network_stock"] /
    inventory_policy["average_daily_demand"]
)

inventory_policy["stock_shortfall"] = (
    inventory_policy["reorder_point"] -
    inventory_policy["current_network_stock"]
).clip(lower=0)

inventory_policy["suggested_order_quantity"] = (
    inventory_policy["EOQ"]
).astype(int)

print("\n========== INVENTORY COVERAGE ==========")

print(
    inventory_policy[
        [
            "product_id",
            "current_network_stock",
            "average_daily_demand",
            "days_of_inventory",
            "reorder_point",
            "stock_shortfall",
            "suggested_order_quantity"
        ]
    ].head(10).to_string(index=False)
)

print("\nAverage inventory coverage:",
      round(inventory_policy["days_of_inventory"].mean(), 2),
      "days")

print("\nTotal suggested order quantity:",
      inventory_policy["suggested_order_quantity"].sum(),
      "units")

# ==========================================
# STEP 18: SUPPLIER CAPACITY ANALYSIS
# ==========================================

total_supplier_capacity = suppliers["capacity_per_cycle"].sum()

total_suggested_quantity = (
    inventory_policy["suggested_order_quantity"].sum()
)

capacity_gap = (
    total_suggested_quantity - total_supplier_capacity
)

capacity_utilization = (
    total_suggested_quantity / total_supplier_capacity
) * 100

print("\n========== SUPPLIER CAPACITY ANALYSIS ==========")

print("Total supplier capacity:", total_supplier_capacity)

print("Total suggested quantity:", total_suggested_quantity)

print("Capacity gap:", capacity_gap)

print(
    "Required capacity as percentage of available capacity:",
    round(capacity_utilization, 2), "%"
)