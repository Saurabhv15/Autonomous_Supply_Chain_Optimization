
import numpy as np
import pandas as pd

from config import (
    RAW_DATA_DIR,
    NUM_SUPPLIERS,
    NUM_WAREHOUSES,
    NUM_STORES,
    NUM_PRODUCTS,
    HISTORICAL_DAYS,
    RANDOM_SEED
)

# Reproducible random generation
rng = np.random.default_rng(RANDOM_SEED)

print("Generating supply chain datasets...")

# ----------------------------------
# 1. PRODUCT DATA
# ----------------------------------

products = pd.DataFrame({
    "product_id": [
        f"P{i:03d}" for i in range(1, NUM_PRODUCTS + 1)
    ],
    "unit_cost": rng.integers(50, 1000, NUM_PRODUCTS),
    "holding_cost_rate": rng.uniform(0.10, 0.30, NUM_PRODUCTS)
})

products["selling_price"] = (
    products["unit_cost"] * 1.5
).round(2)


# ----------------------------------
# 2. SUPPLIER DATA
# ----------------------------------

suppliers = pd.DataFrame({
    "supplier_id": [
        f"SUP{i:02d}" for i in range(1, NUM_SUPPLIERS + 1)
    ],
    "reliability": rng.uniform(0.85, 0.99, NUM_SUPPLIERS),
    "lead_time_days": rng.integers(3, 10, NUM_SUPPLIERS),

    # Maximum quantity supplied in one procurement planning cycle
    "capacity_per_cycle": rng.integers(8000, 15000, NUM_SUPPLIERS)
})

# ----------------------------------
# 3. WAREHOUSE DATA
# ----------------------------------

warehouses = pd.DataFrame({
    "warehouse_id": [
        f"W{i:02d}" for i in range(1, NUM_WAREHOUSES + 1)
    ],
    "capacity": rng.integers(5000, 15000, NUM_WAREHOUSES),
    "holding_cost_per_unit": rng.uniform(2, 10, NUM_WAREHOUSES),
    "latitude": rng.uniform(20, 30, NUM_WAREHOUSES),
    "longitude": rng.uniform(75, 90, NUM_WAREHOUSES)
})

# ----------------------------------
# 4. RETAIL STORE DATA
# ----------------------------------

stores = pd.DataFrame({
    "store_id": [
        f"S{i:02d}" for i in range(1, NUM_STORES + 1)
    ],
    "region": rng.choice(
        ["North", "South", "East", "West"],
        NUM_STORES
    ),
    "demand_factor": rng.uniform(0.5, 2.0, NUM_STORES),
    "latitude": rng.uniform(20, 30, NUM_STORES),
    "longitude": rng.uniform(75, 90, NUM_STORES)
})

# ----------------------------------
# 5. HISTORICAL SALES / DEMAND DATA
# ----------------------------------

dates = pd.date_range(
    end=pd.Timestamp.today().normalize(),
    periods=HISTORICAL_DAYS,
    freq="D"
)

# Generate all product-store-date combinations
history = pd.MultiIndex.from_product(
    [
        products["product_id"],
        stores["store_id"],
        dates
    ],
    names=["product_id", "store_id", "date"]
).to_frame(index=False)

# Product-specific demand base
product_base = pd.Series(
    rng.integers(5, 30, NUM_PRODUCTS),
    index=products["product_id"]
)

history["base_demand"] = history["product_id"].map(product_base)
history["demand_factor"] = history["store_id"].map(
    stores.set_index("store_id")["demand_factor"]
)

# Weekly and monthly seasonality
history["day_of_week"] = history["date"].dt.dayofweek
history["day_number"] = (
    history["date"] - dates[0]
).dt.days

week_effect = np.where(
    history["day_of_week"] >= 5, 1.2, 1.0
)

month_effect = (
    1 + 0.15 * np.sin(
        2 * np.pi * history["day_number"] / 30
    )
)

noise = rng.normal(0, 3, len(history))

history["demand"] = np.maximum(
    0,
    np.round(
        history["base_demand"]
        * history["demand_factor"]
        * week_effect
        * month_effect
        + noise
    )
).astype(int)

sales_history = history[
    ["date", "product_id", "store_id", "demand"]
].copy()


# ----------------------------------
# 6. INITIAL INVENTORY DATA
# Respect warehouse storage capacity
# ----------------------------------

inventory_rows = []

for _, warehouse in warehouses.iterrows():

    warehouse_id = warehouse["warehouse_id"]
    capacity = int(warehouse["capacity"])

    # Target inventory utilization: 60% to 80%
    utilization = rng.uniform(0.60, 0.80)

    total_stock = int(capacity * utilization)

    # Distribute inventory across 100 products
    weights = rng.dirichlet(
        np.ones(NUM_PRODUCTS) * 3
    )

    quantities = rng.multinomial(
        total_stock, weights
    )

    for product_id, quantity in zip(
        products["product_id"], quantities
    ):

        inventory_rows.append({
            "warehouse_id": warehouse_id,
            "product_id": product_id,
            "stock_quantity": int(quantity)
        })

inventory = pd.DataFrame(inventory_rows)


# ----------------------------------
# 7. SAVE DATASETS
# ----------------------------------

datasets = {
    "products": products,
    "suppliers": suppliers,
    "warehouses": warehouses,
    "stores": stores,
    "sales_history": sales_history,
    "inventory": inventory
}

for name, dataframe in datasets.items():

    file_path = RAW_DATA_DIR / f"{name}.csv"

    dataframe.to_csv(file_path, index=False)

    print(
        f"{name}.csv saved | "
        f"Rows: {len(dataframe)} | "
        f"Columns: {len(dataframe.columns)}"
    )

print("\nAll datasets generated successfully!")
print("Data location:", RAW_DATA_DIR)
