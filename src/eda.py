
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import RAW_DATA_DIR, REPORT_DIR

# ---------------------------------
# 1. LOAD DATASETS
# ---------------------------------

sales = pd.read_csv(
    RAW_DATA_DIR / "sales_history.csv",
    parse_dates=["date"]
)

products = pd.read_csv(RAW_DATA_DIR / "products.csv")
suppliers = pd.read_csv(RAW_DATA_DIR / "suppliers.csv")
warehouses = pd.read_csv(RAW_DATA_DIR / "warehouses.csv")
stores = pd.read_csv(RAW_DATA_DIR / "stores.csv")
inventory = pd.read_csv(RAW_DATA_DIR / "inventory.csv")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

print("\n========== SUPPLY CHAIN EDA ==========")

# ---------------------------------
# 2. DEMAND STATISTICS
# ---------------------------------

print("\nSales dataset shape:", sales.shape)
print("\nDemand statistics:")
print(sales["demand"].describe())

print("\nTotal demand:", sales["demand"].sum())

daily_demand = (
    sales.groupby("date")["demand"]
    .sum()
    .sort_index()
)

print("\nAverage daily network demand:", daily_demand.mean())

# ---------------------------------
# 3. DAILY DEMAND TREND
# ---------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    daily_demand.index,
    daily_demand.values,
    alpha=0.5,
    label="Daily Demand"
)

plt.plot(
    daily_demand.index,
    daily_demand.rolling(7).mean(),
    linewidth=2,
    label="7-Day Moving Average"
)

plt.title("Daily Supply Chain Demand Trend")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.tight_layout()
plt.savefig(REPORT_DIR / "daily_demand.png")
plt.show()

# ---------------------------------
# 4. DEMAND DISTRIBUTION
# ---------------------------------

plt.figure(figsize=(10, 5))

sns.histplot(sales["demand"], bins=40, kde=True)

plt.title("Product Demand Distribution")
plt.xlabel("Demand Units")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(REPORT_DIR / "demand_distribution.png")
plt.show()

# ---------------------------------
# 5. TOP 10 PRODUCTS
# ---------------------------------

product_demand = (
    sales.groupby("product_id")["demand"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Products:")
print(product_demand)

plt.figure(figsize=(11, 5))
sns.barplot(x=product_demand.values, y=product_demand.index)

plt.title("Top 10 Products by Total Demand")
plt.xlabel("Total Demand")
plt.ylabel("Product ID")
plt.tight_layout()
plt.savefig(REPORT_DIR / "top_products.png")
plt.show()

# ---------------------------------
# 6. STORE-WISE DEMAND
# ---------------------------------

store_demand = (
    sales.groupby("store_id")["demand"]
    .sum()
    .sort_values(ascending=False)
)

print("\nStore-wise demand:")
print(store_demand)

plt.figure(figsize=(11, 5))
sns.barplot(x=store_demand.values, y=store_demand.index)

plt.title("Total Demand by Store")
plt.xlabel("Total Demand")
plt.ylabel("Store ID")
plt.tight_layout()
plt.savefig(REPORT_DIR / "store_demand.png")
plt.show()

# ---------------------------------
# 7. WEEKDAY DEMAND ANALYSIS
# ---------------------------------

sales["day_of_week"] = sales["date"].dt.day_name()

weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekday_demand = (
    sales.groupby("day_of_week")["demand"]
    .mean()
    .reindex(weekday_order)
)

print("\nAverage demand by weekday:")
print(weekday_demand)

plt.figure(figsize=(10, 5))
sns.barplot(
    x=weekday_demand.index,
    y=weekday_demand.values
)

plt.title("Average Demand by Day of Week")
plt.xlabel("Day")
plt.ylabel("Average Product-Store Demand")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(REPORT_DIR / "weekday_demand.png")
plt.show()

# ---------------------------------
# 8. WAREHOUSE UTILIZATION
# ---------------------------------

warehouse_stock = (
    inventory.groupby("warehouse_id")["stock_quantity"]
    .sum()
    .reset_index()
)

warehouse_analysis = warehouse_stock.merge(
    warehouses[["warehouse_id", "capacity"]],
    on="warehouse_id"
)

warehouse_analysis["utilization_percent"] = (
    warehouse_analysis["stock_quantity"]
    / warehouse_analysis["capacity"]
) * 100

print("\nWarehouse utilization:")
print(warehouse_analysis)

plt.figure(figsize=(11, 5))

sns.barplot(
    data=warehouse_analysis,
    x="warehouse_id",
    y="utilization_percent"
)

plt.axhline(
    100,
    linestyle="--",
    label="Maximum Capacity"
)

plt.title("Warehouse Capacity Utilization")
plt.xlabel("Warehouse")
plt.ylabel("Utilization (%)")
plt.legend()
plt.tight_layout()
plt.savefig(REPORT_DIR / "warehouse_utilization.png")
plt.show()

# ---------------------------------
# 9. SUPPLIER RELIABILITY
# ---------------------------------

print("\nSupplier Analysis:")
print(suppliers)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=suppliers,
    x="supplier_id",
    y="reliability"
)

plt.title("Supplier Reliability Comparison")
plt.xlabel("Supplier")
plt.ylabel("Reliability")
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig(REPORT_DIR / "supplier_reliability.png")
plt.show()

# ---------------------------------
# 10. SAVE EDA SUMMARY
# ---------------------------------

summary = pd.DataFrame({
    "Metric": [
        "Total historical demand",
        "Average daily network demand",
        "Number of products",
        "Number of stores",
        "Number of warehouses",
        "Number of suppliers",
        "Average supplier reliability",
        "Average warehouse utilization (%)"
    ],
    "Value": [
        sales["demand"].sum(),
        daily_demand.mean(),
        products["product_id"].nunique(),
        stores["store_id"].nunique(),
        warehouses["warehouse_id"].nunique(),
        suppliers["supplier_id"].nunique(),
        suppliers["reliability"].mean(),
        warehouse_analysis["utilization_percent"].mean()
    ]
})

summary.to_csv(
    REPORT_DIR / "eda_summary.csv",
    index=False
)

print("\nEDA completed successfully!")
print(summary.to_string(index=False))

print("\nReports saved at:", REPORT_DIR)
