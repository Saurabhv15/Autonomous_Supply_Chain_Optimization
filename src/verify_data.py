import pandas as pd

from config import RAW_DATA_DIR

files = [
    "products.csv",
    "suppliers.csv",
    "warehouses.csv",
    "stores.csv",
    "sales_history.csv",
    "inventory.csv"
]

for filename in files:

    path = RAW_DATA_DIR / filename
    df = pd.read_csv(path)

    print("\nDataset:", filename)
    print("Shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum())
    print(df.head(3))

inventory = pd.read_csv(RAW_DATA_DIR / "inventory.csv")
warehouses = pd.read_csv(RAW_DATA_DIR / "warehouses.csv")

stock = inventory.groupby("warehouse_id")["stock_quantity"].sum()

check = warehouses.set_index("warehouse_id")[["capacity"]]
check["total_stock"] = stock
check["utilization_percent"] = (
    check["total_stock"] / check["capacity"]
) * 100

print("\nWAREHOUSE CAPACITY CHECK")
print(check)

print("\nAll warehouses within capacity:",
      (check["total_stock"] <= check["capacity"]).all())