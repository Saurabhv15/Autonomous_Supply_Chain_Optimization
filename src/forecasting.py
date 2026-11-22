
import pandas as pd
import numpy as np

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# -----------------------------------------
# 1. LOAD HISTORICAL SALES DATA
# -----------------------------------------

sales = pd.read_csv(
    RAW_DATA_DIR / "sales_history.csv",
    parse_dates=["date"]
)

print("Original dataset shape:", sales.shape)

# -----------------------------------------
# 2. SORT DATA
# -----------------------------------------

sales = sales.sort_values(
    ["product_id", "store_id", "date"]
).reset_index(drop=True)

# -----------------------------------------
# 3. CREATE LAG FEATURES
# -----------------------------------------

group_columns = ["product_id", "store_id"]

sales["lag_1"] = (
    sales.groupby(group_columns)["demand"].shift(1)
)

sales["lag_7"] = (
    sales.groupby(group_columns)["demand"].shift(7)
)

# -----------------------------------------
# 4. CREATE ROLLING FEATURES
# -----------------------------------------

sales["rolling_mean_7"] = (
    sales.groupby(group_columns)["demand"]
    .transform(
        lambda x: x.shift(1).rolling(window=7).mean()
    )
)

# -----------------------------------------
# 5. CREATE DATE FEATURES
# -----------------------------------------

sales["day_of_week"] = sales["date"].dt.dayofweek
sales["month"] = sales["date"].dt.month
sales["day_of_year"] = sales["date"].dt.dayofyear

# -----------------------------------------
# 6. REMOVE ROWS WITH MISSING FEATURES
# -----------------------------------------

sales = sales.dropna().reset_index(drop=True)

print("\nDataset after feature engineering:", sales.shape)

# -----------------------------------------
# 7. CHRONOLOGICAL TRAIN-TEST SPLIT
# -----------------------------------------

unique_dates = sorted(sales["date"].unique())

split_index = int(len(unique_dates) * 0.80)

split_date = unique_dates[split_index]

train_data = sales[sales["date"] < split_date].copy()

test_data = sales[sales["date"] >= split_date].copy()

print("\nTrain-test split date:", split_date)

print("Training dataset:", train_data.shape)
print("Testing dataset:", test_data.shape)

# -----------------------------------------
# 8. DEFINE FEATURES AND TARGET
# -----------------------------------------

feature_columns = [
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "day_of_week",
    "month",
    "day_of_year"
]

X_train = train_data[feature_columns]
y_train = train_data["demand"]

X_test = test_data[feature_columns]
y_test = test_data["demand"]

# -----------------------------------------
# 9. SAVE PROCESSED DATA
# -----------------------------------------

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

train_data.to_csv(
    PROCESSED_DATA_DIR / "train_data.csv",
    index=False
)

test_data.to_csv(
    PROCESSED_DATA_DIR / "test_data.csv",
    index=False
)

print("\nFeature engineering completed successfully!")

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nFeature columns:", feature_columns)

print("\nSample training data:")
print(train_data.head(10).to_string(index=False))
