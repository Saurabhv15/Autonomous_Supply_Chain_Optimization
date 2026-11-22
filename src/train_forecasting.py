
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import PROCESSED_DATA_DIR, MODEL_DIR, REPORT_DIR

# -----------------------------------------
# 1. LOAD TRAIN AND TEST DATA
# -----------------------------------------

train = pd.read_csv(PROCESSED_DATA_DIR / "train_data.csv")
test = pd.read_csv(PROCESSED_DATA_DIR / "test_data.csv")

print("Training shape:", train.shape)
print("Testing shape:", test.shape)

# -----------------------------------------
# 2. SELECT FEATURES AND TARGET
# -----------------------------------------

features = [
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "day_of_week",
    "month",
    "day_of_year"
]

X_train = train[features]
y_train = train["demand"]

X_test = test[features]
y_test = test["demand"]

# -----------------------------------------
# 3. DEFINE EVALUATION FUNCTION
# -----------------------------------------

def evaluate_model(name, actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(mean_squared_error(actual, predicted))

    wape = (
        np.sum(np.abs(actual - predicted))
        / np.sum(actual)
    ) * 100

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "WAPE (%)": wape
    }

results = []

# -----------------------------------------
# 4. LAG-7 BASELINE
# -----------------------------------------

lag7_predictions = test["lag_7"].values

results.append(
    evaluate_model(
        "Lag-7 Baseline",
        y_test.values,
        lag7_predictions
    )
)

# -----------------------------------------
# 5. MOVING AVERAGE BASELINE
# -----------------------------------------

moving_predictions = test["rolling_mean_7"].values

results.append(
    evaluate_model(
        "7-Day Moving Average",
        y_test.values,
        moving_predictions
    )
)

# -----------------------------------------
# 6. TRAIN XGBOOST MODEL
# -----------------------------------------

print("\nTraining XGBoost model...")

model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("XGBoost training completed!")

# -----------------------------------------
# 7. MAKE PREDICTIONS
# -----------------------------------------

xgb_predictions = model.predict(X_test)

# Demand cannot be negative
xgb_predictions = np.maximum(xgb_predictions, 0)

results.append(
    evaluate_model(
        "XGBoost",
        y_test.values,
        xgb_predictions
    )
)

# -----------------------------------------
# 8. DISPLAY MODEL COMPARISON
# -----------------------------------------

results_df = pd.DataFrame(results)

print("\n========== MODEL PERFORMANCE ==========")
print(results_df.to_string(index=False))

# -----------------------------------------
# 9. SAVE BEST MODEL
# -----------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

best_model_name = results_df.loc[
    results_df["WAPE (%)"].idxmin(),
    "Model"
]

if best_model_name == "XGBoost":

    joblib.dump(
        model,
        MODEL_DIR / "demand_forecasting_model.pkl"
    )

    print("\nXGBoost saved successfully!")

else:
    print(
        "\nA baseline performed better. "
        "Do not automatically replace it with XGBoost."
    )

# -----------------------------------------
# 10. SAVE RESULTS
# -----------------------------------------

results_df.to_csv(
    REPORT_DIR / "forecasting_results.csv",
    index=False
)

print("\nResults saved successfully!")
print("Best model by WAPE:", best_model_name)
