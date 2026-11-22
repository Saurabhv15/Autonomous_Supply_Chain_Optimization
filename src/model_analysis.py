
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from config import PROCESSED_DATA_DIR, MODEL_DIR, REPORT_DIR

# -----------------------------------
# 1. LOAD MODEL AND TEST DATA
# -----------------------------------

model = joblib.load(
    MODEL_DIR / "demand_forecasting_model.pkl"
)

test = pd.read_csv(
    PROCESSED_DATA_DIR / "test_data.csv"
)

features = [
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "day_of_week",
    "month",
    "day_of_year"
]

X_test = test[features]
y_actual = test["demand"]

# -----------------------------------
# 2. PREDICT DEMAND
# -----------------------------------

predictions = model.predict(X_test)

predictions = np.maximum(predictions, 0)

test["predicted_demand"] = predictions

test["prediction_error"] = (
    test["demand"] - test["predicted_demand"]
)

test["absolute_error"] = (
    test["prediction_error"].abs()
)

# -----------------------------------
# 3. FEATURE IMPORTANCE
# -----------------------------------

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance.to_string(index=False))

plt.figure(figsize=(9, 5))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.title("XGBoost Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "feature_importance.png"
)

plt.show()

# -----------------------------------
# 4. ACTUAL VS PREDICTED DEMAND
# -----------------------------------

# Select one product-store combination
sample = test[
    (test["product_id"] == "P001") &
    (test["store_id"] == "S01")
].copy()

sample = sample.sort_values("date")

plt.figure(figsize=(13, 5))

plt.plot(
    sample["date"],
    sample["demand"],
    label="Actual Demand",
    marker="o"
)

plt.plot(
    sample["date"],
    sample["predicted_demand"],
    label="Predicted Demand",
    marker="x"
)

plt.title("Actual vs Predicted Demand - P001 at S01")
plt.xlabel("Date")
plt.ylabel("Demand Units")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "product_store_forecast.png"
)

plt.show()
# -----------------------------------
# 5. PREDICTION ERROR DISTRIBUTION
# -----------------------------------

plt.figure(figsize=(9, 5))

plt.hist(
    test["prediction_error"],
    bins=50
)

plt.axvline(
    0,
    linestyle="--"
)

plt.title("Prediction Error Distribution")
plt.xlabel("Actual - Predicted Demand")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "prediction_error.png"
)

plt.show()

# -----------------------------------
# 6. SAVE PREDICTIONS
# -----------------------------------
# Product-wise time series visualization

sample = test[
    (test["product_id"] == "P001") &
    (test["store_id"] == "S01")
].copy()

sample = sample.sort_values("date")

if sample.empty:
    print("No test records found for P001 at S01")
else:
    plt.figure(figsize=(13, 5))

    plt.plot(
        sample["date"],
        sample["demand"],
        label="Actual Demand",
        marker="o"
    )

    plt.plot(
        sample["date"],
        sample["predicted_demand"],
        label="Predicted Demand",
        marker="x"
    )

    plt.title("Actual vs Predicted Demand - P001 at S01")
    plt.xlabel("Date")
    plt.ylabel("Demand Units")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "product_store_forecast.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("Product-store forecast graph saved successfully!")

test.to_csv(
    REPORT_DIR / "forecast_predictions.csv",
    index=False
)

print("\nModel analysis completed!")
print("\nReports saved at:", REPORT_DIR)
