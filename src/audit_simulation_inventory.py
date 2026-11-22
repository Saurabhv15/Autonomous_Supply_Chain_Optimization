import pandas as pd
import numpy as np

from config import RAW_DATA_DIR, REPORT_DIR


# ============================================================
# CONSOLIDATED INVENTORY + RECEIPT-FLOW AUDIT
# Existing inventory checks are retained and receipt-level
# validations from the upgraded simulation are added here.
# ============================================================

checks = []


def record_check(name, passed, details):
    checks.append({
        "check": name,
        "status": "PASS" if bool(passed) else "FAIL",
        "details": str(details)
    })
    status = "PASS" if bool(passed) else "FAIL"
    print(f"[{status}] {name} | {details}")


def read_report(filename):
    path = REPORT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required report not found: {path}\n"
            "Run supply_chain_simulation.py first."
        )
    return pd.read_csv(path)


def max_abs(series):
    if len(series) == 0:
        return 0
    return float(series.abs().max())


print("\n========== CONSOLIDATED INVENTORY AUDIT ==========")

# ------------------------------------------------------------
# 1. LOAD SOURCE FILES
# ------------------------------------------------------------

initial_inventory = pd.read_csv(RAW_DATA_DIR / "inventory.csv")
warehouses = pd.read_csv(RAW_DATA_DIR / "warehouses.csv")

required_inventory_columns = {
    "warehouse_id", "product_id", "stock_quantity"
}
required_warehouse_columns = {"warehouse_id", "capacity"}

source_schema_ok = (
    required_inventory_columns.issubset(initial_inventory.columns)
    and required_warehouse_columns.issubset(warehouses.columns)
)

record_check(
    "Required source columns present",
    source_schema_ok,
    "Inventory and warehouse source schemas checked"
)

if not source_schema_ok:
    raise ValueError("Source CSV schema is incomplete. Audit stopped.")

record_check(
    "Initial inventory quantities are non-negative",
    initial_inventory["stock_quantity"].notna().all()
    and (initial_inventory["stock_quantity"] >= 0).all(),
    f"Minimum initial stock: {initial_inventory['stock_quantity'].min()}"
)

record_check(
    "Initial inventory has no duplicate warehouse-product rows",
    not initial_inventory.duplicated(
        ["warehouse_id", "product_id"]
    ).any(),
    f"Duplicate rows: {initial_inventory.duplicated(['warehouse_id', 'product_id']).sum()}"
)

record_check(
    "Warehouse IDs are unique",
    not warehouses["warehouse_id"].duplicated().any(),
    f"Duplicate warehouse IDs: {warehouses['warehouse_id'].duplicated().sum()}"
)

initial_by_warehouse = (
    initial_inventory.groupby("warehouse_id")["stock_quantity"].sum()
)
capacity_by_warehouse = warehouses.set_index("warehouse_id")["capacity"]

initial_check = initial_by_warehouse.to_frame("initial_stock").join(
    capacity_by_warehouse.rename("capacity"), how="left"
)
initial_check["capacity_excess"] = (
    initial_check["initial_stock"] - initial_check["capacity"]
)

record_check(
    "Initial inventory does not exceed warehouse capacity",
    initial_check["capacity"].notna().all()
    and (initial_check["capacity_excess"] <= 0).all(),
    f"Maximum initial capacity excess: {initial_check['capacity_excess'].max()}"
)

initial_stock_lookup = {
    (r.warehouse_id, r.product_id): float(r.stock_quantity)
    for r in initial_inventory.itertuples()
}

# ------------------------------------------------------------
# 2. SCENARIO FILES AND REQUIRED SCHEMAS
# ------------------------------------------------------------

scenario_files = {
    "Baseline": {
        "daily": "simulation_baseline_daily.csv",
        "deliveries": "simulation_baseline_deliveries.csv",
        "receipts": "simulation_baseline_warehouse_receipts.csv",
        "capacity": "simulation_baseline_capacity_events.csv"
    },
    "Disruption": {
        "daily": "simulation_disruption_daily.csv",
        "deliveries": "simulation_disruption_deliveries.csv",
        "receipts": "simulation_disruption_warehouse_receipts.csv",
        "capacity": "simulation_disruption_capacity_events.csv"
    }
}

required_daily = {
    "scenario", "day", "date", "warehouse_id", "product_id",
    "opening_inventory", "receipts", "demand", "fulfilled",
    "lost_sales", "ending_inventory"
}
required_delivery = {
    "supplier_id", "product_id", "quantity", "status",
    "delivery_day", "received_quantity", "unreceived_quantity"
}
required_receipt = {
    "simulation_day", "supplier_id", "product_id", "warehouse_id",
    "received_quantity", "warehouse_stock_after_receipt",
    "warehouse_capacity"
}
required_capacity = {
    "simulation_day", "warehouse_id", "stock_after_receipt",
    "capacity", "capacity_excess"
}

for scenario, files in scenario_files.items():

    print(f"\n========== {scenario.upper()} ==========")

    daily = read_report(files["daily"])
    deliveries = read_report(files["deliveries"])
    receipts = read_report(files["receipts"])
    capacity_events = read_report(files["capacity"])

    schema_checks = [
        (
            f"{scenario}: required daily columns present",
            required_daily.issubset(daily.columns),
            f"Missing: {sorted(required_daily - set(daily.columns))}"
        ),
        (
            f"{scenario}: required delivery columns present",
            required_delivery.issubset(deliveries.columns),
            f"Missing: {sorted(required_delivery - set(deliveries.columns))}"
        ),
        (
            f"{scenario}: required warehouse receipt columns present",
            required_receipt.issubset(receipts.columns),
            f"Missing: {sorted(required_receipt - set(receipts.columns))}"
        ),
        (
            f"{scenario}: required capacity-event columns present",
            required_capacity.issubset(capacity_events.columns),
            f"Missing: {sorted(required_capacity - set(capacity_events.columns))}"
        )
    ]

    for name, passed, details in schema_checks:
        record_check(name, passed, details)

    if not all(item[1] for item in schema_checks):
        print(f"{scenario}: detailed checks skipped due to schema failure.")
        continue

    # Convert numerical fields safely.
    daily_numeric = [
        "day", "opening_inventory", "receipts", "demand",
        "fulfilled", "lost_sales", "ending_inventory"
    ]
    for col in daily_numeric:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")

    for col in ["quantity", "received_quantity", "unreceived_quantity"]:
        deliveries[col] = pd.to_numeric(deliveries[col], errors="coerce")

    for col in ["delivery_day"]:
        deliveries[col] = pd.to_numeric(deliveries[col], errors="coerce")

    for col in ["simulation_day", "received_quantity",
                "warehouse_stock_after_receipt", "warehouse_capacity"]:
        receipts[col] = pd.to_numeric(receipts[col], errors="coerce")

    for col in ["simulation_day", "stock_after_receipt",
                "capacity", "capacity_excess"]:
        capacity_events[col] = pd.to_numeric(
            capacity_events[col], errors="coerce"
        )

    # --------------------------------------------------------
    # 3. RETAINED DAILY INVENTORY / DEMAND CHECKS
    # --------------------------------------------------------

    numeric_valid = (
        daily[daily_numeric].notna().all().all()
        and all(
            pd.api.types.is_numeric_dtype(daily[col])
            for col in daily_numeric
        )
    )
    record_check(
        f"{scenario}: simulation numeric fields are valid",
        numeric_valid,
        "Checked day, opening inventory, receipts, demand, fulfilled, "
        "lost sales and ending inventory"
    )

    nonnegative_cols = [
        "opening_inventory", "receipts", "demand",
        "fulfilled", "lost_sales", "ending_inventory"
    ]
    nonnegative_ok = (
        daily[nonnegative_cols].notna().all().all()
        and (daily[nonnegative_cols] >= 0).all().all()
    )
    record_check(
        f"{scenario}: no negative demand or inventory values",
        nonnegative_ok,
        "All daily stock and demand quantities must be >= 0"
    )

    demand_difference = (
        daily["demand"] - daily["fulfilled"] - daily["lost_sales"]
    )
    record_check(
        f"{scenario}: daily demand balance",
        np.allclose(demand_difference, 0, atol=1e-9),
        "Every row must satisfy demand = fulfilled + lost_sales"
    )

    record_check(
        f"{scenario}: fulfilled quantity does not exceed demand",
        (daily["fulfilled"] <= daily["demand"]).all(),
        f"Maximum fulfilled: {daily['fulfilled'].max()}"
    )

    duplicate_count = daily.duplicated(
        ["day", "warehouse_id", "product_id"]
    ).sum()
    record_check(
        f"{scenario}: no duplicate warehouse-product-day rows",
        duplicate_count == 0,
        f"Duplicate rows: {duplicate_count}"
    )

    # Existing ending inventory capacity check.
    daily_capacity = daily.merge(
        warehouses[["warehouse_id", "capacity"]],
        on="warehouse_id",
        how="left",
        validate="many_to_one"
    )
    missing_capacity = daily_capacity["capacity"].isna().any()
    ending_capacity_ok = (
        not missing_capacity
        and (
            daily_capacity["ending_inventory"]
            <= daily_capacity["capacity"]
        ).all()
    )
    record_check(
        f"{scenario}: ending inventory respects warehouse capacity",
        ending_capacity_ok,
        f"Maximum ending inventory: {daily_capacity['ending_inventory'].max()}; "
        f"capacity data missing: {missing_capacity}"
    )

    # First-day opening inventory must match source inventory.
    first_day = daily[daily["day"] == daily["day"].min()].copy()
    first_day["expected_opening"] = first_day.apply(
        lambda row: initial_stock_lookup.get(
            (row["warehouse_id"], row["product_id"]), np.nan
        ),
        axis=1
    )
    first_day_match = (
        first_day["expected_opening"].notna().all()
        and np.allclose(
            first_day["opening_inventory"],
            first_day["expected_opening"],
            atol=1e-9
        )
    )
    record_check(
        f"{scenario}: first-day opening inventory matches source inventory",
        first_day_match,
        f"Rows checked: {len(first_day)}; "
        f"unmatched rows: {first_day['expected_opening'].isna().sum()}"
    )

    # Daily inventory flow.
    stock_flow_difference = (
        daily["opening_inventory"]
        + daily["receipts"]
        - daily["fulfilled"]
        - daily["ending_inventory"]
    )
    record_check(
        f"{scenario}: daily stock-flow equation",
        np.allclose(stock_flow_difference, 0, atol=1e-9),
        f"Maximum absolute difference: {max_abs(stock_flow_difference)}"
    )

    # Consecutive-day inventory continuity.
    ordered_daily = daily.sort_values(
        ["warehouse_id", "product_id", "day"]
    ).copy()
    ordered_daily["previous_ending"] = ordered_daily.groupby(
        ["warehouse_id", "product_id"]
    )["ending_inventory"].shift(1)
    ordered_daily["previous_day"] = ordered_daily.groupby(
        ["warehouse_id", "product_id"]
    )["day"].shift(1)

    consecutive = ordered_daily[
        ordered_daily["previous_day"].notna()
        & (ordered_daily["day"] == ordered_daily["previous_day"] + 1)
    ]
    continuity_difference = (
        consecutive["opening_inventory"] - consecutive["previous_ending"]
    )
    record_check(
        f"{scenario}: consecutive-day inventory continuity",
        np.allclose(continuity_difference, 0, atol=1e-9),
        f"Transitions checked: {len(consecutive)}; "
        f"maximum difference: {max_abs(continuity_difference)}"
    )

    # --------------------------------------------------------
    # 4. NEW SUPPLIER DELIVERY / RECEIPT CHECKS
    # --------------------------------------------------------

    delivery_balance = (
        deliveries["quantity"]
        - deliveries["received_quantity"]
        - deliveries["unreceived_quantity"]
    )
    record_check(
        f"{scenario}: order quantity = received + unreceived",
        deliveries[
            ["quantity", "received_quantity", "unreceived_quantity"]
        ].notna().all().all()
        and np.allclose(delivery_balance, 0, atol=1e-9),
        f"Maximum absolute difference: {max_abs(delivery_balance)}"
    )

    failed_orders = deliveries["status"].astype(str).str.upper() == "FAILED"
    failed_order_ok = (
        (deliveries.loc[failed_orders, "received_quantity"] == 0).all()
        and (
            deliveries.loc[failed_orders, "unreceived_quantity"]
            == deliveries.loc[failed_orders, "quantity"]
        ).all()
    )
    record_check(
        f"{scenario}: failed orders have zero received quantity",
        failed_order_ok,
        f"Failed orders: {int(failed_orders.sum())}"
    )

    # Reconcile delivery-level totals with warehouse receipt details.
    receipt_by_order = (
        receipts.groupby(["supplier_id", "product_id"])["received_quantity"]
        .sum()
        .reset_index(name="receipt_detail_total")
    )
    delivery_by_order = (
        deliveries.groupby(["supplier_id", "product_id"])["received_quantity"]
        .sum()
        .reset_index(name="delivery_received_total")
    )
    order_reconciliation = delivery_by_order.merge(
        receipt_by_order,
        on=["supplier_id", "product_id"],
        how="outer"
    ).fillna(0)
    order_reconciliation["difference"] = (
        order_reconciliation["delivery_received_total"]
        - order_reconciliation["receipt_detail_total"]
    )

    record_check(
        f"{scenario}: delivery totals reconcile to warehouse receipt details",
        np.allclose(order_reconciliation["difference"], 0, atol=1e-9),
        f"Maximum difference: {max_abs(order_reconciliation['difference'])}"
    )

    # Reconcile daily receipt totals to receipt event records.
    # IMPORTANT: compare recorded simulation_day directly with daily 'day'.
    # The simulation uses day_number + 1 for daily records. If receipt logs
    # use env.now (zero-based), this check will expose the one-day offset.
    receipt_day_totals = (
        receipts.groupby(
            ["simulation_day", "warehouse_id", "product_id"]
        )["received_quantity"]
        .sum()
        .reset_index()
        .rename(columns={
            "simulation_day": "day",
            "received_quantity": "receipt_event_total"
        })
    )
    daily_receipt_totals = (
        daily.groupby(
            ["day", "warehouse_id", "product_id"]
        )["receipts"]
        .sum()
        .reset_index()
    )
    day_reconciliation = daily_receipt_totals.merge(
        receipt_day_totals,
        on=["day", "warehouse_id", "product_id"],
        how="outer"
    ).fillna(0)
    day_reconciliation["difference"] = (
        day_reconciliation["receipts"]
        - day_reconciliation["receipt_event_total"]
    )

    record_check(
        f"{scenario}: daily receipts match receipt event day and quantity",
        np.allclose(day_reconciliation["difference"], 0, atol=1e-9),
        f"Maximum difference: {max_abs(day_reconciliation['difference'])}; "
        "receipt simulation_day compared directly with daily day"
    )

    # Check receipt event stock values against the corresponding warehouse.
    receipt_capacity_diff = (
        receipts["warehouse_stock_after_receipt"]
        - receipts["warehouse_capacity"]
    )
    receipt_capacity_ok = (
        receipts[
            ["warehouse_stock_after_receipt", "warehouse_capacity"]
        ].notna().all().all()
        and (receipt_capacity_diff <= 1e-9).all()
    )
    record_check(
        f"{scenario}: warehouse capacity respected after every receipt",
        receipt_capacity_ok,
        f"Maximum stock-minus-capacity: {receipt_capacity_diff.max()}"
    )

    # Capacity event records must accurately report zero excess.
    event_excess = (
        capacity_events["stock_after_receipt"]
        - capacity_events["capacity"]
    )
    capacity_event_ok = (
        (event_excess <= 1e-9).all()
        and np.allclose(
            capacity_events["capacity_excess"],
            np.maximum(event_excess, 0),
            atol=1e-9
        )
    )
    record_check(
        f"{scenario}: capacity-event log is internally consistent",
        capacity_event_ok,
        f"Maximum actual excess: {event_excess.max()}; "
        f"reported excess max: {capacity_events['capacity_excess'].max()}"
    )

    # Warehouse-level stock after receipts must be at/below capacity.
    record_check(
        f"{scenario}: receipt records contain no negative quantities",
        receipts["received_quantity"].notna().all()
        and (receipts["received_quantity"] > 0).all(),
        f"Receipt rows: {len(receipts)}"
    )

    # Save diagnostic reconciliations.
    day_reconciliation.to_csv(
        REPORT_DIR / f"audit_{scenario.lower()}_daily_receipt_reconciliation.csv",
        index=False
    )
    order_reconciliation.to_csv(
        REPORT_DIR / f"audit_{scenario.lower()}_delivery_receipt_reconciliation.csv",
        index=False
    )

# ------------------------------------------------------------
# 5. SAVE CONSOLIDATED AUDIT REPORT
# ------------------------------------------------------------

audit_df = pd.DataFrame(checks)
output_path = REPORT_DIR / "audit_simulation_inventory_checks.csv"
audit_df.to_csv(output_path, index=False)

passed = int((audit_df["status"] == "PASS").sum())
failed = int((audit_df["status"] == "FAIL").sum())

print("\n========== CONSOLIDATED INVENTORY AUDIT SUMMARY ==========")
print(f"Total checks : {len(audit_df)}")
print(f"Passed       : {passed}")
print(f"Failed       : {failed}")
print("Audit report :", output_path)

if failed == 0:
    print("\nALL CONSOLIDATED INVENTORY AND RECEIPT-FLOW CHECKS PASSED!")
else:
    print("\nSOME CHECKS FAILED. Review the FAIL rows before accepting the simulation.")

print(
    "\nSCOPE NOTE: This audit checks daily inventory arithmetic, "
    "initial inventory, daily continuity, supplier delivery reconciliation, "
    "warehouse-level receipt logs and capacity after each receipt. "
    "A PASS confirms consistency of the recorded outputs; it does not prove "
    "that the simulation policy is optimal or that real-world operations "
    "would behave identically."
)
