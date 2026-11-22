import simpy
import pandas as pd
import numpy as np

from config import RAW_DATA_DIR, REPORT_DIR


# --------------------------------------------------
# 1. LOAD AND VALIDATE INPUT DATA
# --------------------------------------------------

suppliers = pd.read_csv(RAW_DATA_DIR / "suppliers.csv")
inventory_df = pd.read_csv(RAW_DATA_DIR / "inventory.csv")
sales = pd.read_csv(RAW_DATA_DIR / "sales_history.csv")
warehouses = pd.read_csv(RAW_DATA_DIR / "warehouses.csv")

assignment = pd.read_csv(
    REPORT_DIR / "store_warehouse_assignment.csv"
)
supplier_allocation = pd.read_csv(
    REPORT_DIR / "supplier_allocation_v5.csv"
)

required_columns = {
    "suppliers": (suppliers, {"supplier_id", "reliability", "lead_time_days"}),
    "inventory": (inventory_df, {"warehouse_id", "product_id", "stock_quantity"}),
    "sales": (sales, {"date", "product_id", "store_id", "demand"}),
    "warehouses": (warehouses, {"warehouse_id", "capacity"}),
    "assignment": (assignment, {"store_id", "warehouse_id"}),
    "supplier_allocation": (
        supplier_allocation,
        {"supplier_id", "product_id", "allocated_quantity"}
    ),
}

for dataset_name, (dataset, required) in required_columns.items():
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {sorted(missing)}"
        )

if inventory_df.duplicated(["warehouse_id", "product_id"]).any():
    raise ValueError("Duplicate warehouse-product rows in inventory.csv.")

if warehouses["warehouse_id"].duplicated().any():
    raise ValueError("Duplicate warehouse IDs in warehouses.csv.")

if (inventory_df["stock_quantity"] < 0).any():
    raise ValueError("Initial inventory contains negative stock.")

if (warehouses["capacity"] < 0).any():
    raise ValueError("Warehouse capacity cannot be negative.")

sales["date"] = pd.to_datetime(sales["date"])

sales = sales.merge(
    assignment[["store_id", "warehouse_id"]],
    on="store_id",
    how="left",
    validate="many_to_one"
)

if sales["warehouse_id"].isna().any():
    raise ValueError("Some stores do not have warehouse assignments.")

daily_demand = (
    sales.groupby(
        ["date", "warehouse_id", "product_id"],
        as_index=False
    )["demand"].sum()
)

simulation_dates = sorted(daily_demand["date"].unique())[-30:]
daily_demand = daily_demand[
    daily_demand["date"].isin(simulation_dates)
].copy()

orders = supplier_allocation.merge(
    suppliers,
    on="supplier_id",
    how="left",
    validate="many_to_one"
)

if orders["reliability"].isna().any():
    raise ValueError("Supplier information missing for an order.")

if orders["lead_time_days"].isna().any():
    raise ValueError("Supplier lead time missing for an order.")

if not orders["reliability"].between(0, 1).all():
    raise ValueError("Supplier reliability must be between 0 and 1.")

# --------------------------------------------------
# 2. PREPARE LOOKUPS
# --------------------------------------------------

warehouse_capacity = (
    warehouses.set_index("warehouse_id")["capacity"].to_dict()
)

initial_stock = {
    (row.warehouse_id, row.product_id): int(row.stock_quantity)
    for row in inventory_df.itertuples()
}

warehouse_ids = list(warehouse_capacity)

# Demand-based warehouse weights for each product.
warehouse_product_demand = (
    daily_demand.groupby(["warehouse_id", "product_id"])["demand"]
    .sum()
    .to_dict()
)

product_warehouses = {}
for product_id in daily_demand["product_id"].unique():
    product_warehouses[product_id] = {
        warehouse_id: quantity
        for (warehouse_id, pid), quantity in warehouse_product_demand.items()
        if pid == product_id and quantity > 0
    }

# --------------------------------------------------
# 3. CAPACITY-AWARE WEIGHTED ALLOCATION
# --------------------------------------------------

def allocate_quantity_by_capacity(
    product_id, quantity, stock, warehouse_capacity
):
    """
    Distribute an incoming product quantity across its demand warehouses.
    Uses demand weights and repeatedly redistributes quantities that do not
    fit because of warehouse capacity.
    """
    candidates = product_warehouses.get(product_id, {})
    remaining = int(quantity)
    allocations = []

    active = {
        wid: float(weight)
        for wid, weight in candidates.items()
        if weight > 0
    }

    while remaining > 0 and active:
        spaces = {}
        for wid in active:
            current_total = sum(
                value for (warehouse, _), value in stock.items()
                if warehouse == wid
            )
            spaces[wid] = max(
                0, int(warehouse_capacity[wid] - current_total)
            )

        active = {
            wid: weight for wid, weight in active.items()
            if spaces[wid] > 0
        }
        if not active:
            break

        total_weight = sum(active.values())
        proposed = {
            wid: remaining * weight / total_weight
            for wid, weight in active.items()
        }

        round_allocations = {
            wid: min(spaces[wid], int(np.floor(value)))
            for wid, value in proposed.items()
        }

        allocated_this_round = sum(round_allocations.values())

        for wid, amount in round_allocations.items():
            if amount > 0:
                key = (wid, product_id)
                stock[key] = stock.get(key, 0) + amount
                allocations.append((wid, amount))

        remaining -= allocated_this_round

        # Allocate fractional remainders one unit at a time.
        if remaining > 0:
            ranking = sorted(
                active,
                key=lambda wid: (
                    proposed[wid] - np.floor(proposed[wid]),
                    active[wid]
                ),
                reverse=True
            )

            for wid in ranking:
                if remaining <= 0:
                    break

                current_total = sum(
                    value for (warehouse, _), value in stock.items()
                    if warehouse == wid
                )
                if current_total < warehouse_capacity[wid]:
                    key = (wid, product_id)
                    stock[key] = stock.get(key, 0) + 1
                    allocations.append((wid, 1))
                    remaining -= 1

        # Remove full warehouses before redistributing any remainder.
        updated_active = {}
        for wid, weight in active.items():
            current_total = sum(
                value for (warehouse, _), value in stock.items()
                if warehouse == wid
            )
            if current_total < warehouse_capacity[wid]:
                updated_active[wid] = weight
        active = updated_active

        # Prevent an infinite loop if no additional unit could be placed.
        if allocated_this_round == 0 and not ranking:
            break

    return allocations, remaining


# --------------------------------------------------
# 4. SIMULATION
# --------------------------------------------------

def run_simulation(scenario):

    env = simpy.Environment()
    rng = np.random.default_rng(42)

    stock = initial_stock.copy()

    daily_records = []
    delivery_records = []
    receipt_records = []
    capacity_event_records = []

    # Receipts are accumulated by one-based simulation day (day 1..30)
    # and warehouse-product, matching daily_operations() records.
    receipts_by_day = {}

    demand_by_day = {
        day: group
        for day, group in daily_demand.groupby("date")
    }

    def supplier_delivery(order):
        supplier_id = order.supplier_id
        product_id = order.product_id
        quantity = int(order.allocated_quantity)

        reliability = float(order.reliability)
        lead_time = int(order.lead_time_days)

        if scenario == "disruption":
            reliability = max(0.0, reliability - 0.15)
            extra_delay = int(rng.integers(2, 6))
        else:
            extra_delay = 0

        successful = rng.random() < reliability

        if not successful:
            delivery_records.append({
                "scenario": scenario,
                "supplier_id": supplier_id,
                "product_id": product_id,
                "quantity": quantity,
                "status": "FAILED",
                "delivery_day": np.nan,
                "received_quantity": 0,
                "unreceived_quantity": quantity
            })
            return

        yield env.timeout(lead_time + extra_delay)

        # Daily demand records are numbered from day 1, while SimPy's
        # env.now starts at 0. Convert the event time to the same
        # one-based day index before recording receipts.
        delivery_day = int(env.now) + 1
        allocations, unreceived = allocate_quantity_by_capacity(
            product_id, quantity, stock, warehouse_capacity
        )

        received_total = sum(amount for _, amount in allocations)

        for warehouse_id, received in allocations:
            key = (warehouse_id, product_id)
            receipts_by_day[(delivery_day, warehouse_id, product_id)] = (
                receipts_by_day.get(
                    (delivery_day, warehouse_id, product_id), 0
                ) + received
            )

            warehouse_stock_after = sum(
                value for (wid, _), value in stock.items()
                if wid == warehouse_id
            )

            receipt_records.append({
                "scenario": scenario,
                "simulation_day": delivery_day,
                "supplier_id": supplier_id,
                "product_id": product_id,
                "warehouse_id": warehouse_id,
                "received_quantity": received,
                "warehouse_stock_after_receipt": warehouse_stock_after,
                "warehouse_capacity": warehouse_capacity[warehouse_id]
            })

            capacity_event_records.append({
                "scenario": scenario,
                "simulation_day": delivery_day,
                "supplier_id": supplier_id,
                "product_id": product_id,
                "warehouse_id": warehouse_id,
                "stock_after_receipt": warehouse_stock_after,
                "capacity": warehouse_capacity[warehouse_id],
                "capacity_excess": max(
                    0, warehouse_stock_after - warehouse_capacity[warehouse_id]
                )
            })

        delivery_records.append({
            "scenario": scenario,
            "supplier_id": supplier_id,
            "product_id": product_id,
            "quantity": quantity,
            "status": "DELIVERED",
            "delivery_day": delivery_day,
            "received_quantity": received_total,
            "unreceived_quantity": unreceived
        })

    def daily_operations():
        for day_number, date in enumerate(simulation_dates):

            if day_number > 0:
                yield env.timeout(1)

            simulation_day = day_number + 1
            today = demand_by_day[date]

            for row in today.itertuples():
                warehouse_id = row.warehouse_id
                product_id = row.product_id
                key = (warehouse_id, product_id)

                demand = int(row.demand)

                if (
                    scenario == "disruption"
                    and 10 <= day_number <= 14
                ):
                    demand = int(np.ceil(demand * 1.30))

                # Receipts arriving on this simulation day are included
                # in current stock. Subtract them to derive opening stock.
                receipts_today = receipts_by_day.get(
                    (simulation_day, warehouse_id, product_id), 0
                )
                current_stock = stock.get(key, 0)
                opening_stock = current_stock - receipts_today

                fulfilled = min(current_stock, demand)
                lost_sales = demand - fulfilled
                closing_stock = current_stock - fulfilled
                stock[key] = closing_stock

                daily_records.append({
                    "scenario": scenario,
                    "day": simulation_day,
                    "date": date,
                    "warehouse_id": warehouse_id,
                    "product_id": product_id,
                    "opening_inventory": opening_stock,
                    "receipts": receipts_today,
                    "demand": demand,
                    "fulfilled": fulfilled,
                    "lost_sales": lost_sales,
                    "ending_inventory": closing_stock
                })

    # Start supplier processes before daily demand process. This makes
    # deliveries scheduled for a day available before that day's demand event.
    for order in orders.itertuples():
        env.process(supplier_delivery(order))

    env.process(daily_operations())
    env.run(until=30)

    return (
        pd.DataFrame(daily_records),
        pd.DataFrame(delivery_records),
        pd.DataFrame(receipt_records),
        pd.DataFrame(capacity_event_records)
    )


# --------------------------------------------------
# 5. RUN BOTH SCENARIOS
# --------------------------------------------------

(
    baseline_daily,
    baseline_deliveries,
    baseline_receipts,
    baseline_capacity_events
) = run_simulation("baseline")

(
    disruption_daily,
    disruption_deliveries,
    disruption_receipts,
    disruption_capacity_events
) = run_simulation("disruption")


# --------------------------------------------------
# 6. CALCULATE METRICS
# --------------------------------------------------

def calculate_metrics(daily, deliveries, scenario):
    total_demand = int(daily["demand"].sum())
    total_fulfilled = int(daily["fulfilled"].sum())
    total_lost = int(daily["lost_sales"].sum())

    service_level = (
        total_fulfilled / total_demand * 100
        if total_demand > 0 else 0
    )

    delivered = int((deliveries["status"] == "DELIVERED").sum())
    failed = int((deliveries["status"] == "FAILED").sum())

    return {
        "scenario": scenario,
        "total_demand": total_demand,
        "fulfilled_demand": total_fulfilled,
        "lost_sales": total_lost,
        "service_level_percent": service_level,
        "successful_deliveries": delivered,
        "failed_deliveries": failed,
        "total_orders": len(deliveries),
        "total_received_quantity": int(deliveries["received_quantity"].sum()),
        "total_unreceived_quantity": int(
            deliveries["unreceived_quantity"].sum()
        )
    }


comparison = pd.DataFrame([
    calculate_metrics(
        baseline_daily, baseline_deliveries, "Baseline"
    ),
    calculate_metrics(
        disruption_daily, disruption_deliveries, "Disruption"
    )
])


# --------------------------------------------------
# 7. SAVE REPORTS
# --------------------------------------------------

reports = {
    "simulation_baseline_daily.csv": baseline_daily,
    "simulation_disruption_daily.csv": disruption_daily,
    "simulation_baseline_deliveries.csv": baseline_deliveries,
    "simulation_disruption_deliveries.csv": disruption_deliveries,
    "simulation_baseline_warehouse_receipts.csv": baseline_receipts,
    "simulation_disruption_warehouse_receipts.csv": disruption_receipts,
    "simulation_baseline_capacity_events.csv": baseline_capacity_events,
    "simulation_disruption_capacity_events.csv": disruption_capacity_events,
    "simulation_scenario_comparison.csv": comparison
}

for filename, dataframe in reports.items():
    dataframe.to_csv(REPORT_DIR / filename, index=False)

print("\n========== UPDATED SIMULATION RESULTS ==========")
print(comparison.to_string(index=False))

print("\n========== INVENTORY FLOW VALIDATION ==========")
for scenario_name, daily in [
    ("Baseline", baseline_daily),
    ("Disruption", disruption_daily)
]:
    difference = (
        daily["opening_inventory"]
        + daily["receipts"]
        - daily["fulfilled"]
        - daily["ending_inventory"]
    )
    print(
        f"{scenario_name}: maximum absolute stock-flow difference = "
        f"{difference.abs().max()}"
    )

print("\n========== RECEIPT CAPACITY VALIDATION ==========")
all_capacity_events = pd.concat(
    [baseline_capacity_events, disruption_capacity_events],
    ignore_index=True
)

if all_capacity_events.empty:
    print("No warehouse receipt events occurred within the simulation horizon.")
else:
    max_excess = all_capacity_events["capacity_excess"].max()
    print("Maximum capacity excess:", max_excess)
    print(
        "Capacity violations:",
        int((all_capacity_events["capacity_excess"] > 0).sum())
    )

print("\nUpdated simulation completed successfully!")
print("Reports saved in:", REPORT_DIR)
