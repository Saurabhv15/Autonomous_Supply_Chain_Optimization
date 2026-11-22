import numpy as np
import pandas as pd

from config import REPORT_DIR

# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

VEHICLE_CAPACITY = 5000
TOLERANCE = 0.01

print("\n========== LOGISTICS INDEPENDENT AUDIT ==========")


# ============================================================
# STEP 2: LOAD GENERATED REPORTS
# ============================================================

routes = pd.read_csv(
    REPORT_DIR / "optimized_routes.csv"
)

assignments = pd.read_csv(
    REPORT_DIR / "store_warehouse_assignment.csv"
)

warehouse_summary = pd.read_csv(
    REPORT_DIR / "warehouse_logistics_summary.csv"
)

print("\nAll logistics reports loaded successfully.")


# ============================================================
# STEP 3: VALIDATE REQUIRED COLUMNS
# ============================================================

required_route_columns = {
    "warehouse_id",
    "route_id",
    "stop_sequence",
    "location_type",
    "location_id",
    "delivery_demand",
    "leg_distance_km",
    "cumulative_load"
}

required_assignment_columns = {
    "store_id",
    "warehouse_id",
    "delivery_demand"
}

required_summary_columns = {
    "warehouse_id",
    "number_of_stores",
    "total_daily_demand",
    "vehicles_used",
    "total_distance_km",
    "solver_status"
}

if not required_route_columns.issubset(routes.columns):
    raise ValueError(
        f"Missing route columns: "
        f"{required_route_columns - set(routes.columns)}"
    )

if not required_assignment_columns.issubset(assignments.columns):
    raise ValueError(
        f"Missing assignment columns: "
        f"{required_assignment_columns - set(assignments.columns)}"
    )

if not required_summary_columns.issubset(warehouse_summary.columns):
    raise ValueError(
        f"Missing summary columns: "
        f"{required_summary_columns - set(warehouse_summary.columns)}"
    )

checks = []


def record_check(name, passed, details):
    checks.append({
        "check_name": name,
        "status": "PASS" if passed else "FAIL",
        "details": str(details)
    })


# ============================================================
# STEP 4: CHECK STORE ASSIGNMENT
# ============================================================

duplicate_assignments = assignments["store_id"].duplicated().sum()

record_check(
    "No duplicate store assignments",
    duplicate_assignments == 0,
    f"Duplicate assignments: {duplicate_assignments}"
)

assigned_store_ids = set(assignments["store_id"])

# Extract stores from optimized routes.
route_stores = routes[
    routes["location_type"] == "store"
].copy()

duplicate_route_visits = (
    route_stores["location_id"].duplicated().sum()
)

record_check(
    "No duplicate store visits",
    duplicate_route_visits == 0,
    f"Duplicate route visits: {duplicate_route_visits}"
)

visited_store_ids = set(route_stores["location_id"])

missing_stores = assigned_store_ids - visited_store_ids
unexpected_stores = visited_store_ids - assigned_store_ids

record_check(
    "All assigned stores are visited",
    len(missing_stores) == 0,
    f"Missing stores: {sorted(missing_stores)}"
)

record_check(
    "No unexpected stores in routes",
    len(unexpected_stores) == 0,
    f"Unexpected stores: {sorted(unexpected_stores)}"
)

record_check(
    "Every assigned store visited exactly once",
    len(assigned_store_ids) == len(visited_store_ids)
    and duplicate_route_visits == 0,
    f"Assigned: {len(assigned_store_ids)}, "
    f"Visited: {len(visited_store_ids)}"
)


# ============================================================
# STEP 5: CHECK ROUTE START AND END
# ============================================================

route_checks = []

for route_id, group in routes.groupby("route_id"):

    group = group.sort_values("stop_sequence")

    warehouse_id = group["warehouse_id"].iloc[0]

    first_stop = group.iloc[0]
    last_stop = group.iloc[-1]

    starts_at_depot = (
        first_stop["location_type"] == "warehouse"
        and first_stop["location_id"] == warehouse_id
    )

    ends_at_depot = (
        last_stop["location_type"] == "warehouse"
        and last_stop["location_id"] == warehouse_id
    )

    same_warehouse = (
        group["warehouse_id"].nunique() == 1
        and (
            group.loc[
                group["location_type"] == "warehouse",
                "location_id"
            ] == warehouse_id
        ).all()
    )

    route_checks.append({
        "route_id": route_id,
        "warehouse_id": warehouse_id,
        "starts_at_depot": starts_at_depot,
        "ends_at_depot": ends_at_depot,
        "same_warehouse": same_warehouse
    })

route_checks_df = pd.DataFrame(route_checks)

record_check(
    "All routes start at assigned warehouse",
    route_checks_df["starts_at_depot"].all(),
    f"Valid routes: {route_checks_df['starts_at_depot'].sum()}"
)

record_check(
    "All routes return to assigned warehouse",
    route_checks_df["ends_at_depot"].all(),
    f"Valid routes: {route_checks_df['ends_at_depot'].sum()}"
)

record_check(
    "Each route belongs to one warehouse",
    route_checks_df["same_warehouse"].all(),
    f"Valid routes: {route_checks_df['same_warehouse'].sum()}"
)


# ============================================================
# STEP 6: VEHICLE CAPACITY AUDIT
# ============================================================

vehicle_checks = []

for route_id, group in routes.groupby("route_id"):

    store_rows = group[
        group["location_type"] == "store"
    ]

    total_load = store_rows["delivery_demand"].sum()

    vehicle_checks.append({
        "route_id": route_id,
        "warehouse_id": group["warehouse_id"].iloc[0],
        "total_load": total_load,
        "vehicle_capacity": VEHICLE_CAPACITY,
        "capacity_valid": total_load <= VEHICLE_CAPACITY
    })

vehicle_checks_df = pd.DataFrame(vehicle_checks)

record_check(
    "All vehicles respect capacity",
    vehicle_checks_df["capacity_valid"].all(),
    f"Vehicles exceeding capacity: "
    f"{(~vehicle_checks_df['capacity_valid']).sum()}"
)


# ============================================================
# STEP 7: CUMULATIVE LOAD AUDIT
# ============================================================

cumulative_checks = []

for route_id, group in routes.groupby("route_id"):

    group = group.sort_values("stop_sequence")

    expected_load = group["delivery_demand"].cumsum()

    actual_load = group["cumulative_load"]

    valid = np.allclose(
        expected_load,
        actual_load,
        atol=TOLERANCE
    )

    cumulative_checks.append({
        "route_id": route_id,
        "cumulative_load_valid": valid
    })

cumulative_checks_df = pd.DataFrame(cumulative_checks)

record_check(
    "Cumulative vehicle loads are correct",
    cumulative_checks_df["cumulative_load_valid"].all(),
    f"Invalid routes: "
    f"{(~cumulative_checks_df['cumulative_load_valid']).sum()}"
)


# ============================================================
# STEP 8: STORE DEMAND CONSISTENCY
# ============================================================

assignment_demand = (
    assignments
    .groupby("store_id")["delivery_demand"]
    .sum()
)

route_demand = (
    route_stores
    .groupby("location_id")["delivery_demand"]
    .sum()
)

demand_comparison = pd.concat(
    [
        assignment_demand.rename("assigned_demand"),
        route_demand.rename("routed_demand")
    ],
    axis=1
).fillna(0)

demand_comparison["difference"] = (
    demand_comparison["assigned_demand"]
    - demand_comparison["routed_demand"]
)

record_check(
    "Store demand matches route demand",
    np.allclose(
        demand_comparison["assigned_demand"],
        demand_comparison["routed_demand"],
        atol=TOLERANCE
    ),
    f"Demand mismatches: "
    f"{(demand_comparison['difference'].abs() > TOLERANCE).sum()}"
)


# ============================================================
# STEP 9: WAREHOUSE ASSIGNMENT CONSISTENCY
# ============================================================

assignment_map = assignments.set_index(
    "store_id"
)["warehouse_id"].to_dict()

route_assignment_errors = []

for _, row in route_stores.iterrows():

    store_id = row["location_id"]
    route_warehouse = row["warehouse_id"]

    expected_warehouse = assignment_map.get(store_id)

    if route_warehouse != expected_warehouse:
        route_assignment_errors.append(store_id)

record_check(
    "Store routes match warehouse assignments",
    len(route_assignment_errors) == 0,
    f"Mismatched stores: {route_assignment_errors}"
)


# ============================================================
# STEP 10: ROUTE DISTANCE AUDIT
# ============================================================

route_distance = (
    routes.groupby("warehouse_id")["leg_distance_km"]
    .sum()
)

summary_distance = (
    warehouse_summary.set_index("warehouse_id")
    ["total_distance_km"]
)

distance_comparison = pd.concat(
    [
        route_distance.rename("route_distance"),
        summary_distance.rename("summary_distance")
    ],
    axis=1
).fillna(0)

distance_comparison["difference"] = (
    distance_comparison["route_distance"]
    - distance_comparison["summary_distance"]
)

distance_valid = np.allclose(
    distance_comparison["route_distance"],
    distance_comparison["summary_distance"],
    atol=TOLERANCE
)

record_check(
    "Warehouse route distances match summary",
    distance_valid,
    f"Maximum distance difference: "
    f"{distance_comparison['difference'].abs().max():.6f} km"
)


# ============================================================
# STEP 11: WAREHOUSE DEMAND AUDIT
# ============================================================

assignment_warehouse_demand = (
    assignments.groupby("warehouse_id")["delivery_demand"]
    .sum()
)

summary_warehouse_demand = (
    warehouse_summary.set_index("warehouse_id")
    ["total_daily_demand"]
)

warehouse_demand_comparison = pd.concat(
    [
        assignment_warehouse_demand.rename("assigned_demand"),
        summary_warehouse_demand.rename("summary_demand")
    ],
    axis=1
).fillna(0)

warehouse_demand_valid = np.allclose(
    warehouse_demand_comparison["assigned_demand"],
    warehouse_demand_comparison["summary_demand"],
    atol=TOLERANCE
)

record_check(
    "Warehouse demand matches assigned stores",
    warehouse_demand_valid,
    f"Mismatched warehouses: "
    f"{(
        warehouse_demand_comparison["assigned_demand"]
        - warehouse_demand_comparison["summary_demand"]
    ).abs().gt(TOLERANCE).sum()}"
)


# ============================================================
# STEP 12: VEHICLE COUNT AUDIT
# ============================================================

actual_vehicle_count = routes["route_id"].nunique()

reported_vehicle_count = warehouse_summary[
    "vehicles_used"
].sum()

record_check(
    "Vehicle count matches warehouse summary",
    actual_vehicle_count == reported_vehicle_count,
    f"Routes: {actual_vehicle_count}, "
    f"Reported vehicles: {reported_vehicle_count}"
)


# ============================================================
# STEP 13: SAVE AUDIT REPORTS
# ============================================================

audit_df = pd.DataFrame(checks)

audit_df.to_csv(
    REPORT_DIR / "audit_logistics_checks.csv",
    index=False
)

vehicle_checks_df.to_csv(
    REPORT_DIR / "audit_vehicle_capacity.csv",
    index=False
)

route_checks_df.to_csv(
    REPORT_DIR / "audit_route_checks.csv",
    index=False
)

demand_comparison.to_csv(
    REPORT_DIR / "audit_store_demand.csv"
)

distance_comparison.to_csv(
    REPORT_DIR / "audit_distance_comparison.csv"
)

warehouse_demand_comparison.to_csv(
    REPORT_DIR / "audit_warehouse_demand.csv"
)


# ============================================================
# STEP 14: FINAL AUDIT SUMMARY
# ============================================================

passed = (audit_df["status"] == "PASS").sum()
failed = (audit_df["status"] == "FAIL").sum()

print("\n========== AUDIT RESULTS ==========")

print(audit_df.to_string(index=False))

print("\n========== FINAL AUDIT SUMMARY ==========")

print("Total checks:", len(audit_df))
print("Passed:", passed)
print("Failed:", failed)

if failed == 0:
    print("\nALL LOGISTICS AUDIT CHECKS PASSED!")
else:
    print("\nSOME LOGISTICS AUDIT CHECKS FAILED.")
    print("Please inspect the generated audit CSV files.")

print("\nAudit reports saved successfully.")
