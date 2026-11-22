import math
import numpy as np
import pandas as pd

from ortools.constraint_solver import (
    pywrapcp,
    routing_enums_pb2
)

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORT_DIR
)


# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

RANDOM_SEED = 42

VEHICLE_CAPACITY = 5000       # Units per vehicle per trip
SOLVER_TIME_LIMIT = 5         # Seconds per warehouse

# Synthetic coordinate region.
# Used only if real coordinates are not present.
SYNTHETIC_CENTER_LAT = 22.5
SYNTHETIC_CENTER_LON = 87.3
COORDINATE_SPREAD = 0.30

np.random.seed(RANDOM_SEED)

print("\n========== LOGISTICS OPTIMIZATION ==========")


# ============================================================
# STEP 2: LOAD DATA
# ============================================================

sales = pd.read_csv(
    RAW_DATA_DIR / "sales_history.csv"
)

warehouses = pd.read_csv(
    RAW_DATA_DIR / "warehouses.csv"
)

stores = pd.read_csv(
    RAW_DATA_DIR / "stores.csv"
)

print("\nData loaded successfully.")

print("Sales records:", len(sales))
print("Warehouses:", len(warehouses))
print("Stores:", len(stores))


# ============================================================
# STEP 3: VALIDATE INPUT DATA
# ============================================================

required_sales = {
    "store_id",
    "date",
    "demand"
}

required_warehouses = {"warehouse_id"}
required_stores = {"store_id"}

if not required_sales.issubset(sales.columns):
    raise ValueError(
        f"Sales data missing columns: "
        f"{required_sales - set(sales.columns)}"
    )

if not required_warehouses.issubset(warehouses.columns):
    raise ValueError("warehouse_id missing from warehouses.csv")

if not required_stores.issubset(stores.columns):
    raise ValueError("store_id missing from stores.csv")

if warehouses["warehouse_id"].duplicated().any():
    raise ValueError("Duplicate warehouse IDs found.")

if stores["store_id"].duplicated().any():
    raise ValueError("Duplicate store IDs found.")

if sales["demand"].isna().any():
    raise ValueError("Missing sales quantities found.")

if (sales["demand"] < 0).any():
    raise ValueError("Negative sales quantities found.")

if not set(sales["store_id"]).issubset(
    set(stores["store_id"])
):
    raise ValueError("Sales contain unknown store IDs.")

if len(warehouses) == 0 or len(stores) == 0:
    raise ValueError("Warehouses and stores cannot be empty.")

print("Input validation passed.")


# ============================================================
# STEP 4: CALCULATE AVERAGE DAILY STORE DEMAND
# ============================================================

sales["date"] = pd.to_datetime(
    sales["date"]
)

daily_store_demand = (
    sales.groupby(
        ["store_id", "date"]
    )["demand"]
    .sum()
    .reset_index()
)

store_demand = (
    daily_store_demand.groupby("store_id")["demand"]
    .mean()
    .reset_index()
)

store_demand.rename(
    columns={
        "demand": "average_daily_demand"
    },
    inplace=True
)

# Round up because vehicle capacity is an integer.
store_demand["delivery_demand"] = np.ceil(
    store_demand["average_daily_demand"]
).astype(int)

stores = stores.merge(
    store_demand,
    on="store_id",
    how="left",
    validate="one_to_one"
)

stores["average_daily_demand"] = (
    stores["average_daily_demand"].fillna(0)
)

stores["delivery_demand"] = (
    stores["delivery_demand"].fillna(0).astype(int)
)

print("\nAverage daily network demand:",
      round(stores["average_daily_demand"].sum(), 2))

print("Total integer delivery demand:",
      stores["delivery_demand"].sum())


# ============================================================
# STEP 5: GENERATE OR LOAD COORDINATES
# ============================================================

def find_coordinate_columns(df):

    latitude_candidates = [
        "latitude",
        "lat"
    ]

    longitude_candidates = [
        "longitude",
        "lon",
        "lng"
    ]

    lat_col = next(
        (
            col for col in latitude_candidates
            if col in df.columns
        ),
        None
    )

    lon_col = next(
        (
            col for col in longitude_candidates
            if col in df.columns
        ),
        None
    )

    return lat_col, lon_col


warehouse_lat, warehouse_lon = find_coordinate_columns(
    warehouses
)

store_lat, store_lon = find_coordinate_columns(
    stores
)

use_real_coordinates = all([
    warehouse_lat,
    warehouse_lon,
    store_lat,
    store_lon
])

if use_real_coordinates:

    print("\nUsing coordinates from input CSV files.")

    warehouses["latitude"] = pd.to_numeric(
        warehouses[warehouse_lat],
        errors="coerce"
    )

    warehouses["longitude"] = pd.to_numeric(
        warehouses[warehouse_lon],
        errors="coerce"
    )

    stores["latitude"] = pd.to_numeric(
        stores[store_lat],
        errors="coerce"
    )

    stores["longitude"] = pd.to_numeric(
        stores[store_lon],
        errors="coerce"
    )

    coordinates = pd.concat([
        warehouses[["latitude", "longitude"]],
        stores[["latitude", "longitude"]]
    ])

    if coordinates.isna().any().any():
        raise ValueError(
            "Missing or invalid geographic coordinates."
        )

    if not (
        coordinates["latitude"].between(-90, 90).all()
        and coordinates["longitude"].between(-180, 180).all()
    ):
        raise ValueError("Invalid geographic coordinates.")

else:

    print(
        "\nReal coordinates not found."
        "\nGenerating synthetic coordinates."
    )

    rng = np.random.default_rng(RANDOM_SEED)

    warehouses["latitude"] = (
        SYNTHETIC_CENTER_LAT
        + rng.uniform(
            -COORDINATE_SPREAD,
            COORDINATE_SPREAD,
            len(warehouses)
        )
    )

    warehouses["longitude"] = (
        SYNTHETIC_CENTER_LON
        + rng.uniform(
            -COORDINATE_SPREAD,
            COORDINATE_SPREAD,
            len(warehouses)
        )
    )

    stores["latitude"] = (
        SYNTHETIC_CENTER_LAT
        + rng.uniform(
            -COORDINATE_SPREAD,
            COORDINATE_SPREAD,
            len(stores)
        )
    )

    stores["longitude"] = (
        SYNTHETIC_CENTER_LON
        + rng.uniform(
            -COORDINATE_SPREAD,
            COORDINATE_SPREAD,
            len(stores)
        )
    )

# Preserve the coordinates used by this experiment.
warehouses.to_csv(
    PROCESSED_DATA_DIR / "warehouses_logistics.csv",
    index=False
)

stores.to_csv(
    PROCESSED_DATA_DIR / "stores_logistics.csv",
    index=False
)


# ============================================================
# STEP 6: HAVERSINE DISTANCE FUNCTION
# ============================================================

def haversine_distance(lat1, lon1, lat2, lon2):

    earth_radius_km = 6371.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)

    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return earth_radius_km * c


# ============================================================
# STEP 7: ASSIGN EACH STORE TO NEAREST WAREHOUSE
# ============================================================

assignment_rows = []

for _, store in stores.iterrows():

    distances = []

    for _, warehouse in warehouses.iterrows():

        distance = haversine_distance(
            store["latitude"],
            store["longitude"],
            warehouse["latitude"],
            warehouse["longitude"]
        )

        distances.append({
            "warehouse_id": warehouse["warehouse_id"],
            "distance_km": distance
        })

    nearest = min(
        distances,
        key=lambda x: x["distance_km"]
    )

    assignment_rows.append({

        "store_id": store["store_id"],

        "warehouse_id": nearest["warehouse_id"],

        "distance_km": nearest["distance_km"],

        "average_daily_demand": (
            store["average_daily_demand"]
        ),

        "delivery_demand": store["delivery_demand"]

    })


store_assignment = pd.DataFrame(
    assignment_rows
)

store_assignment.to_csv(
    REPORT_DIR / "store_warehouse_assignment.csv",
    index=False
)

print("\nStore-to-warehouse assignment completed.")

print(
    store_assignment[
        "warehouse_id"
    ].value_counts().sort_index()
)


# ============================================================
# STEP 8: DISTANCE MATRIX
# ============================================================

def create_distance_matrix(locations):

    size = len(locations)

    matrix = [
        [0] * size
        for _ in range(size)
    ]

    for i in range(size):

        for j in range(size):

            if i == j:
                continue

            distance = haversine_distance(
                locations[i]["latitude"],
                locations[i]["longitude"],
                locations[j]["latitude"],
                locations[j]["longitude"]
            )

            # Convert kilometres into integer metres.
            matrix[i][j] = int(
                round(distance * 1000)
            )

    return matrix


# ============================================================
# STEP 9: SOLVE CVRP FOR EACH WAREHOUSE
# ============================================================

all_route_rows = []
warehouse_summary_rows = []

for _, warehouse in warehouses.iterrows():

    warehouse_id = warehouse["warehouse_id"]

    assigned_stores = stores[
        stores["store_id"].isin(
            store_assignment.loc[
                store_assignment["warehouse_id"] == warehouse_id,
                "store_id"
            ]
        )
    ].copy()

    if assigned_stores.empty:

        warehouse_summary_rows.append({

            "warehouse_id": warehouse_id,
            "number_of_stores": 0,
            "total_daily_demand": 0,
            "vehicles_required": 0,
            "vehicles_used": 0,
            "total_distance_km": 0,
            "solver_status": "NO_ASSIGNED_STORES"

        })

        continue

    assigned_stores = assigned_stores.sort_values(
        "store_id"
    ).reset_index(drop=True)

    demands = assigned_stores[
        "delivery_demand"
    ].astype(int).tolist()

    if any(
        demand > VEHICLE_CAPACITY
        for demand in demands
    ):
        raise ValueError(
            f"A store demand exceeds vehicle capacity "
            f"for warehouse {warehouse_id}."
        )

    total_demand = sum(demands)

    number_of_vehicles = max(
        1,
        math.ceil(
            total_demand / VEHICLE_CAPACITY
        )
    )

    # Node 0 is the warehouse.
    locations = [{
        "location_type": "warehouse",
        "location_id": warehouse_id,
        "latitude": warehouse["latitude"],
        "longitude": warehouse["longitude"],
        "demand": 0
    }]

    for _, store in assigned_stores.iterrows():

        locations.append({

            "location_type": "store",

            "location_id": store["store_id"],

            "latitude": store["latitude"],

            "longitude": store["longitude"],

            "demand": int(store["delivery_demand"])

        })

    distance_matrix = create_distance_matrix(
        locations
    )

    # --------------------------------------------------------
    # OR-TOOLS ROUTING MODEL
    # --------------------------------------------------------

    manager = pywrapcp.RoutingIndexManager(
        len(distance_matrix),
        number_of_vehicles,
        0
    )

    routing = pywrapcp.RoutingModel(
        manager
    )

    def distance_callback(from_index, to_index):

        from_node = manager.IndexToNode(
            from_index
        )

        to_node = manager.IndexToNode(
            to_index
        )

        return distance_matrix[
            from_node
        ][
            to_node
        ]

    distance_callback_index = (
        routing.RegisterTransitCallback(
            distance_callback
        )
    )

    routing.SetArcCostEvaluatorOfAllVehicles(
        distance_callback_index
    )

    def demand_callback(from_index):

        node = manager.IndexToNode(
            from_index
        )

        return locations[node]["demand"]

    demand_callback_index = (
        routing.RegisterUnaryTransitCallback(
            demand_callback
        )
    )

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [VEHICLE_CAPACITY] * number_of_vehicles,
        True,
        "Capacity"
    )

    # --------------------------------------------------------
    # SEARCH STRATEGY
    # --------------------------------------------------------

    search_parameters = (
        pywrapcp.DefaultRoutingSearchParameters()
    )

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy
        .PATH_CHEAPEST_ARC
    )

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic
        .GUIDED_LOCAL_SEARCH
    )

    search_parameters.time_limit.seconds = (
        SOLVER_TIME_LIMIT
    )

    solution = routing.SolveWithParameters(
        search_parameters
    )

    if solution is None:

        warehouse_summary_rows.append({

            "warehouse_id": warehouse_id,

            "number_of_stores": len(assigned_stores),

            "total_daily_demand": total_demand,

            "vehicles_required": number_of_vehicles,

            "vehicles_used": 0,

            "total_distance_km": np.nan,

            "solver_status": "NO_SOLUTION"

        })

        continue

    # --------------------------------------------------------
    # EXTRACT OPTIMIZED ROUTES
    # --------------------------------------------------------

    warehouse_distance = 0
    vehicles_used = 0

    for vehicle_id in range(number_of_vehicles):

        index = routing.Start(
            vehicle_id
        )

        route_nodes = []

        while not routing.IsEnd(index):

            node = manager.IndexToNode(index)

            route_nodes.append(node)

            index = solution.Value(
                routing.NextVar(index)
            )

        # Ignore unused vehicles.
        if len(route_nodes) <= 1:
            continue

        vehicles_used += 1

        route_id = (
            f"{warehouse_id}_V{vehicle_id + 1}"
        )

        previous_node = None
        route_distance = 0
        cumulative_load = 0

        for sequence, node in enumerate(route_nodes):

            current_location = locations[node]

            if previous_node is None:

                leg_distance = 0

            else:

                leg_distance = (
                    distance_matrix[
                        previous_node
                    ][
                        node
                    ] / 1000
                )

            route_distance += leg_distance

            cumulative_load += (
                current_location["demand"]
            )

            all_route_rows.append({

                "warehouse_id": warehouse_id,

                "route_id": route_id,

                "vehicle_id": vehicle_id + 1,

                "stop_sequence": sequence,

                "location_type": (
                    current_location["location_type"]
                ),

                "location_id": (
                    current_location["location_id"]
                ),

                "delivery_demand": (
                    current_location["demand"]
                ),

                "leg_distance_km": leg_distance,

                "cumulative_load": cumulative_load

            })

            previous_node = node

        # Return to warehouse.
        end_node = manager.IndexToNode(
            routing.End(vehicle_id)
        )

        return_distance = (
            distance_matrix[
                previous_node
            ][
                end_node
            ] / 1000
        )

        route_distance += return_distance

        cumulative_load += (
            locations[end_node]["demand"]
        )

        all_route_rows.append({

            "warehouse_id": warehouse_id,

            "route_id": route_id,

            "vehicle_id": vehicle_id + 1,

            "stop_sequence": len(route_nodes),

            "location_type": "warehouse",

            "location_id": warehouse_id,

            "delivery_demand": 0,

            "leg_distance_km": return_distance,

            "cumulative_load": cumulative_load

        })

        warehouse_distance += route_distance

    warehouse_summary_rows.append({

        "warehouse_id": warehouse_id,

        "number_of_stores": len(assigned_stores),

        "total_daily_demand": total_demand,

        "vehicles_required": number_of_vehicles,

        "vehicles_used": vehicles_used,

        "total_distance_km": warehouse_distance,

        "solver_status": "SOLUTION_FOUND"

    })

    print(
        f"{warehouse_id}: "
        f"{len(assigned_stores)} stores | "
        f"{vehicles_used} vehicles | "
        f"{warehouse_distance:.2f} km"
    )


# ============================================================
# STEP 10: SAVE OPTIMIZED ROUTES
# ============================================================

routes_df = pd.DataFrame(
    all_route_rows
)

warehouse_summary = pd.DataFrame(
    warehouse_summary_rows
)

routes_df.to_csv(
    REPORT_DIR / "optimized_routes.csv",
    index=False
)

warehouse_summary.to_csv(
    REPORT_DIR / "warehouse_logistics_summary.csv",
    index=False
)

# ============================================================
# STEP 11: NETWORK SUMMARY
# ============================================================

successful_warehouses = warehouse_summary[
    warehouse_summary["solver_status"] == "SOLUTION_FOUND"
]

failed_warehouses = warehouse_summary[
    warehouse_summary["solver_status"] == "NO_SOLUTION"
]

warehouses_without_stores = warehouse_summary[
    warehouse_summary["solver_status"] == "NO_ASSIGNED_STORES"
]

total_distance = successful_warehouses[
    "total_distance_km"
].sum()

total_vehicles = successful_warehouses[
    "vehicles_used"
].sum()

total_stores_assigned = successful_warehouses[
    "number_of_stores"
].sum()

total_stores = len(stores)

network_summary = pd.DataFrame([{

    "total_warehouses": len(warehouses),

    "total_stores": total_stores,

    "stores_assigned": total_stores_assigned,

    "total_daily_demand": stores[
        "delivery_demand"
    ].sum(),

    "vehicle_capacity": VEHICLE_CAPACITY,

    "total_vehicles_used": total_vehicles,

    "total_route_distance_km": total_distance,

    "successful_warehouses": len(successful_warehouses),

    "failed_warehouses": len(failed_warehouses),

    "warehouses_without_stores": len(
        warehouses_without_stores
    )

}])

network_summary.to_csv(
    REPORT_DIR / "logistics_optimization_summary.csv",
    index=False
)

# ============================================================
# STEP 12: FINAL OUTPUT
# ============================================================

print("\n========== LOGISTICS SUMMARY ==========")

print(
    network_summary.to_string(index=False)
)

print("\n========== WAREHOUSE SUMMARY ==========")

print(
    warehouse_summary.to_string(index=False)
)

print("\nSaved reports:")

print("1. store_warehouse_assignment.csv")
print("2. optimized_routes.csv")
print("3. warehouse_logistics_summary.csv")
print("4. logistics_optimization_summary.csv")

print("\nLOGISTICS OPTIMIZATION COMPLETED!")
