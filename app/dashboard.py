
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Supply Chain Intelligence Platform",
    page_icon="📦",
    layout="wide"
)

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports"


# --------------------------------------------------
# 2. DATA LOADING FUNCTION
# --------------------------------------------------

@st.cache_data
def load_csv(file_path):
    """Load CSV data and cache it for faster dashboard performance."""
    if not file_path.exists():
        return None

    return pd.read_csv(file_path)


# Load master data
products = load_csv(DATA_DIR / "products.csv")
suppliers = load_csv(DATA_DIR / "suppliers.csv")
warehouses = load_csv(DATA_DIR / "warehouses.csv")
stores = load_csv(DATA_DIR / "stores.csv")
sales = load_csv(DATA_DIR / "sales_history.csv")

# Load optimization and AI reports
forecast_results = load_csv(REPORT_DIR / "forecasting_results.csv")
procurement = load_csv(REPORT_DIR / "optimization_summary_v5.csv")
supplier_summary = load_csv(REPORT_DIR / "supplier_summary_v5.csv")
simulation = load_csv(REPORT_DIR / "simulation_scenario_comparison.csv")
ai_recommendations = load_csv(
    REPORT_DIR / "llm_agent_recommendations_critic_reviewed.csv"
)

# Inventory optimization reports
inventory_requirements = load_csv(
    REPORT_DIR / "replenishment_requirements_v5.csv"
)

procurement_shortage = load_csv(
    REPORT_DIR / "procurement_shortage_v5.csv"
)

# Logistics optimization reports
store_assignments = load_csv(
    REPORT_DIR / "store_warehouse_assignment.csv"
)

optimized_routes = load_csv(
    REPORT_DIR / "optimized_routes.csv"
)

warehouse_logistics = load_csv(
    REPORT_DIR / "warehouse_logistics_summary.csv"
)

logistics_summary = load_csv(
    REPORT_DIR / "logistics_optimization_summary.csv"
)

# --------------------------------------------------
# 3. CHECK REQUIRED FILES
# --------------------------------------------------

required_files = {
    "Products": products,
    "Suppliers": suppliers,
    "Warehouses": warehouses,
    "Stores": stores,
    "Sales History": sales,
    "Forecasting Results": forecast_results,
    "Procurement Summary": procurement,
    "Supplier Summary": supplier_summary,
    "Simulation Comparison": simulation,
    "AI Recommendations": ai_recommendations,
    "Inventory Requirements": inventory_requirements,
    "Procurement Shortage": procurement_shortage,
    "Store Assignments": store_assignments,
    "Optimized Routes": optimized_routes,
    "Warehouse Logistics": warehouse_logistics,
    "Logistics Summary": logistics_summary
}

missing_files = [
    name for name, data in required_files.items()
    if data is None
]

if missing_files:
    st.error(
        "Some required CSV files are missing: "
        + ", ".join(missing_files)
    )
    st.stop()


# --------------------------------------------------
# 4. DASHBOARD HEADER
# --------------------------------------------------

st.title("📦 Autonomous Supply Chain Intelligence Platform")

st.caption(
    "Integrated demand forecasting, inventory optimization, "
    "supplier management, logistics and AI-powered decision support."
)

st.divider()


# --------------------------------------------------
# 5. SIDEBAR
# --------------------------------------------------

st.sidebar.title("Navigation")

selected_page = st.sidebar.radio(
    "Select Dashboard Section",
    [
        "Executive Overview",
        "Demand Forecasting",
        "Inventory Optimization",
        "Supplier Management",
        "Logistics Control Tower",
        "Disruption Analysis",
        "AI Decision Center"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Autonomous Supply Chain Optimization Project"
)


# --------------------------------------------------
# 6. EXECUTIVE OVERVIEW
# --------------------------------------------------

if selected_page == "Executive Overview":

    st.header("Executive Overview")

    # Calculate actual KPIs
    total_demand = sales["demand"].sum()

    total_planned = procurement.iloc[0][
        "total_planned_quantity"
    ]

    total_allocated = procurement.iloc[0][
        "total_allocated_quantity"
    ]

    total_shortage = procurement.iloc[0][
        "total_shortage"
    ]

    fill_rate = procurement.iloc[0][
        "network_fill_rate_percent"
    ]

    baseline_service = simulation.loc[
        simulation["scenario"] == "Baseline",
        "service_level_percent"
    ].iloc[0]

    disruption_service = simulation.loc[
        simulation["scenario"] == "Disruption",
        "service_level_percent"
    ].iloc[0]

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Products",
        f"{len(products):,}"
    )

    col2.metric(
        "Historical Demand",
        f"{total_demand:,.0f}"
    )

    col3.metric(
        "Warehouses",
        f"{len(warehouses):,}"
    )

    col4.metric(
        "Retail Stores",
        f"{len(stores):,}"
    )

    col5, col6, col7, col8 = st.columns(4)

    col5.metric(
        "Total Suppliers",
        f"{len(suppliers):,}"
    )

    col6.metric(
        "Procurement Fill Rate",
        f"{fill_rate:.2f}%"
    )

    col7.metric(
        "Procurement Shortage",
        f"{total_shortage:,.0f}"
    )

    col8.metric(
        "Supplier Allocation",
        f"{total_allocated:,.0f}"
    )

    st.divider()

    # Procurement performance
    st.subheader("Procurement Performance")

    procurement_chart = pd.DataFrame({
        "Category": [
            "Planned Quantity",
            "Allocated Quantity",
            "Shortage"
        ],
        "Quantity": [
            total_planned,
            total_allocated,
            total_shortage
        ]
    })

    fig_procurement = px.bar(
        procurement_chart,
        x="Category",
        y="Quantity",
        text_auto=".2s",
        title="Procurement Quantity Analysis",
        labels={
            "Category": "Procurement Category",
            "Quantity": "Units"
        }
    )

    st.plotly_chart(
        fig_procurement,
        use_container_width=True
    )

    st.divider()

    # Simulation performance
    st.subheader("Supply Chain Service Level")

    simulation_chart = px.bar(
        simulation,
        x="scenario",
        y="service_level_percent",
        color="scenario",
        text_auto=".2f",
        title="Baseline vs Disruption Scenario",
        labels={
            "scenario": "Scenario",
            "service_level_percent": "Service Level (%)"
        }
    )

    st.plotly_chart(
        simulation_chart,
        use_container_width=True
    )

    st.metric(
        "Service Level Change",
        f"{disruption_service - baseline_service:.2f} percentage points",
        delta_color="inverse"
    )

    st.divider()

    # Supplier utilization
    st.subheader("Supplier Capacity Utilization")

    fig_supplier = px.bar(
        supplier_summary,
        x="supplier_id",
        y="utilization_percent",
        color="supplier_id",
        text_auto=".2f",
        title="Supplier-wise Capacity Utilization",
        labels={
            "supplier_id": "Supplier",
            "utilization_percent": "Utilization (%)"
        }
    )

    st.plotly_chart(
        fig_supplier,
        use_container_width=True
    )


# --------------------------------------------------
# INVENTORY OPTIMIZATION
# --------------------------------------------------

elif selected_page == "Inventory Optimization":

    st.header("Inventory Optimization Control Tower")
    st.caption("Product-level inventory health, replenishment priorities and EOQ analysis.")

    # Interactive filters
    f1, f2 = st.columns(2)
    tiers = ["All"] + sorted(inventory_requirements["priority_tier"].dropna().unique().tolist())
    selected_tier = f1.selectbox("Priority tier", tiers)
    product_options = sorted(inventory_requirements["product_id"].unique().tolist())
    selected_products = f2.multiselect("Filter products", product_options, default=[])

    filtered_inventory = inventory_requirements.copy()
    if selected_tier != "All":
        filtered_inventory = filtered_inventory[filtered_inventory["priority_tier"] == selected_tier]
    if selected_products:
        filtered_inventory = filtered_inventory[filtered_inventory["product_id"].isin(selected_products)]

    if filtered_inventory.empty:
        st.warning("No products match the selected filters.")
        st.stop()

    # KPIs are calculated on the currently filtered product set
    total_stock = filtered_inventory["current_network_stock"].sum()
    total_planned_replenishment = filtered_inventory["planned_quantity"].sum()
    total_safety_stock = filtered_inventory["safety_stock"].sum()
    total_shortfall = filtered_inventory["stock_shortfall"].sum()
    reorder_products = (filtered_inventory["inventory_status"] == "REORDER REQUIRED").sum()
    average_coverage = filtered_inventory["days_of_inventory"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products in View", f"{len(filtered_inventory):,}")
    c2.metric("Current Network Stock", f"{total_stock:,.0f}")
    c3.metric("Planned Replenishment", f"{total_planned_replenishment:,.0f}")
    c4.metric("Products Requiring Reorder", f"{int(reorder_products):,}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Total Safety Stock", f"{total_safety_stock:,.0f}")
    c6.metric("Stock Shortfall", f"{total_shortfall:,.0f}")
    c7.metric("Average Inventory Coverage", f"{average_coverage:.2f} days")

    st.divider()
    st.subheader("Product Inventory Position")
    inventory_columns = [
        "product_id", "priority_tier", "priority_rank", "EOQ",
        "safety_stock", "reorder_point", "current_network_stock",
        "target_stock", "planned_quantity", "stock_shortfall", "days_of_inventory"
    ]
    st.dataframe(
        filtered_inventory[inventory_columns].sort_values("priority_rank"),
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.subheader("Product Drill-down")
    drill_product = st.selectbox(
        "Select a product for detailed analysis",
        sorted(filtered_inventory["product_id"].unique().tolist())
    )
    product_row = filtered_inventory.loc[
        filtered_inventory["product_id"] == drill_product
    ].iloc[0]

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Average Daily Demand", f"{product_row['average_daily_demand']:,.1f}")
    d2.metric("Current Stock", f"{product_row['current_network_stock']:,.0f}")
    d3.metric("Reorder Point", f"{product_row['reorder_point']:,.0f}")
    d4.metric("Target Stock", f"{product_row['target_stock']:,.0f}")

    stock_position = pd.DataFrame({
        "Inventory Measure": ["Current Stock", "Reorder Point", "Target Stock"],
        "Units": [product_row["current_network_stock"], product_row["reorder_point"], product_row["target_stock"]]
    })
    fig_position = px.bar(
        stock_position, x="Inventory Measure", y="Units", color="Inventory Measure",
        text_auto=".2s", title=f"{drill_product}: Current Stock vs Reorder and Target Levels"
    )
    st.plotly_chart(fig_position, use_container_width=True)

    st.write(
        f"**EOQ:** {product_row['EOQ']:,.0f} units  |  "
        f"**Safety stock:** {product_row['safety_stock']:,.0f} units  |  "
        f"**Priority:** {product_row['priority_tier']} (rank {int(product_row['priority_rank'])})"
    )

    st.divider()
    st.subheader("Products with Highest Stock Shortfall")
    top_shortages = filtered_inventory.nlargest(15, "stock_shortfall").sort_values("stock_shortfall", ascending=True)
    fig_shortage = px.bar(
        top_shortages, x="stock_shortfall", y="product_id", orientation="h",
        color="priority_tier", text_auto=".2s",
        title="Top Products Requiring Replenishment",
        labels={"stock_shortfall": "Stock Shortfall (Units)", "product_id": "Product"}
    )
    st.plotly_chart(fig_shortage, use_container_width=True)

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Annual Demand vs EOQ")
        fig_eoq = px.scatter(
            filtered_inventory, x="annual_demand", y="EOQ", size="unit_cost",
            color="priority_tier", hover_name="product_id",
            labels={"annual_demand": "Annual Demand (Units)", "EOQ": "Economic Order Quantity"}
        )
        st.plotly_chart(fig_eoq, use_container_width=True)
    with right:
        st.subheader("Inventory Coverage Distribution")
        fig_coverage = px.histogram(
            filtered_inventory, x="days_of_inventory", nbins=20,
            labels={"days_of_inventory": "Days of Inventory"}
        )
        st.plotly_chart(fig_coverage, use_container_width=True)

    # Download filtered data for analysis
    st.download_button(
        "Download filtered inventory report",
        data=filtered_inventory.to_csv(index=False).encode("utf-8"),
        file_name="filtered_inventory_analysis.csv",
        mime="text/csv"
    )

    st.info(
        "Values are read from the existing V5 replenishment report. "
        "Inventory position does not include outstanding purchase orders or backorders "
        "because those inputs are not available in the current dataset. "
        "The filters change the displayed analysis; they do not rerun the optimization model."
    )


# --------------------------------------------------
# 7. DEMAND FORECASTING
# --------------------------------------------------

elif selected_page == "Demand Forecasting":

    st.header("Demand Forecasting Analysis")

    st.subheader("Model Performance Comparison")

    st.dataframe(
        forecast_results,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        forecast_results,
        x="Model",
        y="WAPE (%)",
        color="Model",
        text_auto=".2f",
        title="Forecasting Model Comparison (Lower WAPE is Better)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    best_model = forecast_results.loc[
        forecast_results["WAPE (%)"].idxmin()
    ]

    st.success(
        f"Best model: {best_model['Model']} "
        f"with WAPE of {best_model['WAPE (%)']:.2f}%"
    )


# --------------------------------------------------
# LOGISTICS CONTROL TOWER
# --------------------------------------------------

elif selected_page == "Logistics Control Tower":

    st.header("Logistics Optimization Control Tower")

    summary = logistics_summary.iloc[0]

    # Network KPIs
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Warehouses",
        int(summary["total_warehouses"])
    )

    c2.metric(
        "Stores Assigned",
        int(summary["stores_assigned"])
    )

    c3.metric(
        "Vehicles Used",
        int(summary["total_vehicles_used"])
    )

    c4.metric(
        "Total Route Distance",
        f"{summary['total_route_distance_km']:,.2f} km"
    )

    c5, c6, c7 = st.columns(3)

    c5.metric(
        "Daily Delivery Demand",
        f"{summary['total_daily_demand']:,.0f}"
    )

    c6.metric(
        "Successful Warehouses",
        int(summary["successful_warehouses"])
    )

    c7.metric(
        "Warehouses Without Stores",
        int(summary["warehouses_without_stores"])
    )

    st.divider()

    # Warehouse performance
    st.subheader("Warehouse-wise Logistics Performance")

    st.dataframe(
        warehouse_logistics,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Distance by warehouse
    st.subheader("Route Distance by Warehouse")

    fig_distance = px.bar(
        warehouse_logistics,
        x="warehouse_id",
        y="total_distance_km",
        color="solver_status",
        text_auto=".2f",
        title="Total Route Distance",
        labels={
            "warehouse_id": "Warehouse",
            "total_distance_km": "Distance (KM)"
        }
    )

    st.plotly_chart(
        fig_distance,
        use_container_width=True
    )

    st.divider()

    # Store assignments
    st.subheader("Store-to-Warehouse Assignments")

    st.dataframe(
        store_assignments.sort_values("warehouse_id"),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Route details
    st.subheader("Optimized Vehicle Routes")

    warehouse_filter = st.selectbox(
        "Select Warehouse",
        sorted(optimized_routes["warehouse_id"].unique())
    )

    filtered_routes = optimized_routes[
        optimized_routes["warehouse_id"] == warehouse_filter
    ].sort_values(
        ["route_id", "stop_sequence"]
    )

    st.dataframe(
        filtered_routes,
        use_container_width=True,
        hide_index=True
    )

    # Route distance summary
    selected_distance = warehouse_logistics.loc[
        warehouse_logistics["warehouse_id"] == warehouse_filter,
        "total_distance_km"
    ].iloc[0]

    st.metric(
        f"{warehouse_filter} Total Route Distance",
        f"{selected_distance:.2f} km"
    )

    st.info(
        "Routes are optimized using OR-Tools CVRP with fixed "
        "nearest-warehouse store assignments. Distances use "
        "straight-line Haversine calculations, not road distances. "
        "The solver produces feasible heuristic solutions, "
        "not guaranteed global optima."
    )


# --------------------------------------------------
# 8. SUPPLIER MANAGEMENT
# --------------------------------------------------

elif selected_page == "Supplier Management":

    st.header("Supplier Management")

    st.subheader("Supplier Performance")

    st.dataframe(
        supplier_summary,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        supplier_summary,
        x="supplier_id",
        y="allocated_quantity",
        color="supplier_id",
        text_auto=".2s",
        title="Supplier-wise Allocated Quantity"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig_reliability = px.bar(
        supplier_summary,
        x="supplier_id",
        y="reliability",
        color="supplier_id",
        title="Supplier Reliability"
    )

    st.plotly_chart(
        fig_reliability,
        use_container_width=True
    )


# --------------------------------------------------
# 9. DISRUPTION ANALYSIS
# --------------------------------------------------

elif selected_page == "Disruption Analysis":

    st.header("Supply Chain Disruption Analysis")

    st.subheader("Scenario Comparison")

    st.dataframe(
        simulation,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        simulation,
        x="scenario",
        y=["fulfilled_demand", "lost_sales"],
        barmode="group",
        title="Fulfilled Demand vs Lost Sales"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig_delivery = px.bar(
        simulation,
        x="scenario",
        y=["successful_deliveries", "failed_deliveries"],
        barmode="group",
        title="Delivery Performance"
    )

    st.plotly_chart(
        fig_delivery,
        use_container_width=True
    )


# --------------------------------------------------
# 10. AI DECISION CENTER
# --------------------------------------------------

elif selected_page == "AI Decision Center":

    st.header("AI-Powered Decision Support")

    st.caption(
        "Recommendations generated by the GroqCloud LLM agent "
        "and evaluated by the Critic."
    )

    st.subheader("Recommendation Summary")

    total_recommendations = len(ai_recommendations)

    pass_count = (
        ai_recommendations["critic_verdict"] == "PASS"
    ).sum()

    review_count = (
        ai_recommendations["critic_verdict"] == "REVIEW"
    ).sum()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Recommendations",
        total_recommendations
    )

    col2.metric(
        "Critic PASS",
        int(pass_count)
    )

    col3.metric(
        "Critic REVIEW",
        int(review_count)
    )

    st.divider()

    st.subheader("AI Recommendations")

    display_columns = [
        "risk_id",
        "severity",
        "title",
        "recommended_action",
        "critic_verdict",
        "critic_actionability_score",
        "human_approval_required"
    ]

    st.dataframe(
        ai_recommendations[display_columns],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Recommendation Details")

    for _, row in ai_recommendations.iterrows():

        with st.expander(
            f"{row['risk_id']} | {row['severity']} | {row['title']}"
        ):

            st.write("**Evidence:**")
            st.write(row["evidence"])

            st.write("**Recommended Action:**")
            st.write(row["recommended_action"])

            st.write("**Critic Verdict:**")
            st.write(row["critic_verdict"])

            st.write("**Critic Issues:**")
            st.write(row["critic_issues"])

            st.write("**Improved Action:**")
            st.write(row["critic_improved_action"])

            st.write("**Human Approval Required:**")
            st.write(row["human_approval_required"])


# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Supply Chain Intelligence Platform | "
    "M.Tech Operations Research Project"
)
