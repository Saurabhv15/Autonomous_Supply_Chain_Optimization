"""
Rule-Based Supply Chain Decision Agent
--------------------------------------
Reads the already generated optimization and simulation reports,
creates explainable recommendations, and saves them under reports/.

This is the first agent layer. It does not place orders or change
inventory. It only analyses existing reports and recommends actions.
"""

from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import REPORT_DIR


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

SERVICE_LEVEL_TARGET = 95.0  # Business target assumption; change if required.
TOP_PRODUCTS_TO_REPORT = 20

REQUIREMENTS_FILE = REPORT_DIR / "replenishment_requirements_v5.csv"
SUPPLIER_ALLOCATION_FILE = REPORT_DIR / "supplier_allocation_v5.csv"
SIMULATION_FILE = REPORT_DIR / "simulation_scenario_comparison.csv"

OUTPUT_RECOMMENDATIONS = REPORT_DIR / "agent_recommendations.csv"
OUTPUT_PRODUCT_PRIORITIES = REPORT_DIR / "agent_product_priorities.csv"
OUTPUT_SUPPLIER_RISK = REPORT_DIR / "agent_supplier_risk.csv"
OUTPUT_SUMMARY = REPORT_DIR / "agent_decision_summary.txt"


# --------------------------------------------------
# 2. LOAD AND VALIDATE REPORTS
# --------------------------------------------------

def load_csv(path: Path, required_columns: set[str], report_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{report_name} not found: {path}\n"
            "Run the relevant optimization/simulation script first."
        )

    dataframe = pd.read_csv(path)
    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{report_name} is missing columns: {sorted(missing)}\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    return dataframe


requirements = load_csv(
    REQUIREMENTS_FILE,
    {"product_id", "planned_quantity", "policy_allocation", "priority_score"},
    "V5 replenishment requirements"
)

supplier_allocations = load_csv(
    SUPPLIER_ALLOCATION_FILE,
    {
        "product_id",
        "supplier_id",
        "allocated_quantity",
        "unit_cost",
        "risk_multiplier",
        "risk_adjusted_unit_cost",
    },
    "V5 supplier allocation"
)

simulation = load_csv(
    SIMULATION_FILE,
    {
        "scenario",
        "total_demand",
        "lost_sales",
        "service_level_percent",
    },
    "Simulation scenario comparison"
)


# --------------------------------------------------
# 3. PREPARE PRODUCT-LEVEL PRIORITIES
# --------------------------------------------------

numeric_columns = [
    "planned_quantity",
    "policy_allocation",
    "priority_score",
]

for column in numeric_columns:
    requirements[column] = pd.to_numeric(
        requirements[column], errors="coerce"
    )

if requirements[numeric_columns].isna().any().any():
    raise ValueError("Invalid numeric values found in replenishment requirements.")

if (requirements[["planned_quantity", "policy_allocation"]] < 0).any().any():
    raise ValueError("Planned quantity and policy allocation cannot be negative.")

requirements["shortage_quantity"] = (
    requirements["planned_quantity"] - requirements["policy_allocation"]
).clip(lower=0)

requirements["product_fill_rate_percent"] = (
    requirements["policy_allocation"]
    .div(requirements["planned_quantity"].replace(0, pd.NA))
    .fillna(0)
    * 100
)

product_priorities = requirements[
    [
        "product_id",
        "planned_quantity",
        "policy_allocation",
        "shortage_quantity",
        "priority_score",
        "product_fill_rate_percent",
    ]
].copy()

product_priorities = product_priorities.sort_values(
    ["priority_score", "shortage_quantity"],
    ascending=[False, False],
).reset_index(drop=True)

product_priorities["priority_rank"] = (
    product_priorities.index + 1
)

product_priorities.to_csv(OUTPUT_PRODUCT_PRIORITIES, index=False)


# --------------------------------------------------
# 4. ANALYSE PROCUREMENT CAPACITY
# --------------------------------------------------

total_planned = float(requirements["planned_quantity"].sum())
total_allocated = float(requirements["policy_allocation"].sum())
total_shortage = float(requirements["shortage_quantity"].sum())

procurement_fill_rate = (
    total_allocated / total_planned * 100
    if total_planned > 0 else 0
)

recommendations = []


def add_recommendation(priority, area, finding, recommended_action, evidence):
    recommendations.append({
        "priority": priority,
        "area": area,
        "finding": finding,
        "recommended_action": recommended_action,
        "evidence": evidence,
    })


if total_shortage > 0:
    add_recommendation(
        "CRITICAL",
        "Procurement capacity",
        "Available supplier allocation is below the planned replenishment requirement.",
        (
            "Review additional supplier capacity, qualify alternate suppliers, "
            "and phase procurement by product priority tier. Re-run supplier "
            "allocation after capacity is updated."
        ),
        (
            f"Planned={total_planned:,.0f}; allocated={total_allocated:,.0f}; "
            f"shortage={total_shortage:,.0f}; fill rate={procurement_fill_rate:.2f}%."
        ),
    )

zero_allocated = product_priorities[
    (product_priorities["planned_quantity"] > 0)
    & (product_priorities["policy_allocation"] == 0)
]

if not zero_allocated.empty:
    sample_products = ", ".join(
        zero_allocated["product_id"].astype(str).head(10)
    )
    add_recommendation(
        "HIGH",
        "Product allocation",
        f"{len(zero_allocated)} products have a positive requirement but zero allocation.",
        (
            "Review the zero-allocation products against business criticality, "
            "customer impact and substitution options. Consider a minimum "
            "allocation rule if operationally justified."
        ),
        f"Examples: {sample_products}.",
    )


# --------------------------------------------------
# 5. ANALYSE SIMULATION SCENARIOS
# --------------------------------------------------

simulation["scenario_normalized"] = (
    simulation["scenario"].astype(str).str.strip().str.lower()
)

baseline_rows = simulation[
    simulation["scenario_normalized"] == "baseline"
]
disruption_rows = simulation[
    simulation["scenario_normalized"] == "disruption"
]

if baseline_rows.empty or disruption_rows.empty:
    raise ValueError(
        "Simulation report must contain 'Baseline' and 'Disruption' scenarios."
    )

baseline = baseline_rows.iloc[0]
disruption = disruption_rows.iloc[0]

baseline_service = float(baseline["service_level_percent"])
disruption_service = float(disruption["service_level_percent"])
service_level_drop = baseline_service - disruption_service

if baseline_service < SERVICE_LEVEL_TARGET:
    add_recommendation(
        "CRITICAL",
        "Customer service",
        "Baseline service level is below the configured target.",
        (
            "Investigate the replenishment-capacity constraint and increase "
            "available supply before treating the service-level target as achievable."
        ),
        (
            f"Baseline service level={baseline_service:.2f}%; "
            f"configured target={SERVICE_LEVEL_TARGET:.2f}%."
        ),
    )

if disruption_service < SERVICE_LEVEL_TARGET:
    add_recommendation(
        "HIGH",
        "Disruption preparedness",
        "Disruption scenario service level is below the configured target.",
        (
            "Prepare contingency sourcing and review safety-stock coverage "
            "for critical products. Test alternate lead-time and reliability assumptions."
        ),
        (
            f"Disruption service level={disruption_service:.2f}%; "
            f"configured target={SERVICE_LEVEL_TARGET:.2f}%."
        ),
    )

if service_level_drop > 0:
    add_recommendation(
        "HIGH",
        "Supplier resilience",
        "The disruption scenario reduces the network service level.",
        (
            "Evaluate backup suppliers and supplier diversification. "
            "Compare the cost of resilience against the service-level benefit "
            "in a separate scenario analysis."
        ),
        (
            f"Baseline={baseline_service:.2f}%; disruption={disruption_service:.2f}%; "
            f"drop={service_level_drop:.2f} percentage points."
        ),
    )

if float(disruption["lost_sales"]) > 0:
    add_recommendation(
        "HIGH",
        "Lost sales",
        "The disruption simulation records unfulfilled demand.",
        (
            "Use the product-priority report to focus scarce inventory on "
            "high-priority items, and test whether alternate sourcing reduces lost sales."
        ),
        f"Disruption lost sales={float(disruption['lost_sales']):,.0f} units.",
    )


# --------------------------------------------------
# 6. ANALYSE SUPPLIER ALLOCATION RISK
# --------------------------------------------------

for column in [
    "allocated_quantity",
    "unit_cost",
    "risk_multiplier",
    "risk_adjusted_unit_cost",
]:
    supplier_allocations[column] = pd.to_numeric(
        supplier_allocations[column], errors="coerce"
    )

if supplier_allocations[
    ["allocated_quantity", "unit_cost", "risk_multiplier", "risk_adjusted_unit_cost"]
].isna().any().any():
    raise ValueError("Invalid numeric values found in supplier allocation report.")

supplier_allocations["base_procurement_value"] = (
    supplier_allocations["allocated_quantity"] * supplier_allocations["unit_cost"]
)
supplier_allocations["risk_adjusted_value"] = (
    supplier_allocations["allocated_quantity"]
    * supplier_allocations["risk_adjusted_unit_cost"]
)

supplier_risk = (
    supplier_allocations.groupby("supplier_id", as_index=False)
    .agg(
        allocated_quantity=("allocated_quantity", "sum"),
        base_procurement_value=("base_procurement_value", "sum"),
        risk_adjusted_value=("risk_adjusted_value", "sum"),
    )
)

supplier_risk["risk_premium"] = (
    supplier_risk["risk_adjusted_value"]
    - supplier_risk["base_procurement_value"]
)

supplier_risk["weighted_risk_multiplier"] = (
    supplier_allocations.assign(
        weighted_risk=(
            supplier_allocations["allocated_quantity"]
            * supplier_allocations["risk_multiplier"]
        )
    )
    .groupby("supplier_id")["weighted_risk"].sum()
    .div(
        supplier_allocations.groupby("supplier_id")["allocated_quantity"].sum()
        .replace(0, pd.NA)
    )
    .reindex(supplier_risk["supplier_id"])
    .fillna(0)
    .to_numpy()
)

supplier_risk.to_csv(OUTPUT_SUPPLIER_RISK, index=False)

if not supplier_risk.empty:
    highest_risk_supplier = supplier_risk.sort_values(
        "weighted_risk_multiplier", ascending=False
    ).iloc[0]

    add_recommendation(
        "MEDIUM",
        "Supplier risk",
        (
            f"{highest_risk_supplier['supplier_id']} has the highest "
            "quantity-weighted risk multiplier among allocated suppliers."
        ),
        (
            "Review this supplier's reliability and lead-time assumptions. "
            "Assess whether some future volume should be shifted to qualified alternatives."
        ),
        (
            f"Weighted risk multiplier="
            f"{highest_risk_supplier['weighted_risk_multiplier']:.4f}; "
            f"allocated quantity={highest_risk_supplier['allocated_quantity']:,.0f}."
        ),
    )


# --------------------------------------------------
# 7. SAVE EXPLAINABLE RECOMMENDATIONS
# --------------------------------------------------

recommendation_df = pd.DataFrame(recommendations)

priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
if not recommendation_df.empty:
    recommendation_df["_sort"] = recommendation_df["priority"].map(priority_order)
    recommendation_df = (
        recommendation_df.sort_values(["_sort", "area"])
        .drop(columns="_sort")
        .reset_index(drop=True)
    )

recommendation_df.to_csv(OUTPUT_RECOMMENDATIONS, index=False)

summary_lines = [
    "SUPPLY CHAIN RULE-BASED DECISION AGENT",
    "=" * 48,
    "",
    "This agent analyses existing project reports and generates recommendations.",
    "It does not execute procurement, routing or inventory changes.",
    "",
    "PROCUREMENT POSITION",
    f"Planned replenishment: {total_planned:,.0f} units",
    f"Allocated quantity:   {total_allocated:,.0f} units",
    f"Unallocated shortage: {total_shortage:,.0f} units",
    f"Procurement fill rate: {procurement_fill_rate:.2f}%",
    "",
    "SIMULATION SERVICE LEVEL",
    f"Baseline:   {baseline_service:.2f}%",
    f"Disruption: {disruption_service:.2f}%",
    f"Service-level drop: {service_level_drop:.2f} percentage points",
    f"Configured service target: {SERVICE_LEVEL_TARGET:.2f}%",
    "",
    f"Products prioritised: {len(product_priorities)}",
    f"Products with zero allocation: {len(zero_allocated)}",
    f"Recommendations generated: {len(recommendation_df)}",
    "",
    "RECOMMENDATIONS",
    "-" * 48,
]

for index, row in recommendation_df.iterrows():
    summary_lines.extend([
        f"{index + 1}. [{row['priority']}] {row['area']}",
        f"   Finding: {row['finding']}",
        f"   Action: {row['recommended_action']}",
        f"   Evidence: {row['evidence']}",
        "",
    ])

summary_lines.extend([
    "OUTPUT FILES",
    str(OUTPUT_RECOMMENDATIONS),
    str(OUTPUT_PRODUCT_PRIORITIES),
    str(OUTPUT_SUPPLIER_RISK),
])

OUTPUT_SUMMARY.write_text("\n".join(summary_lines), encoding="utf-8")

print("\n========== RULE-BASED AGENT RESULTS ==========")
print(f"Planned replenishment: {total_planned:,.0f}")
print(f"Allocated quantity:   {total_allocated:,.0f}")
print(f"Shortage quantity:    {total_shortage:,.0f}")
print(f"Procurement fill rate: {procurement_fill_rate:.2f}%")
print(f"Baseline service level:   {baseline_service:.2f}%")
print(f"Disruption service level: {disruption_service:.2f}%")
print(f"Recommendations generated: {len(recommendation_df)}")

if recommendation_df.empty:
    print("\nNo recommendations were triggered by the configured rules.")
else:
    print("\n========== RECOMMENDATIONS ==========")
    for row in recommendation_df.itertuples(index=False):
        print(f"\n[{row.priority}] {row.area}")
        print("Finding:", row.finding)
        print("Action:", row.recommended_action)
        print("Evidence:", row.evidence)

print("\nAgent reports saved successfully:")
print(OUTPUT_RECOMMENDATIONS)
print(OUTPUT_PRODUCT_PRIORITIES)
print(OUTPUT_SUPPLIER_RISK)
print(OUTPUT_SUMMARY)
