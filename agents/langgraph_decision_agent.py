"""
LangGraph Supply Chain Decision Workflow
========================================
A deterministic, explainable workflow built on LangGraph.

Nodes:
1. analyse_inputs
2. detect_risks
3. choose_actions
4. generate_recommendations
5. validate_decisions

This version uses existing CSV reports and does not call an LLM or
execute real procurement/inventory actions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict
import sys

import pandas as pd
from langgraph.graph import StateGraph, START, END


# --------------------------------------------------
# 1. PROJECT PATHS AND BUSINESS CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import REPORT_DIR  # noqa: E402

SERVICE_LEVEL_TARGET = 95.0  # Configurable business assumption
TOP_PRODUCTS_TO_REPORT = 20

REQUIREMENTS_FILE = REPORT_DIR / "replenishment_requirements_v5.csv"
SUPPLIER_ALLOCATION_FILE = REPORT_DIR / "supplier_allocation_v5.csv"
SIMULATION_FILE = REPORT_DIR / "simulation_scenario_comparison.csv"

OUTPUT_RECOMMENDATIONS = REPORT_DIR / "langgraph_recommendations.csv"
OUTPUT_PRODUCT_PRIORITIES = REPORT_DIR / "langgraph_product_priorities.csv"
OUTPUT_SUPPLIER_RISK = REPORT_DIR / "langgraph_supplier_risk.csv"
OUTPUT_SUMMARY = REPORT_DIR / "langgraph_decision_summary.txt"


# --------------------------------------------------
# 2. SHARED GRAPH STATE
# --------------------------------------------------

class SupplyChainState(TypedDict, total=False):
    requirements: pd.DataFrame
    supplier_allocations: pd.DataFrame
    simulation: pd.DataFrame

    procurement_metrics: dict[str, float]
    simulation_metrics: dict[str, float]
    product_priorities: pd.DataFrame
    supplier_risk: pd.DataFrame

    detected_risks: list[dict[str, Any]]
    selected_actions: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    validation_results: list[dict[str, Any]]

    status: str
    error: str


# --------------------------------------------------
# 3. COMMON HELPERS
# --------------------------------------------------

def load_report(
    path: Path,
    required_columns: set[str],
    report_name: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{report_name} not found: {path}. "
            "Run its upstream project module first."
        )

    dataframe = pd.read_csv(path)
    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{report_name} is missing columns: {sorted(missing)}"
        )

    return dataframe


def add_risk(
    risks: list[dict[str, Any]],
    risk_id: str,
    severity: str,
    area: str,
    finding: str,
    evidence: str,
) -> None:
    risks.append({
        "risk_id": risk_id,
        "severity": severity,
        "area": area,
        "finding": finding,
        "evidence": evidence,
    })


# --------------------------------------------------
# 4. NODE 1 — ANALYSE INPUT REPORTS
# --------------------------------------------------

def analyse_inputs(state: SupplyChainState) -> dict[str, Any]:
    print("\n[NODE 1] Analysing project reports...")

    requirements = load_report(
        REQUIREMENTS_FILE,
        {"product_id", "planned_quantity", "policy_allocation", "priority_score"},
        "V5 replenishment requirements",
    )

    supplier_allocations = load_report(
        SUPPLIER_ALLOCATION_FILE,
        {
            "product_id",
            "supplier_id",
            "allocated_quantity",
            "unit_cost",
            "risk_multiplier",
            "risk_adjusted_unit_cost",
        },
        "V5 supplier allocation",
    )

    simulation = load_report(
        SIMULATION_FILE,
        {"scenario", "total_demand", "lost_sales", "service_level_percent"},
        "Simulation scenario comparison",
    )

    for column in ["planned_quantity", "policy_allocation", "priority_score"]:
        requirements[column] = pd.to_numeric(
            requirements[column], errors="coerce"
        )

    if requirements[
        ["planned_quantity", "policy_allocation", "priority_score"]
    ].isna().any().any():
        raise ValueError("Invalid numeric values in replenishment requirements.")

    if (requirements[["planned_quantity", "policy_allocation"]] < 0).any().any():
        raise ValueError("Negative planned or allocated quantities detected.")

    if (
        requirements["policy_allocation"] > requirements["planned_quantity"]
    ).any():
        raise ValueError("Allocated quantity exceeds planned quantity.")

    total_planned = float(requirements["planned_quantity"].sum())
    total_allocated = float(requirements["policy_allocation"].sum())
    total_shortage = max(0.0, total_planned - total_allocated)

    procurement_metrics = {
        "planned_quantity": total_planned,
        "allocated_quantity": total_allocated,
        "shortage_quantity": total_shortage,
        "fill_rate_percent": (
            total_allocated / total_planned * 100
            if total_planned > 0 else 0.0
        ),
    }

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
            "Simulation report must contain Baseline and Disruption rows."
        )

    baseline = baseline_rows.iloc[0]
    disruption = disruption_rows.iloc[0]

    baseline_service = float(baseline["service_level_percent"])
    disruption_service = float(disruption["service_level_percent"])

    simulation_metrics = {
        "baseline_service_level": baseline_service,
        "disruption_service_level": disruption_service,
        "service_level_drop": baseline_service - disruption_service,
        "baseline_lost_sales": float(baseline["lost_sales"]),
        "disruption_lost_sales": float(disruption["lost_sales"]),
    }

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

    product_priorities["priority_rank"] = product_priorities.index + 1

    print("[NODE 1] Reports loaded and metrics calculated.")

    return {
        "requirements": requirements,
        "supplier_allocations": supplier_allocations,
        "simulation": simulation,
        "procurement_metrics": procurement_metrics,
        "simulation_metrics": simulation_metrics,
        "product_priorities": product_priorities,
        "status": "ANALYSED",
    }


# --------------------------------------------------
# 5. NODE 2 — DETECT RISKS
# --------------------------------------------------

def detect_risks(state: SupplyChainState) -> dict[str, Any]:
    print("\n[NODE 2] Detecting supply-chain risks...")

    risks: list[dict[str, Any]] = []
    procurement = state["procurement_metrics"]
    simulation = state["simulation_metrics"]
    products = state["product_priorities"]

    if procurement["shortage_quantity"] > 0:
        add_risk(
            risks,
            "RISK_PROCUREMENT_CAPACITY",
            "CRITICAL",
            "Procurement capacity",
            "Supplier allocation is below planned replenishment requirement.",
            (
                f"Planned={procurement['planned_quantity']:,.0f}; "
                f"allocated={procurement['allocated_quantity']:,.0f}; "
                f"shortage={procurement['shortage_quantity']:,.0f}; "
                f"fill={procurement['fill_rate_percent']:.2f}%."
            ),
        )

    zero_allocated = products[
        (products["planned_quantity"] > 0)
        & (products["policy_allocation"] == 0)
    ]

    if not zero_allocated.empty:
        add_risk(
            risks,
            "RISK_ZERO_ALLOCATION",
            "HIGH",
            "Product allocation",
            "Some products have positive requirements but zero allocation.",
            (
                f"Count={len(zero_allocated)}; examples="
                + ", ".join(zero_allocated["product_id"].astype(str).head(10))
            ),
        )

    if simulation["baseline_service_level"] < SERVICE_LEVEL_TARGET:
        add_risk(
            risks,
            "RISK_BASELINE_SERVICE",
            "CRITICAL",
            "Customer service",
            "Baseline service level is below the configured business target.",
            (
                f"Baseline={simulation['baseline_service_level']:.2f}%; "
                f"target={SERVICE_LEVEL_TARGET:.2f}%."
            ),
        )

    if simulation["disruption_service_level"] < SERVICE_LEVEL_TARGET:
        add_risk(
            risks,
            "RISK_DISRUPTION_SERVICE",
            "HIGH",
            "Disruption preparedness",
            "Disruption service level is below the configured business target.",
            (
                f"Disruption={simulation['disruption_service_level']:.2f}%; "
                f"target={SERVICE_LEVEL_TARGET:.2f}%."
            ),
        )

    if simulation["service_level_drop"] > 0:
        add_risk(
            risks,
            "RISK_SERVICE_DEGRADATION",
            "HIGH",
            "Supplier resilience",
            "Disruption scenario reduces network service level.",
            (
                f"Baseline={simulation['baseline_service_level']:.2f}%; "
                f"disruption={simulation['disruption_service_level']:.2f}%; "
                f"drop={simulation['service_level_drop']:.2f} percentage points."
            ),
        )

    if simulation["disruption_lost_sales"] > 0:
        add_risk(
            risks,
            "RISK_LOST_SALES",
            "HIGH",
            "Lost sales",
            "Unfulfilled demand exists in the disruption scenario.",
            f"Lost sales={simulation['disruption_lost_sales']:,.0f} units.",
        )

    print(f"[NODE 2] {len(risks)} risks detected.")

    return {"detected_risks": risks, "status": "RISKS_DETECTED"}


# --------------------------------------------------
# 6. NODE 3 — CHOOSE ACTIONS
# --------------------------------------------------

def choose_actions(state: SupplyChainState) -> dict[str, Any]:
    print("\n[NODE 3] Selecting response actions...")

    action_map = {
        "RISK_PROCUREMENT_CAPACITY": (
            "Review additional supplier capacity, qualify alternate suppliers "
            "and phase procurement by product priority tier."
        ),
        "RISK_ZERO_ALLOCATION": (
            "Review zero-allocation products against business criticality, "
            "customer impact and substitution options."
        ),
        "RISK_BASELINE_SERVICE": (
            "Investigate the replenishment-capacity constraint before treating "
            "the configured service target as achievable."
        ),
        "RISK_DISRUPTION_SERVICE": (
            "Prepare contingency sourcing and review safety-stock coverage "
            "for critical products."
        ),
        "RISK_SERVICE_DEGRADATION": (
            "Evaluate backup suppliers and supplier diversification; compare "
            "resilience cost against service-level benefit."
        ),
        "RISK_LOST_SALES": (
            "Test priority-based allocation and alternate sourcing to reduce "
            "unfulfilled demand."
        ),
    }

    actions = []
    for risk in state["detected_risks"]:
        actions.append({
            "risk_id": risk["risk_id"],
            "priority": risk["severity"],
            "area": risk["area"],
            "recommended_action": action_map[risk["risk_id"]],
        })

    print(f"[NODE 3] {len(actions)} response actions selected.")

    return {"selected_actions": actions, "status": "ACTIONS_SELECTED"}


# --------------------------------------------------
# 7. NODE 4 — GENERATE EXPLAINABLE RECOMMENDATIONS
# --------------------------------------------------

def generate_recommendations(state: SupplyChainState) -> dict[str, Any]:
    print("\n[NODE 4] Generating explainable recommendations...")

    risk_by_id = {
        risk["risk_id"]: risk for risk in state["detected_risks"]
    }

    recommendations = []
    for action in state["selected_actions"]:
        risk = risk_by_id[action["risk_id"]]
        recommendations.append({
            "priority": action["priority"],
            "area": action["area"],
            "finding": risk["finding"],
            "recommended_action": action["recommended_action"],
            "evidence": risk["evidence"],
            "risk_id": action["risk_id"],
        })

    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    recommendations.sort(
        key=lambda item: (
            priority_order.get(item["priority"], 99),
            item["area"],
        )
    )

    print(f"[NODE 4] {len(recommendations)} recommendations generated.")

    return {
        "recommendations": recommendations,
        "status": "RECOMMENDATIONS_GENERATED",
    }


# --------------------------------------------------
# 8. NODE 5 — VALIDATE OUTPUTS
# --------------------------------------------------

def validate_decisions(state: SupplyChainState) -> dict[str, Any]:
    print("\n[NODE 5] Validating recommendations...")

    checks = []

    def record_check(name: str, passed: bool, detail: str) -> None:
        checks.append({
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })

    recommendations = state["recommendations"]
    risks = state["detected_risks"]
    actions = state["selected_actions"]

    record_check(
        "Every detected risk has a response action",
        len(risks) == len(actions),
        f"Risks={len(risks)}, actions={len(actions)}",
    )

    risk_ids = {risk["risk_id"] for risk in risks}
    action_ids = {action["risk_id"] for action in actions}
    record_check(
        "Action risk IDs match detected risk IDs",
        risk_ids == action_ids,
        f"Unmapped risks={sorted(risk_ids - action_ids)}",
    )

    recommendation_ids = {
        recommendation["risk_id"] for recommendation in recommendations
    }
    record_check(
        "Every action has a recommendation",
        action_ids == recommendation_ids,
        f"Unmapped actions={sorted(action_ids - recommendation_ids)}",
    )

    required_fields = {
        "priority", "area", "finding", "recommended_action", "evidence"
    }
    complete = all(
        required_fields.issubset(recommendation)
        and all(str(recommendation[field]).strip() for field in required_fields)
        for recommendation in recommendations
    )
    record_check(
        "Recommendations contain evidence and action",
        complete,
        f"Recommendations checked={len(recommendations)}",
    )

    procurement = state["procurement_metrics"]
    arithmetic_ok = (
        procurement["planned_quantity"] >= procurement["allocated_quantity"]
        >= 0
        and abs(
            procurement["planned_quantity"]
            - procurement["allocated_quantity"]
            - procurement["shortage_quantity"]
        ) < 1e-6
    )
    record_check(
        "Procurement quantities reconcile",
        arithmetic_ok,
        (
            f"Planned={procurement['planned_quantity']:.0f}, "
            f"allocated={procurement['allocated_quantity']:.0f}, "
            f"shortage={procurement['shortage_quantity']:.0f}"
        ),
    )

    failed = [check for check in checks if check["status"] == "FAIL"]
    print(
        f"[NODE 5] Validation: {len(checks) - len(failed)}/{len(checks)} passed."
    )

    return {
        "validation_results": checks,
        "status": "VALIDATED" if not failed else "VALIDATION_FAILED",
    }


# --------------------------------------------------
# 9. BUILD LANGGRAPH WORKFLOW
# --------------------------------------------------

workflow = StateGraph(SupplyChainState)

workflow.add_node("analyse_inputs", analyse_inputs)
workflow.add_node("detect_risks", detect_risks)
workflow.add_node("choose_actions", choose_actions)
workflow.add_node("generate_recommendations", generate_recommendations)
workflow.add_node("validate_decisions", validate_decisions)

workflow.add_edge(START, "analyse_inputs")
workflow.add_edge("analyse_inputs", "detect_risks")
workflow.add_edge("detect_risks", "choose_actions")
workflow.add_edge("choose_actions", "generate_recommendations")
workflow.add_edge("generate_recommendations", "validate_decisions")
workflow.add_edge("validate_decisions", END)

decision_graph = workflow.compile()


# --------------------------------------------------
# 10. RUN WORKFLOW AND SAVE REPORTS
# --------------------------------------------------

if __name__ == "__main__":
    print("\n========== LANGGRAPH SUPPLY CHAIN AGENT ==========")

    final_state = decision_graph.invoke({})

    if final_state.get("status") != "VALIDATED":
        raise RuntimeError(
            "Decision workflow validation failed. "
            "Inspect validation_results before using recommendations."
        )

    recommendations_df = pd.DataFrame(final_state["recommendations"])
    product_priorities = final_state["product_priorities"]
    supplier_allocations = final_state["supplier_allocations"].copy()

    # Supplier-level risk summary based on actual allocated quantities.
    for column in [
        "allocated_quantity",
        "unit_cost",
        "risk_multiplier",
        "risk_adjusted_unit_cost",
    ]:
        supplier_allocations[column] = pd.to_numeric(
            supplier_allocations[column], errors="coerce"
        )

    numeric_supplier_fields = [
        "allocated_quantity",
        "unit_cost",
        "risk_multiplier",
        "risk_adjusted_unit_cost",
    ]

    if supplier_allocations[numeric_supplier_fields].isna().any().any():
        raise ValueError("Invalid supplier allocation numeric values.")

    supplier_allocations["base_value"] = (
        supplier_allocations["allocated_quantity"]
        * supplier_allocations["unit_cost"]
    )
    supplier_allocations["risk_adjusted_value"] = (
        supplier_allocations["allocated_quantity"]
        * supplier_allocations["risk_adjusted_unit_cost"]
    )
    supplier_allocations["weighted_risk"] = (
        supplier_allocations["allocated_quantity"]
        * supplier_allocations["risk_multiplier"]
    )

    supplier_risk = (
        supplier_allocations.groupby("supplier_id", as_index=False)
        .agg(
            allocated_quantity=("allocated_quantity", "sum"),
            base_procurement_value=("base_value", "sum"),
            risk_adjusted_value=("risk_adjusted_value", "sum"),
            weighted_risk_total=("weighted_risk", "sum"),
        )
    )

    supplier_risk["risk_premium"] = (
        supplier_risk["risk_adjusted_value"]
        - supplier_risk["base_procurement_value"]
    )
    supplier_risk["weighted_risk_multiplier"] = (
        supplier_risk["weighted_risk_total"]
        .div(supplier_risk["allocated_quantity"].replace(0, pd.NA))
        .fillna(0)
    )

    validation_df = pd.DataFrame(final_state["validation_results"])

    recommendations_df.to_csv(OUTPUT_RECOMMENDATIONS, index=False)
    product_priorities.to_csv(OUTPUT_PRODUCT_PRIORITIES, index=False)
    supplier_risk.to_csv(OUTPUT_SUPPLIER_RISK, index=False)
    validation_df.to_csv(REPORT_DIR / "langgraph_validation_checks.csv", index=False)

    procurement = final_state["procurement_metrics"]
    simulation = final_state["simulation_metrics"]

    summary = [
        "LANGGRAPH SUPPLY CHAIN DECISION WORKFLOW",
        "=" * 52,
        f"Workflow status: {final_state['status']}",
        "",
        "PROCUREMENT",
        f"Planned quantity: {procurement['planned_quantity']:,.0f}",
        f"Allocated quantity: {procurement['allocated_quantity']:,.0f}",
        f"Shortage quantity: {procurement['shortage_quantity']:,.0f}",
        f"Procurement fill rate: {procurement['fill_rate_percent']:.2f}%",
        "",
        "SIMULATION SERVICE LEVEL",
        f"Baseline: {simulation['baseline_service_level']:.2f}%",
        f"Disruption: {simulation['disruption_service_level']:.2f}%",
        f"Drop: {simulation['service_level_drop']:.2f} percentage points",
        "",
        f"Risks detected: {len(final_state['detected_risks'])}",
        f"Actions selected: {len(final_state['selected_actions'])}",
        f"Recommendations: {len(recommendations_df)}",
        "",
        "VALIDATION CHECKS",
    ]

    for check in final_state["validation_results"]:
        summary.append(
            f"[{check['status']}] {check['check']} — {check['detail']}"
        )

    summary.extend(["", "RECOMMENDATIONS"])
    for number, recommendation in enumerate(
        final_state["recommendations"], start=1
    ):
        summary.extend([
            (
                f"{number}. [{recommendation['priority']}] "
                f"{recommendation['area']}"
            ),
            f"Finding: {recommendation['finding']}",
            f"Action: {recommendation['recommended_action']}",
            f"Evidence: {recommendation['evidence']}",
            "",
        ])

    summary.extend([
        "OUTPUT FILES",
        str(OUTPUT_RECOMMENDATIONS),
        str(OUTPUT_PRODUCT_PRIORITIES),
        str(OUTPUT_SUPPLIER_RISK),
        str(REPORT_DIR / "langgraph_validation_checks.csv"),
    ])

    OUTPUT_SUMMARY.write_text("\n".join(summary), encoding="utf-8")

    print("\n========== WORKFLOW SUMMARY ==========")
    print("Status:", final_state["status"])
    print("Risks detected:", len(final_state["detected_risks"]))
    print("Actions selected:", len(final_state["selected_actions"]))
    print("Recommendations:", len(recommendations_df))
    print("\n========== VALIDATION ==========")

    for check in final_state["validation_results"]:
        print(f"[{check['status']}] {check['check']}: {check['detail']}")

    print("\nReports saved in:", REPORT_DIR)
    print("LangGraph workflow completed.")
