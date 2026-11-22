"""
LLM-Powered Supply Chain Decision Agent
---------------------------------------
Uses LangGraph + GroqCloud (OpenAI-compatible API) to explain supply-chain risks and recommend actions.

Safety:
- Read-only: never places orders or changes inventory.
- Quantitative facts are calculated from project CSV reports, not invented by the LLM.
- LLM recommendations are advisory and require an explicit human approval record.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# Make project-root imports and paths work when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402


REPORT_DIR = Path(config.REPORT_DIR)
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
SERVICE_LEVEL_TARGET = 95.0  # Business assumption; change only with justification.

REQUIRED_FILES = {
    "replenishment": REPORT_DIR / "replenishment_requirements_v5.csv",
    "supplier_allocation": REPORT_DIR / "supplier_allocation_v5.csv",
    "simulation": REPORT_DIR / "simulation_scenario_comparison.csv",
}


class Recommendation(BaseModel):
    risk_id: str = Field(description="Stable ID of the risk, e.g. RISK_01")
    severity: str = Field(description="One of CRITICAL, HIGH, MEDIUM, LOW")
    title: str
    evidence: str = Field(description="Must refer only to supplied metrics or clearly labelled assumptions")
    recommended_action: str
    rationale: str
    expected_business_effect: str
    human_approval_required: bool = True


class LLMDecision(BaseModel):
    executive_summary: str
    recommendations: list[Recommendation]
    limitations: list[str]


class AgentState(TypedDict, total=False):
    inputs: dict[str, Any]
    evidence: dict[str, Any]
    llm_decision: dict[str, Any]
    validation_errors: list[str]
    status: str
    output_files: list[str]


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required report not found: {path}\n"
            f"Run the corresponding optimization/simulation scripts first."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{label} report is empty: {path}")
    return df


def load_inputs_node(state: AgentState) -> AgentState:
    print("\n[NODE 1] Loading validated project reports...")
    frames = {
        name: _read_csv(path, name)
        for name, path in REQUIRED_FILES.items()
    }
    return {"inputs": {name: df.to_dict(orient="records") for name, df in frames.items()}}


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def calculate_evidence_node(state: AgentState) -> AgentState:
    print("[NODE 2] Calculating quantitative evidence...")
    replenishment = pd.DataFrame(state["inputs"]["replenishment"])
    allocation = pd.DataFrame(state["inputs"]["supplier_allocation"])
    simulation = pd.DataFrame(state["inputs"]["simulation"])

    required_cols = {"planned_quantity", "policy_allocation"}
    missing = required_cols - set(replenishment.columns)
    if missing:
        raise ValueError(f"Replenishment report missing columns: {sorted(missing)}")

    planned = pd.to_numeric(replenishment["planned_quantity"], errors="coerce").fillna(0)
    allocated = pd.to_numeric(replenishment["policy_allocation"], errors="coerce").fillna(0)
    planned_total = int(planned.sum())
    allocated_total = int(allocated.sum())
    shortage_total = max(planned_total - allocated_total, 0)
    fill_rate = (allocated_total / planned_total * 100) if planned_total else 0.0

    evidence: dict[str, Any] = {
        "procurement": {
            "planned_quantity": planned_total,
            "allocated_quantity": allocated_total,
            "shortage_quantity": shortage_total,
            "fill_rate_percent": round(fill_rate, 4),
            "service_level_target_percent": SERVICE_LEVEL_TARGET,
        },
        "input_row_counts": {
            "replenishment_products": int(len(replenishment)),
            "supplier_allocation_rows": int(len(allocation)),
            "simulation_rows": int(len(simulation)),
        },
        "supplier_allocation_columns": list(allocation.columns),
        "simulation_columns": list(simulation.columns),
        "simulation_scenarios": simulation.head(20).to_dict(orient="records"),
    }

    # Add a compact list of the most under-supplied products where columns exist.
    product_col = _first_existing_column(replenishment, ["product_id", "sku_id"])
    if product_col:
        product_view = replenishment.copy()
        product_view["_planned"] = planned
        product_view["_allocated"] = allocated
        product_view["_gap"] = (product_view["_planned"] - product_view["_allocated"]).clip(lower=0)
        sort_col = _first_existing_column(product_view, ["priority_score"])
        if sort_col:
            product_view[sort_col] = pd.to_numeric(product_view[sort_col], errors="coerce").fillna(0)
            product_view = product_view.sort_values(sort_col, ascending=False)
        else:
            product_view = product_view.sort_values("_gap", ascending=False)
        evidence["top_products"] = product_view[
            [product_col, "_planned", "_allocated", "_gap"]
            + ([sort_col] if sort_col else [])
        ].head(10).to_dict(orient="records")

    # Detect service-level columns without assuming a fixed report schema.
    service_cols = [
        c for c in simulation.columns
        if "service" in c.lower() and ("level" in c.lower() or "rate" in c.lower())
    ]
    if service_cols:
        evidence["simulation_service_columns"] = service_cols

    print("[NODE 2] Quantitative evidence calculated.")
    return {"evidence": evidence}


def generate_llm_decision_node(state: AgentState) -> AgentState:
    print("[NODE 3] Asking the LLM to reason over the measured evidence...")
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv()

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add GROQ_API_KEY to the project-root .env file, "
            "then restart the run. Do not share your API key."
        )

    model_name = os.getenv("GROQ_MODEL", MODEL_NAME)
    llm = ChatOpenAI(
        model=model_name,
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0,
        timeout=60,
        max_retries=2,
    )
    # Groq GPT OSS 120B supports JSON-schema structured output.
    # This keeps the LLM response aligned with the Pydantic decision schema.
    structured_llm = llm.with_structured_output(
        LLMDecision,
        method="json_schema",
        strict=True,
    )

    system_prompt = """
You are a supply-chain operations decision-support analyst.
Analyze only the evidence supplied in the user message.
Do not invent measurements, supplier facts, costs, dates, or model performance.
Separate measured facts from assumptions and recommendations.
Prioritize customer service, shortage exposure, supplier resilience, and disruption risk.
Give practical actions that an operations team could review.
Do not claim that an action has been executed.
Every recommendation must require human approval.
If a metric is absent, say it is unavailable instead of guessing.
Keep recommendations specific and explain the evidence behind each one.
"""

    user_payload = {
        "measured_project_evidence": state["evidence"],
        "business_assumption": {
            "service_level_target_percent": SERVICE_LEVEL_TARGET,
            "note": "This is a configurable project assumption, not an observed result."
        },
        "task": (
            "Identify the most important supply-chain risks and produce up to 8 "
            "ranked recommendations. Use stable risk IDs. Include an executive "
            "summary and limitations. Do not create unsupported numerical forecasts."
        ),
    }

    result = structured_llm.invoke([
        ("system", system_prompt),
        ("human", json.dumps(user_payload, default=str)),
    ])
    return {"llm_decision": result.model_dump()}


def validate_decision_node(state: AgentState) -> AgentState:
    print("[NODE 4] Validating LLM recommendations...")
    errors: list[str] = []
    decision = state.get("llm_decision", {})
    recommendations = decision.get("recommendations", [])

    if not isinstance(recommendations, list) or not recommendations:
        errors.append("LLM returned no recommendations.")

    seen_ids: set[str] = set()
    for i, rec in enumerate(recommendations):
        risk_id = str(rec.get("risk_id", "")).strip()
        if not risk_id:
            errors.append(f"Recommendation {i + 1} has no risk_id.")
        elif risk_id in seen_ids:
            errors.append(f"Duplicate risk_id: {risk_id}")
        seen_ids.add(risk_id)

        if rec.get("severity") not in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}:
            errors.append(f"Recommendation {i + 1} has invalid severity.")
        for field in ("title", "evidence", "recommended_action", "rationale"):
            if not str(rec.get(field, "")).strip():
                errors.append(f"Recommendation {i + 1} is missing {field}.")
        if rec.get("human_approval_required") is not True:
            errors.append(f"Recommendation {i + 1} does not require human approval.")

    # Recalculate procurement arithmetic independently of the LLM.
    proc = state["evidence"]["procurement"]
    if proc["planned_quantity"] != (
        proc["allocated_quantity"] + proc["shortage_quantity"]
    ):
        errors.append("Procurement quantities do not reconcile.")

    return {
        "validation_errors": errors,
        "status": "VALIDATED" if not errors else "REVIEW_REQUIRED",
    }


def save_outputs_node(state: AgentState) -> AgentState:
    print("[NODE 5] Saving decision-support outputs...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    decision = state.get("llm_decision", {})
    recommendations = decision.get("recommendations", [])

    rec_path = REPORT_DIR / "llm_agent_recommendations.csv"
    pd.DataFrame(recommendations).to_csv(rec_path, index=False)

    summary_path = REPORT_DIR / "llm_agent_decision_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("LLM-POWERED SUPPLY CHAIN DECISION SUPPORT\n")
        f.write("=" * 48 + "\n")
        f.write(f"Generated UTC: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Workflow status: {state.get('status', 'UNKNOWN')}\n\n")
        f.write("EXECUTIVE SUMMARY\n")
        f.write(str(decision.get("executive_summary", "Not provided")) + "\n\n")
        f.write("QUANTITATIVE EVIDENCE\n")
        f.write(json.dumps(state["evidence"], indent=2, default=str) + "\n\n")
        f.write("VALIDATION ERRORS\n")
        f.write(json.dumps(state.get("validation_errors", []), indent=2) + "\n\n")
        f.write("LIMITATIONS\n")
        f.write(json.dumps(decision.get("limitations", []), indent=2) + "\n\n")
        f.write("SAFETY NOTE\n")
        f.write(
            "Recommendations are advisory only. No purchase order, inventory "
            "transfer, supplier communication, or route execution was performed.\n"
        )

    approval_path = REPORT_DIR / "llm_agent_human_approval_template.csv"
    approval_df = pd.DataFrame(recommendations)
    if not approval_df.empty:
        approval_df["human_decision"] = "PENDING"
        approval_df["reviewer"] = ""
        approval_df["review_comment"] = ""
        approval_df["reviewed_at_utc"] = ""
    approval_df.to_csv(approval_path, index=False)

    outputs = [str(rec_path), str(summary_path), str(approval_path)]
    print(f"[NODE 5] Workflow status: {state.get('status', 'UNKNOWN')}")
    print(f"[NODE 5] Recommendations saved: {len(recommendations)}")
    return {"output_files": outputs}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("load_inputs", load_inputs_node)
    graph.add_node("calculate_evidence", calculate_evidence_node)
    graph.add_node("generate_llm_decision", generate_llm_decision_node)
    graph.add_node("validate_decision", validate_decision_node)
    graph.add_node("save_outputs", save_outputs_node)

    graph.add_edge(START, "load_inputs")
    graph.add_edge("load_inputs", "calculate_evidence")
    graph.add_edge("calculate_evidence", "generate_llm_decision")
    graph.add_edge("generate_llm_decision", "validate_decision")
    graph.add_edge("validate_decision", "save_outputs")
    graph.add_edge("save_outputs", END)
    return graph.compile()


if __name__ == "__main__":
    print("\n========== LLM-POWERED LANGGRAPH SUPPLY CHAIN AGENT ==========")
    app = build_graph()
    final_state = app.invoke({})
    print("\n========== WORKFLOW SUMMARY ==========")
    print(f"Status: {final_state.get('status', 'UNKNOWN')}")
    print(f"Recommendations: {len(final_state.get('llm_decision', {}).get('recommendations', []))}")
    print(f"Validation errors: {len(final_state.get('validation_errors', []))}")
    print("\nReports saved:")
    for path in final_state.get("output_files", []):
        print(f"- {path}")
    print("\nNo operational action was executed. Human review is required.")
