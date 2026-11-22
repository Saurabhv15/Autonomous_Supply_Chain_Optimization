"""
LLM-Powered Supply Chain Decision Agent with Recommendation Critic
--------------------------------------------------------------------
LangGraph workflow:
1. Load validated project reports.
2. Calculate quantitative evidence from source CSVs.
3. Generate recommendations using GroqCloud.
4. Critic LLM reviews each recommendation for evidence and actionability.
5. Validate recommendation/critic consistency.
6. Save final advisory outputs and human-approval template.

Safety:
- Read-only; does not place orders or change inventory.
- Quantitative facts are calculated from project reports.
- LLM output is advisory and requires human approval.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

REPORT_DIR = Path(config.REPORT_DIR)
DEFAULT_MODEL = "openai/gpt-oss-120b"
SERVICE_LEVEL_TARGET = 95.0  # Configurable business assumption.

REQUIRED_FILES = {
    "replenishment": REPORT_DIR / "replenishment_requirements_v5.csv",
    "supplier_allocation": REPORT_DIR / "supplier_allocation_v5.csv",
    "simulation": REPORT_DIR / "simulation_scenario_comparison.csv",
}


class Recommendation(BaseModel):
    risk_id: str = Field(description="Stable unique risk ID, e.g. RISK_01")
    severity: str = Field(description="CRITICAL, HIGH, MEDIUM, or LOW")
    title: str
    evidence: str = Field(description="Use only measured project evidence or labelled assumptions")
    recommended_action: str
    rationale: str
    expected_business_effect: str
    human_approval_required: bool = True


class LLMDecision(BaseModel):
    executive_summary: str
    recommendations: list[Recommendation]
    limitations: list[str]


class CriticItem(BaseModel):
    risk_id: str
    evidence_supported: bool = Field(
        description="True only if claims in evidence can be found in the supplied evidence"
    )
    actionability_score: int = Field(description="Integer from 1 (vague) to 5 (specific)")
    numeric_claims_supported: bool = Field(
        description="True only if numeric claims are directly supported by supplied evidence"
    )
    issues: list[str]
    improved_action: str = Field(
        description="Rewrite as a concrete reviewable action; do not invent numbers or claim execution"
    )
    critic_verdict: str = Field(description="PASS or REVIEW")


class CriticDecision(BaseModel):
    overall_assessment: str
    reviews: list[CriticItem]
    general_limitations: list[str]


class AgentState(TypedDict, total=False):
    inputs: dict[str, Any]
    evidence: dict[str, Any]
    llm_decision: dict[str, Any]
    critic_decision: dict[str, Any]
    validation_errors: list[str]
    status: str
    output_files: list[str]


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required report not found: {path}\n"
            "Run the corresponding optimization/simulation scripts first."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{label} report is empty: {path}")
    return df



def load_llm(max_tokens: int = 2000):
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is missing. Add it to the project-root .env file."
        )

    model_name = os.getenv("GROQ_MODEL", DEFAULT_MODEL)

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0,
        timeout=60,
        max_retries=2,
        max_tokens=max_tokens,
    )


def load_inputs_node(state: AgentState) -> AgentState:
    print("\n[NODE 1] Loading validated project reports...")
    frames = {name: _read_csv(path, name) for name, path in REQUIRED_FILES.items()}
    return {"inputs": {name: df.to_dict(orient="records") for name, df in frames.items()}}


def calculate_evidence_node(state: AgentState) -> AgentState:
    print("[NODE 2] Calculating quantitative evidence...")
    replenishment = pd.DataFrame(state["inputs"]["replenishment"])
    allocation = pd.DataFrame(state["inputs"]["supplier_allocation"])
    simulation = pd.DataFrame(state["inputs"]["simulation"])

    required = {"planned_quantity", "policy_allocation"}
    missing = required - set(replenishment.columns)
    if missing:
        raise ValueError(f"Replenishment report missing columns: {sorted(missing)}")

    planned = pd.to_numeric(replenishment["planned_quantity"], errors="coerce").fillna(0)
    allocated = pd.to_numeric(replenishment["policy_allocation"], errors="coerce").fillna(0)
    planned_total = int(planned.sum())
    allocated_total = int(allocated.sum())
    shortage_total = max(planned_total - allocated_total, 0)
    fill_rate = allocated_total / planned_total * 100 if planned_total else 0.0

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
    }

    product_col = next(
        (c for c in ["product_id", "sku_id"] if c in replenishment.columns), None
    )
    if product_col:
        view = replenishment.copy()
        view["_planned"] = planned
        view["_allocated"] = allocated
        view["_gap"] = (view["_planned"] - view["_allocated"]).clip(lower=0)
        if "priority_score" in view.columns:
            view["priority_score"] = pd.to_numeric(
                view["priority_score"], errors="coerce"
            ).fillna(0)
            view = view.sort_values("priority_score", ascending=False)
        else:
            view = view.sort_values("_gap", ascending=False)
        columns = [product_col, "_planned", "_allocated", "_gap"]
        if "priority_score" in view.columns:
            columns.append("priority_score")
        evidence["top_products"] = view[columns].head(10).to_dict(orient="records")

    # Keep all scenario rows (normally just baseline and disruption).
    evidence["simulation_scenarios"] = simulation.to_dict(orient="records")

    # Derive supplier-level facts from the complete allocation report.
    supplier_col = "supplier_id" if "supplier_id" in allocation.columns else None
    if supplier_col:
        supplier_rows = []
        for supplier_id, group in allocation.groupby(supplier_col):
            row = {"supplier_id": str(supplier_id)}
            for col in ["allocated_quantity", "unit_cost", "risk_adjusted_unit_cost", "risk_multiplier"]:
                if col in group.columns:
                    values = pd.to_numeric(group[col], errors="coerce")
                    if col == "allocated_quantity":
                        row["total_" + col] = int(values.fillna(0).sum())
                    elif col in ["risk_multiplier"]:
                        row["weighted_average_" + col] = round(
                            float((values.fillna(0) * pd.to_numeric(
                                group.get("allocated_quantity", 0), errors="coerce"
                            ).fillna(0)).sum() /
                            max(pd.to_numeric(group.get("allocated_quantity", 0), errors="coerce").fillna(0).sum(), 1)), 4
                        )
                    else:
                        row["average_" + col] = round(float(values.mean()), 4)
            supplier_rows.append(row)
        evidence["supplier_summary"] = supplier_rows

    # Exact product-level values mentioned in recommendations can be checked.
    if product_col and "product_id" in allocation.columns:
        evidence["allocation_products"] = allocation[
            [c for c in [
                "product_id", "supplier_id", "allocated_quantity", "unit_cost",
                "risk_multiplier", "risk_adjusted_unit_cost"
            ] if c in allocation.columns]
        ].to_dict(orient="records")

    print("[NODE 2] Quantitative evidence calculated.")
    return {"evidence": evidence}



def generate_recommendations_node(state: AgentState) -> AgentState:
    print("[NODE 3] Generating recommendations with GroqCloud LLM...")

    llm = load_llm(max_tokens=4000).with_structured_output(
        LLMDecision,
        method="json_schema",
        strict=True
    )
    system = """
You are a supply-chain operations decision-support analyst.

Use only the supplied project evidence.
Never invent metrics, costs, dates, supplier facts, or improvements.
Clearly distinguish measured facts from business assumptions.
Prioritize procurement shortage, customer service, supplier risk,
and disruption preparedness.

Generate at most 6 concise, ranked recommendations.
Each recommendation must contain:
- A unique risk ID
- Valid severity: CRITICAL, HIGH, MEDIUM, or LOW
- Specific evidence
- One practical reviewable action
- Short rationale
- Expected business effect without invented numerical claims

Every recommendation requires human approval.
Do not claim that any action has been executed.
Mention unavailable metrics as unavailable.
Keep the executive summary and limitations concise.
"""

    # Only summarized evidence is sent to the recommendation LLM.
    # Complete allocation details remain available for the critic.
    evidence = state["evidence"]

    compact_evidence = {
        "procurement": evidence.get("procurement", {}),
        "top_products": evidence.get("top_products", []),
        "simulation_scenarios": evidence.get("simulation_scenarios", []),
        "supplier_summary": evidence.get("supplier_summary", [])
    }

    payload = {
        "measured_project_evidence": compact_evidence,
        "business_assumption": {
            "service_level_target_percent": SERVICE_LEVEL_TARGET,
            "note": "This is a configurable business assumption, not an observed result."
        },
        "task": (
            "Identify the most important supply-chain risks. "
            "Generate an executive summary, up to 6 ranked recommendations, "
            "and concise limitations."
        )
    }

    result = llm.invoke([
        ("system", system),
        ("human", json.dumps(payload, default=str))
    ])

    print(
        f"[NODE 3] Generated "
        f"{len(result.recommendations)} recommendations."
    )

    return {"llm_decision": result.model_dump()}


def _numeric_values_in_object(value: Any) -> list[float]:
    """Collect numeric facts from source-derived evidence recursively."""
    found: list[float] = []
    if isinstance(value, dict):
        for item in value.values():
            found.extend(_numeric_values_in_object(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_numeric_values_in_object(item))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        found.append(float(value))
    return found


def _extract_numeric_claims(text: str) -> list[tuple[str, float]]:
    """
    Extract numeric claims while ignoring digits embedded in identifiers such as
    P064, SUP03, and RISK_01. Percent signs and comma-separated numbers are supported.
    """
    pattern = r"(?<![A-Za-z_])[-+]?\d[\d,]*(?:\.\d+)?%?"
    claims = []
    for match in re.finditer(pattern, str(text)):
        raw = match.group(0)
        cleaned = raw.replace(",", "").replace("%", "")
        try:
            claims.append((raw, float(cleaned)))
        except ValueError:
            continue
    return claims


def critic_node(state: AgentState) -> AgentState:
    print("[NODE 4] Critic is reviewing qualitative evidence and actionability...")
    critic_llm = load_llm(max_tokens=4000).with_structured_output(
        CriticDecision,
        method="json_schema",
        strict=True
    )

    system = """
    You are a strict supply-chain recommendation auditor.

    Review each recommendation for:
    1. Evidence relevance
    2. Logical correctness
    3. Practical actionability
    4. Unsupported assumptions or promises

    Rules:
    - Do not invent facts, numbers, targets, or timelines.
    - Do not treat configured business targets as actual performance.
    - If evidence is insufficient, mark REVIEW.
    - PASS only when the recommendation is supported and actionable.
    - Suggest a correction when needed.

    Return exactly one structured review for every recommendation.
    Keep each review concise.
    """

    # Provide complete small source summaries, not arbitrary first-N row samples.

    evidence = state["evidence"]

    # Keep only the most relevant evidence for qualitative review.
    compact_evidence = {
        "procurement": evidence.get("procurement", {}),
        "top_products": evidence.get("top_products", [])[:5],
        "simulation_scenarios": evidence.get("simulation_scenarios", []),
        "supplier_summary": evidence.get("supplier_summary", [])
    }


    compact_recommendations = []
    for rec in state.get("llm_decision", {}).get("recommendations", []):
        compact_recommendations.append({
            "risk_id": rec.get("risk_id"),
            "severity": rec.get("severity"),
            "title": rec.get("title"),
            "evidence": rec.get("evidence"),
            "recommended_action": rec.get("recommended_action"),
            "rationale": rec.get("rationale"),
            "expected_business_effect": rec.get("expected_business_effect"),
        })

    result = critic_llm.invoke([
        ("system", system),
        ("human", json.dumps({
            "source_derived_evidence": compact_evidence,
            "recommendations_to_review": compact_recommendations,
        }, default=str)),
    ])

    # Deterministic numerical audit: compare every numeric claim with source facts.
    # A tolerance handles harmless CSV/rounding representation differences.
    trusted_numbers = _numeric_values_in_object(state["evidence"])
    reviews_by_id = {str(item.risk_id): item.model_dump() for item in result.reviews}
    for rec in state.get("llm_decision", {}).get("recommendations", []):
        risk_id = str(rec.get("risk_id"))
        review = reviews_by_id.get(risk_id)
        if review is None:
            continue

        claim_fields = [
            "title", "evidence", "recommended_action", "rationale",
            "expected_business_effect"
        ]
        all_claims: list[tuple[str, float]] = []
        for field in claim_fields:
            all_claims.extend(_extract_numeric_claims(str(rec.get(field, ""))))

        unsupported = []
        for raw, number in all_claims:
            supported = any(
                abs(number - known) <= max(0.0001, abs(known) * 0.0001)
                for known in trusted_numbers
            )
            if not supported:
                unsupported.append(raw)

        review["numeric_claims_found"] = [raw for raw, _ in all_claims]
        review["unsupported_numeric_claims"] = sorted(set(unsupported))
        review["numeric_claims_supported"] = len(unsupported) == 0

        # PASS requires qualitative evidence support, actionability, and no
        # unsupported numeric claims. This remains a review aid, not proof.
        review["critic_verdict"] = (
            "PASS"
            if review.get("evidence_supported") is True
            and int(review.get("actionability_score", 0)) >= 4
            and review["numeric_claims_supported"] is True
            else "REVIEW"
        )

    final_reviews = list(reviews_by_id.values())
    expected_ids = {str(r.get("risk_id")) for r in compact_recommendations}
    actual_ids = [str(r.get("risk_id")) for r in final_reviews]
    if set(actual_ids) != expected_ids or len(actual_ids) != len(set(actual_ids)):
        print("[NODE 4] Warning: critic returned missing or duplicate risk IDs.")

    print(f"[NODE 4] Critic reviewed {len(final_reviews)} recommendations.")
    return {"critic_decision": {
        "overall_assessment": result.overall_assessment,
        "reviews": final_reviews,
        "general_limitations": result.general_limitations,
    }}


def validate_decisions_node(state: AgentState) -> AgentState:
    print("[NODE 5] Validating recommendations and critic results...")
    errors: list[str] = []
    recs = state.get("llm_decision", {}).get("recommendations", [])
    reviews = state.get("critic_decision", {}).get("reviews", [])

    rec_ids = [str(r.get("risk_id", "")).strip() for r in recs]
    review_ids = [str(r.get("risk_id", "")).strip() for r in reviews]

    if not recs:
        errors.append("No recommendations were generated.")
    if len(rec_ids) != len(set(rec_ids)) or "" in rec_ids:
        errors.append("Recommendation risk IDs are blank or duplicated.")
    if set(rec_ids) != set(review_ids) or len(review_ids) != len(set(review_ids)):
        errors.append("Critic reviews do not map one-to-one to recommendation risk IDs.")

    for i, rec in enumerate(recs):
        if rec.get("severity") not in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}:
            errors.append(f"Recommendation {i + 1} has invalid severity.")
        if rec.get("human_approval_required") is not True:
            errors.append(f"Recommendation {i + 1} does not require human approval.")
        for field in ("title", "evidence", "recommended_action", "rationale"):
            if not str(rec.get(field, "")).strip():
                errors.append(f"Recommendation {i + 1} is missing {field}.")

    proc = state["evidence"]["procurement"]
    if proc["planned_quantity"] != proc["allocated_quantity"] + proc["shortage_quantity"]:
        errors.append("Procurement quantities do not reconcile.")

    status = "VALIDATED" if not errors else "REVIEW_REQUIRED"
    return {"validation_errors": errors, "status": status}


def save_outputs_node(state: AgentState) -> AgentState:
    print("[NODE 6] Saving audited recommendations...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    decision = state.get("llm_decision", {})
    critic = state.get("critic_decision", {})
    recs = decision.get("recommendations", [])
    reviews = critic.get("reviews", [])
    review_by_id = {str(item.get("risk_id")): item for item in reviews}

    final_rows = []
    for rec in recs:
        row = dict(rec)
        review = review_by_id.get(str(rec.get("risk_id")), {})
        row["critic_verdict"] = review.get("critic_verdict", "MISSING")
        row["critic_evidence_supported"] = review.get("evidence_supported", False)
        row["critic_numeric_claims_supported"] = review.get("numeric_claims_supported", False)
        row["critic_actionability_score"] = review.get("actionability_score", 0)
        row["critic_issues"] = " | ".join(review.get("issues", []))
        row["critic_improved_action"] = review.get("improved_action", "")
        row["numeric_claims_found"] = " | ".join(review.get("numeric_claims_found", []))
        row["unsupported_numeric_claims"] = " | ".join(review.get("unsupported_numeric_claims", []))
        # Keep original LLM recommendation unchanged; critic rewrite is a separate field.
        final_rows.append(row)

    recommendations_path = REPORT_DIR / "llm_agent_recommendations_critic_reviewed.csv"
    critic_path = REPORT_DIR / "llm_agent_critic_review.csv"
    summary_path = REPORT_DIR / "llm_agent_critic_decision_summary.txt"
    approval_path = REPORT_DIR / "llm_agent_critic_human_approval_template.csv"

    pd.DataFrame(final_rows).to_csv(recommendations_path, index=False)
    pd.DataFrame(reviews).to_csv(critic_path, index=False)

    approval_df = pd.DataFrame(final_rows)
    if not approval_df.empty:
        approval_df["human_decision"] = "PENDING"
        approval_df["reviewer"] = ""
        approval_df["review_comment"] = ""
        approval_df["reviewed_at_utc"] = ""
    approval_df.to_csv(approval_path, index=False)

    critic_pass = sum(1 for r in reviews if r.get("critic_verdict") == "PASS")
    critic_review = sum(1 for r in reviews if r.get("critic_verdict") != "PASS")

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("LLM SUPPLY CHAIN AGENT — CRITIC REVIEW\n")
        f.write("=" * 52 + "\n")
        f.write(f"Generated UTC: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Workflow status: {state.get('status', 'UNKNOWN')}\n")
        f.write(f"Recommendations: {len(recs)}\n")
        f.write(f"Critic PASS: {critic_pass}\n")
        f.write(f"Critic REVIEW: {critic_review}\n\n")
        f.write("EXECUTIVE SUMMARY\n")
        f.write(str(decision.get("executive_summary", "")) + "\n\n")
        f.write("CRITIC ASSESSMENT\n")
        f.write(str(critic.get("overall_assessment", "")) + "\n\n")
        f.write("QUANTITATIVE EVIDENCE\n")
        f.write(json.dumps(state["evidence"], indent=2, default=str) + "\n\n")
        f.write("VALIDATION ERRORS\n")
        f.write(json.dumps(state.get("validation_errors", []), indent=2) + "\n\n")
        f.write("LIMITATIONS\n")
        f.write(json.dumps(
            decision.get("limitations", []) + critic.get("general_limitations", []),
            indent=2
        ) + "\n\n")
        f.write(
            "SAFETY: Recommendations are advisory only. No purchase order, inventory "
            "transfer, supplier communication, or route execution was performed. "
            "Human approval is required.\n"
        )

    outputs = [
        str(recommendations_path),
        str(critic_path),
        str(summary_path),
        str(approval_path),
    ]
    print(f"[NODE 6] Workflow status: {state.get('status', 'UNKNOWN')}")
    print(f"[NODE 6] Critic PASS={critic_pass}, REVIEW={critic_review}")
    return {"output_files": outputs}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("load_inputs", load_inputs_node)
    graph.add_node("calculate_evidence", calculate_evidence_node)
    graph.add_node("generate_recommendations", generate_recommendations_node)
    graph.add_node("critic", critic_node)
    graph.add_node("validate_decisions", validate_decisions_node)
    graph.add_node("save_outputs", save_outputs_node)

    graph.add_edge(START, "load_inputs")
    graph.add_edge("load_inputs", "calculate_evidence")
    graph.add_edge("calculate_evidence", "generate_recommendations")
    graph.add_edge("generate_recommendations", "critic")
    graph.add_edge("critic", "validate_decisions")
    graph.add_edge("validate_decisions", "save_outputs")
    graph.add_edge("save_outputs", END)
    return graph.compile()


if __name__ == "__main__":
    print("\n========== LANGGRAPH SUPPLY CHAIN AGENT + CRITIC ==========")
    app = build_graph()
    final_state = app.invoke({})

    print("\n========== WORKFLOW SUMMARY ==========")
    print(f"Status: {final_state.get('status', 'UNKNOWN')}")
    print(f"Recommendations: {len(final_state.get('llm_decision', {}).get('recommendations', []))}")
    print(f"Critic reviews: {len(final_state.get('critic_decision', {}).get('reviews', []))}")
    print(f"Validation errors: {len(final_state.get('validation_errors', []))}")

    print("\nReports saved:")
    for output in final_state.get("output_files", []):
        print(f"- {output}")

    print("\nNo operational action was executed. Human approval is required.")
