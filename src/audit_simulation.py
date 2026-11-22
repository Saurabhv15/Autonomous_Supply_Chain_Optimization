from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Make project root importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / "reports"

SUMMARY_FILE = REPORT_DIR / "simulation_scenario_comparison.csv"
BASELINE_FILE = REPORT_DIR / "simulation_baseline_deliveries.csv"
DISRUPTION_FILE = REPORT_DIR / "simulation_disruption_deliveries.csv"

checks = []


def record_check(name, condition, details=""):
    passed = bool(condition)
    checks.append({
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "details": str(details)
    })
    symbol = "PASS" if passed else "FAIL"
    print(f"[{symbol}] {name}" + (f" | {details}" if details else ""))


def require_columns(df, required, file_name):
    missing = sorted(set(required) - set(df.columns))
    record_check(
        f"{file_name}: required columns present",
        len(missing) == 0,
        f"Missing columns: {missing}" if missing else "All required columns found"
    )
    return len(missing) == 0


def audit_delivery_file(df, scenario_name):
    print(f"\n--- {scenario_name} delivery audit ---")

    required = [
        "supplier_id", "product_id", "quantity",
        "status", "delivery_day", "unreceived_quantity"
    ]
    if not require_columns(df, required, scenario_name):
        return None

    record_check(
        f"{scenario_name}: file contains orders",
        len(df) > 0,
        f"Rows: {len(df)}"
    )

    allowed_statuses = {"DELIVERED", "FAILED"}
    unexpected = sorted(set(df["status"].dropna().unique()) - allowed_statuses)
    record_check(
        f"{scenario_name}: only valid delivery statuses",
        len(unexpected) == 0 and df["status"].notna().all(),
        f"Unexpected statuses: {unexpected}"
    )

    quantity_numeric = pd.to_numeric(df["quantity"], errors="coerce")
    record_check(
        f"{scenario_name}: order quantities are positive",
        quantity_numeric.notna().all() and (quantity_numeric > 0).all(),
        f"Minimum quantity: {quantity_numeric.min()}"
    )

    delivered = df[df["status"] == "DELIVERED"].copy()
    failed = df[df["status"] == "FAILED"].copy()

    delivered_days = pd.to_numeric(delivered["delivery_day"], errors="coerce")
    record_check(
        f"{scenario_name}: delivered orders have valid delivery days",
        delivered_days.notna().all() and (delivered_days >= 0).all(),
        f"Delivered orders: {len(delivered)}"
    )

    unreceived = pd.to_numeric(
        delivered["unreceived_quantity"], errors="coerce"
    )
    delivered_qty = pd.to_numeric(delivered["quantity"], errors="coerce")
    record_check(
        f"{scenario_name}: unreceived quantity is within order quantity",
        unreceived.notna().all()
        and (unreceived >= 0).all()
        and (unreceived <= delivered_qty).all(),
        "For delivered orders: 0 <= unreceived_quantity <= quantity"
    )

    record_check(
        f"{scenario_name}: failed orders have no delivery day",
        failed["delivery_day"].isna().all(),
        f"Failed orders: {len(failed)}"
    )

    record_check(
        f"{scenario_name}: failed orders have no received-quantity record",
        failed["unreceived_quantity"].isna().all(),
        "Failed orders should have NaN unreceived_quantity in this simulation output"
    )

    return {
        "total_orders": len(df),
        "successful_deliveries": len(delivered),
        "failed_deliveries": len(failed)
    }


def main():
    print("========== SUPPLY CHAIN SIMULATION AUDIT ==========")

    missing_files = [
        str(path) for path in
        [SUMMARY_FILE, BASELINE_FILE, DISRUPTION_FILE]
        if not path.exists()
    ]

    record_check(
        "All required simulation report files exist",
        len(missing_files) == 0,
        f"Missing files: {missing_files}" if missing_files else "All 3 files found"
    )

    if missing_files:
        print("\nAudit stopped because required files are missing.")
        return

    summary = pd.read_csv(SUMMARY_FILE)
    baseline = pd.read_csv(BASELINE_FILE)
    disruption = pd.read_csv(DISRUPTION_FILE)

    print(f"\nScenario summary rows: {len(summary)}")
    print(f"Baseline delivery rows: {len(baseline)}")
    print(f"Disruption delivery rows: {len(disruption)}")

    required_summary = [
        "scenario", "total_demand", "fulfilled_demand", "lost_sales",
        "service_level_percent", "successful_deliveries",
        "failed_deliveries", "total_orders"
    ]

    if not require_columns(summary, required_summary, "scenario summary"):
        print("\nAudit stopped because summary columns are missing.")
        return

    record_check(
        "Exactly Baseline and Disruption scenarios are present",
        set(summary["scenario"]) == {"Baseline", "Disruption"}
        and len(summary) == 2,
        f"Scenarios found: {summary['scenario'].tolist()}"
    )

    baseline_counts = audit_delivery_file(baseline, "Baseline")
    disruption_counts = audit_delivery_file(disruption, "Disruption")

    print("\n--- Scenario summary audit ---")

    for _, row in summary.iterrows():
        scenario = row["scenario"]
        total = row["total_demand"]
        fulfilled = row["fulfilled_demand"]
        lost = row["lost_sales"]

        record_check(
            f"{scenario}: demand balance",
            total == fulfilled + lost,
            f"{total} = {fulfilled} + {lost}"
        )

        calculated_service = (
            fulfilled / total * 100 if total > 0 else np.nan
        )
        record_check(
            f"{scenario}: service level calculation",
            np.isfinite(calculated_service)
            and np.isclose(
                calculated_service,
                row["service_level_percent"],
                atol=0.00001
            ),
            f"Reported: {row['service_level_percent']:.6f}%, "
            f"Calculated: {calculated_service:.6f}%"
        )

        counts = (
            baseline_counts if scenario == "Baseline"
            else disruption_counts
        )

        if counts is not None:
            record_check(
                f"{scenario}: total order count matches delivery file",
                int(row["total_orders"]) == counts["total_orders"],
                f"Summary: {row['total_orders']}, "
                f"Delivery file: {counts['total_orders']}"
            )
            record_check(
                f"{scenario}: successful delivery count matches",
                int(row["successful_deliveries"])
                == counts["successful_deliveries"],
                f"Summary: {row['successful_deliveries']}, "
                f"Delivery file: {counts['successful_deliveries']}"
            )
            record_check(
                f"{scenario}: failed delivery count matches",
                int(row["failed_deliveries"])
                == counts["failed_deliveries"],
                f"Summary: {row['failed_deliveries']}, "
                f"Delivery file: {counts['failed_deliveries']}"
            )

    audit_df = pd.DataFrame(checks)
    output_file = REPORT_DIR / "audit_simulation_checks.csv"
    audit_df.to_csv(output_file, index=False)

    passed = (audit_df["status"] == "PASS").sum()
    failed = (audit_df["status"] == "FAIL").sum()

    print("\n========== AUDIT SUMMARY ==========")
    print(f"Total checks : {len(audit_df)}")
    print(f"Passed       : {passed}")
    print(f"Failed       : {failed}")
    print(f"Audit report : {output_file}")

    if failed == 0:
        print("\nALL AVAILABLE SIMULATION AUDIT CHECKS PASSED!")
    else:
        print("\nSOME AUDIT CHECKS FAILED. Review the FAIL rows in the report.")

    print(
        "\nScope note: This audit checks report-level arithmetic, delivery "
        "record consistency, and summary counts. It does not independently "
        "reconstruct inventory balances or prove that the simulation model "
        "is globally optimal."
    )


if __name__ == "__main__":
    main()
