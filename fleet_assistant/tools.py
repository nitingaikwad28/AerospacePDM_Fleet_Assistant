# tools.py
# The "tools" the assistant can call - each one answers exactly one kind of question,
# using only data the core pipeline already produced. None of these tools SEND anything
# or change any state (no calls to an MRO system, no file writes) - they only read and
# summarize. Turning a draft into a real, sent action always stays a separate, explicit,
# human-confirmed step (see draft_work_order below and the guardrail note in agent.py).

from config import TOP_PRIORITY_DEFAULT_N, ROLLING_TREND_CYCLES
from data_access import (
    load_fleet_status, load_work_orders, load_model_metrics, load_telemetry_history, find_unit,
)


def get_fleet_summary():
    df = load_fleet_status()
    metrics = load_model_metrics()
    counts = df["health_status"].value_counts().to_dict()
    anomaly = metrics.get("anomaly_detection", {})
    return {
        "total_units": int(len(df)),
        "status_counts": counts,
        "model_rul_mae_cycles": metrics.get("rul_overall", {}).get("MAE"),
        "model_anomaly_recall": anomaly.get("recall"),
        "model_mean_detection_lead_time_cycles": anomaly.get("mean_detection_lead_time_cycles"),
    }


def get_top_priority(n=TOP_PRIORITY_DEFAULT_N):
    df = load_fleet_status()
    severity_rank = {"Critical": 0, "Warning": 1, "Healthy": 2}
    df = df.copy()
    df["_severity"] = df["health_status"].map(severity_rank)
    ranked = df.sort_values(["_severity", "predicted_RUL"])
    top = ranked.head(n)
    return top[["aircraft_id", "unit_id", "component", "predicted_RUL",
                "anomaly_score", "is_anomaly", "health_status"]].to_dict("records")


def get_unit_detail(unit_id_or_aircraft):
    df = load_fleet_status()
    matches = find_unit(df, unit_id_or_aircraft)
    if len(matches) == 0:
        return {"found": False, "query": unit_id_or_aircraft}

    results = []
    history = load_telemetry_history()
    for _, row in matches.iterrows():
        detail = {
            "found": True,
            "unit_id": row["unit_id"],
            "aircraft_id": row["aircraft_id"],
            "component": row["component"],
            "cycle": int(row["cycle"]),
            "predicted_RUL": round(float(row["predicted_RUL"]), 1),
            "anomaly_score": round(float(row["anomaly_score"]), 3),
            "is_anomaly": bool(row["is_anomaly"]),
            "health_status": row["health_status"],
            "trend": None,
        }
        if history is not None:
            unit_hist = history[history["unit_id"] == row["unit_id"]].sort_values("cycle")
            if len(unit_hist) >= 2:
                recent = unit_hist.tail(ROLLING_TREND_CYCLES)
                sensor_cols = [c for c in ["vibration_mm_s", "temperature_c", "pressure_psi",
                                            "oil_debris_ppm", "rotational_speed_deviation_pct"]
                               if c in recent.columns]
                trend = {}
                for col in sensor_cols:
                    delta = recent[col].iloc[-1] - recent[col].iloc[0]
                    direction = "rising" if delta > 0 else ("falling" if delta < 0 else "flat")
                    trend[col] = {"direction": direction, "change": round(float(delta), 3)}
                detail["trend"] = trend
                detail["trend_window_cycles"] = int(len(recent))
        results.append(detail)
    return {"found": True, "units": results}


def find_similar_incidents(component, exclude_unit_id=None):
    # Fleet-wide cross-reference: which OTHER units of the same component type are
    # currently flagged? This is an honest simplification of "similar historical
    # incidents" - it compares against the fleet's current snapshot, not a deep
    # incident/root-cause database (a real deployment would maintain one).
    df = load_fleet_status()
    same_component = df[(df["component"] == component) & (df["is_anomaly"] == 1)]
    if exclude_unit_id:
        same_component = same_component[same_component["unit_id"] != exclude_unit_id]
    return {
        "component": component,
        "similar_flagged_count": int(len(same_component)),
        "similar_units": same_component[["unit_id", "aircraft_id", "predicted_RUL",
                                          "anomaly_score"]].to_dict("records"),
    }


def draft_work_order(unit_id_or_aircraft):
    # READ-ONLY: looks up whether the core pipeline already auto-generated a work order
    # for this unit (predict_service.py -> mro_integration_v2.py does this for every
    # Warning/Critical unit). This tool never creates or sends a NEW work order itself -
    # it only surfaces what already exists, clearly labeled as not yet sent/confirmed.
    orders = load_work_orders()
    key = unit_id_or_aircraft.strip().upper()
    matches = [wo for wo in orders
               if wo["unit_id"].upper() == key or wo["aircraft_id"].upper() == key]
    if not matches:
        return {"has_draft": False, "query": unit_id_or_aircraft}
    return {"has_draft": True, "work_orders": matches}
