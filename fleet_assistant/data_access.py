# data_access.py
# Reads the core pipeline's already-existing output files. No sensor math, no models -
# by the time these files exist, predict_service.py (in the core project) has already
# done all the anomaly detection and RUL prediction. This module's only job is loading
# and looking things up.

import json
import os

import pandas as pd

from config import FLEET_STATUS_CSV, WORK_ORDERS_JSON, MODEL_METRICS_JSON, TELEMETRY_HISTORY_CSV


class DataUnavailable(Exception):
    # Raised when a required output file doesn't exist yet - e.g. predict_service.py
    # hasn't been run. The agent turns this into a plain-English message rather than a
    # stack trace.
    pass


def load_fleet_status():
    if not os.path.exists(FLEET_STATUS_CSV):
        raise DataUnavailable(
            "No fleet_status_v2.csv found yet. Run train.py then predict_service.py "
            "in the core src_v2/ project first.")
    return pd.read_csv(FLEET_STATUS_CSV)


def load_work_orders():
    if not os.path.exists(WORK_ORDERS_JSON):
        return []
    with open(WORK_ORDERS_JSON) as f:
        return json.load(f)


def load_model_metrics():
    if not os.path.exists(MODEL_METRICS_JSON):
        return {}
    with open(MODEL_METRICS_JSON) as f:
        return json.load(f)


def load_telemetry_history():
    # Full per-cycle sensor history for every unit, from the training snapshot. Used for
    # "what's the trend" style questions. In a real deployment this would be a proper
    # telemetry warehouse/database rather than a CSV re-read on every question - noted in
    # the documentation as a known simplification, not hidden.
    if not os.path.exists(TELEMETRY_HISTORY_CSV):
        return None
    return pd.read_csv(TELEMETRY_HISTORY_CSV)


def find_unit(fleet_df, unit_id_or_aircraft):
    # Accepts either an exact unit_id (e.g. "AC-014-landing_gear-54") or an aircraft_id
    # (e.g. "AC-014", returning every unit on that aircraft).
    key = unit_id_or_aircraft.strip().upper()
    exact = fleet_df[fleet_df["unit_id"].str.upper() == key]
    if len(exact) > 0:
        return exact
    return fleet_df[fleet_df["aircraft_id"].str.upper() == key]
