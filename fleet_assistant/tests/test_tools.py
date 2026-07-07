# test_tools.py
# Unit tests for tools.py, using small fixture files instead of the real outputs_v2/
# data - fast, deterministic, and doesn't require train.py/predict_service.py to have
# been run first.
#
#   python -m unittest tests.test_tools -v      (run from fleet_assistant/)

import json
import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import data_access
import tools


FLEET_STATUS_ROWS = [
    {"aircraft_id": "AC-001", "unit_id": "AC-001-engine-1", "component": "engine", "cycle": 50,
     "predicted_RUL": 100.0, "anomaly_score": 0.1, "is_anomaly": 0, "health_status": "Healthy"},
    {"aircraft_id": "AC-002", "unit_id": "AC-002-landing_gear-2", "component": "landing_gear",
     "cycle": 80, "predicted_RUL": 20.0, "anomaly_score": 0.7, "is_anomaly": 1, "health_status": "Critical"},
    {"aircraft_id": "AC-003", "unit_id": "AC-003-landing_gear-3", "component": "landing_gear",
     "cycle": 60, "predicted_RUL": 45.0, "anomaly_score": 0.6, "is_anomaly": 1, "health_status": "Warning"},
    {"aircraft_id": "AC-004", "unit_id": "AC-004-brakes-4", "component": "brakes",
     "cycle": 30, "predicted_RUL": 90.0, "anomaly_score": 0.2, "is_anomaly": 0, "health_status": "Healthy"},
]

WORK_ORDERS = [
    {"work_order_id": "WO-AC-002-landing_gear-2-80", "aircraft_id": "AC-002",
     "unit_id": "AC-002-landing_gear-2", "priority": "Critical",
     "recommended_action": "Immediate inspection and maintenance"},
]

MODEL_METRICS = {
    "rul_overall": {"MAE": 5.0, "RMSE": 9.0},
    "anomaly_detection": {"recall": 0.5, "mean_detection_lead_time_cycles": 100.0},
}

TELEMETRY_ROWS = []
for cyc in range(1, 6):
    TELEMETRY_ROWS.append({
        "unit_id": "AC-002-landing_gear-2", "cycle": cyc,
        "vibration_mm_s": 1.0 + cyc * 0.2, "temperature_c": 50 + cyc,
        "pressure_psi": 2000 - cyc * 10, "oil_debris_ppm": 1.0 + cyc * 0.1,
        "rotational_speed_deviation_pct": cyc * 0.05,
    })


class ToolsTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        fleet_csv = os.path.join(self.tmpdir, "fleet_status.csv")
        wo_json = os.path.join(self.tmpdir, "work_orders.json")
        metrics_json = os.path.join(self.tmpdir, "metrics.json")
        telemetry_csv = os.path.join(self.tmpdir, "telemetry.csv")

        pd.DataFrame(FLEET_STATUS_ROWS).to_csv(fleet_csv, index=False)
        with open(wo_json, "w") as f:
            json.dump(WORK_ORDERS, f)
        with open(metrics_json, "w") as f:
            json.dump(MODEL_METRICS, f)
        pd.DataFrame(TELEMETRY_ROWS).to_csv(telemetry_csv, index=False)

        # monkeypatch the already-imported path constants (same pattern the core
        # project's own tests use for monkeypatching functions)
        self._orig = (data_access.FLEET_STATUS_CSV, data_access.WORK_ORDERS_JSON,
                      data_access.MODEL_METRICS_JSON, data_access.TELEMETRY_HISTORY_CSV)
        data_access.FLEET_STATUS_CSV = fleet_csv
        data_access.WORK_ORDERS_JSON = wo_json
        data_access.MODEL_METRICS_JSON = metrics_json
        data_access.TELEMETRY_HISTORY_CSV = telemetry_csv

    def tearDown(self):
        (data_access.FLEET_STATUS_CSV, data_access.WORK_ORDERS_JSON,
         data_access.MODEL_METRICS_JSON, data_access.TELEMETRY_HISTORY_CSV) = self._orig

    def test_get_fleet_summary_counts(self):
        summary = tools.get_fleet_summary()
        self.assertEqual(summary["total_units"], 4)
        self.assertEqual(summary["status_counts"]["Healthy"], 2)
        self.assertEqual(summary["status_counts"]["Critical"], 1)
        self.assertEqual(summary["model_rul_mae_cycles"], 5.0)

    def test_get_top_priority_orders_critical_first(self):
        top = tools.get_top_priority(n=2)
        self.assertEqual(top[0]["health_status"], "Critical")
        self.assertEqual(len(top), 2)

    def test_get_unit_detail_found_with_trend(self):
        detail = tools.get_unit_detail("AC-002-landing_gear-2")
        self.assertTrue(detail["found"])
        unit = detail["units"][0]
        self.assertEqual(unit["health_status"], "Critical")
        self.assertIsNotNone(unit["trend"])
        self.assertEqual(unit["trend"]["vibration_mm_s"]["direction"], "rising")
        self.assertEqual(unit["trend"]["pressure_psi"]["direction"], "falling")

    def test_get_unit_detail_by_aircraft_id(self):
        detail = tools.get_unit_detail("AC-002")
        self.assertTrue(detail["found"])
        self.assertEqual(len(detail["units"]), 1)

    def test_get_unit_detail_not_found(self):
        detail = tools.get_unit_detail("AC-999")
        self.assertFalse(detail["found"])

    def test_find_similar_incidents(self):
        result = tools.find_similar_incidents("landing_gear", exclude_unit_id="AC-002-landing_gear-2")
        self.assertEqual(result["similar_flagged_count"], 1)
        self.assertEqual(result["similar_units"][0]["unit_id"], "AC-003-landing_gear-3")

    def test_draft_work_order_found(self):
        result = tools.draft_work_order("AC-002-landing_gear-2")
        self.assertTrue(result["has_draft"])
        self.assertEqual(result["work_orders"][0]["priority"], "Critical")

    def test_draft_work_order_not_found(self):
        result = tools.draft_work_order("AC-004-brakes-4")   # healthy, no work order exists
        self.assertFalse(result["has_draft"])


if __name__ == "__main__":
    unittest.main()
