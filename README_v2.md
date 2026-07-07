# Aerospace Predictive Maintenance - v2 (scikit-learn Production Architecture)

A production-oriented evolution of the original hackathon MVP, now built on real
**scikit-learn** ensemble algorithms: `IsolationForest` for anomaly detection and
`RandomForestRegressor` for Remaining Useful Life (RUL). An earlier pass at v2 used
from-scratch numpy implementations of these algorithms (no internet access was
available to install scikit-learn at the time); that code has been **deleted** now
that scikit-learn is available, so only this version remains.

This repository now contains only the v2 pipeline; the original from-scratch v1
MVP has been removed. See `outputs_v2/code_documentation_v2.pdf` for full
documentation of this version, `outputs_v2/code_documentation_v3.pdf` for a
from-scratch, beginner-friendly guide (Python + ML + aerospace concepts explained),
and `outputs_v2/FleetGuard_AI_Architecture.pdf` for the train/serve pipeline diagram.

## Fleet Assistant (new)

This project also includes a **conversational Fleet Assistant** (`fleet_assistant/`)
layered on top of the core pipeline above - ask it things like *"why is AC-014
flagged?"* or *"what needs attention this week?"* in plain English, from a terminal
or a simple web chat page, instead of reading the dashboard table by hand.

- It's **read-only**: it only reads the core pipeline's existing output files
  (`fleet_status_v2.csv`, `work_orders_v2.json`, `model_metrics_v2.json`,
  `data_v2/fleet_telemetry_v2.csv`) and never sends, schedules, or modifies anything.
- It works **fully offline with zero API cost** using deterministic templates, and
  gets nicer natural-language phrasing for free if a local [Ollama](https://ollama.com)
  model is installed - never a hard requirement. On CPU-only hardware, expect ~3-10s
  per LLM-phrased reply (measured ~10 tokens/second); template-mode answers are
  instant. See `outputs_v2/Fleet_Assistant_Documentation.pdf` section 6 for the full
  performance-tuning writeup.
- It does **not** import any code from `src_v2/` - a decoupled add-on, not a rewrite,
  the same design principle as this project's own train/serve split.

See `outputs_v2/Fleet_Assistant_Documentation.pdf` for the full architecture,
a sequence diagram of a conversation turn, and known limitations. See "Steps to Run
version 2.txt" for how to start it.

## Verified end-to-end

This pipeline has been run end-to-end with scikit-learn 1.9 installed: training,
serving, and the unit test suite all complete successfully.

```bash
pip install -r requirements_v2.txt
cd src_v2
python train.py
python predict_service.py
python -m unittest tests.test_models -v
```

Latest verified run: test-set anomaly detection recall=0.667, early-life false
alarm rate=0.082; RUL regression overall MAE=4.91, RMSE=9.42. All 5 unit tests
pass. See `outputs_v2/model_metrics_v2.json` for the full benchmark report.

## What's in src_v2/ now

| File | Role |
|---|---|
| `config.py` | Central hyperparameters (now scikit-learn constructor arguments) and paths |
| `data_simulator_v2.py` | 40-aircraft, 160-part, 5-sensor synthetic fleet + train/val/test unit split (unchanged from before) |
| `features.py` | 31 rolling-window engineered features (unchanged from before) |
| `anomaly_detection_v2.py` | Trains/scores one `sklearn.ensemble.IsolationForest` per component |
| `rul_model_v2.py` | Trains/predicts with one `sklearn.ensemble.RandomForestRegressor` per component |
| `evaluation_v2.py` | Detection lead time, false alarm rate, per-component RUL accuracy (unchanged) |
| `model_registry.py` | Saves/loads models via `joblib` + a JSON model card |
| `dashboard_v2.py` | Fleet health dashboard + model-performance panel (unchanged) |
| `mro_integration_v2.py` | Mock MRO adapter with idempotency key + retry/backoff (unchanged) |
| `train.py` | Offline training entry point |
| `predict_service.py` | Production serving entry point (loads saved models, no retraining) |
| `tests/test_models.py` | Unit tests for the scikit-learn-backed wrapper functions |

## What's in fleet_assistant/

| File | Role |
|---|---|
| `config.py` | Paths to the core pipeline's output files; optional LLM settings |
| `data_access.py` | Loads those output files into DataFrames/dicts |
| `tools.py` | Read-only functions: fleet summary, top priority, unit detail + trend, similar incidents, work-order lookup |
| `agent.py` | Keyword/regex intent routing + response generation (LLM-phrased if available, template fallback otherwise) |
| `llm_client.py` | Optional call to a free, local Ollama model; returns `None` on any failure so the agent always has a fallback |
| `chat_cli.py` | Terminal chat interface |
| `chat_web.py` + `static/chat.html` | A one-page Flask web chat UI |
| `tests/test_tools.py`, `tests/test_agent.py` | 19 unit tests against fixture data (no Ollama required) |

## What changed vs. the from-scratch v2

| Aspect | From-scratch v2 (deleted) | This version |
|---|---|---|
| Anomaly detection | Custom `_IsolationTree` / `IsolationForest` classes in numpy | `sklearn.ensemble.IsolationForest` |
| RUL estimation | Custom `_DecisionTreeRegressor` / `RandomForestRegressor` classes in numpy | `sklearn.ensemble.RandomForestRegressor` |
| Isolation Forest size | 100 trees | 200 trees |
| Random Forest size | 25 trees, max_depth=6 | 300 trees, max_depth=12 |
| Model persistence | Raw `pickle`, `.pkl` files | `joblib`, `.joblib` files |
| Feature importance | Custom split-count proxy | Built-in `model.feature_importances_` |

## Why scikit-learn

Battle-tested and optimized (Cython/C, parallelized with `n_jobs=-1`), which makes
much larger ensembles (200-300 trees vs. 25-100) practical; full ecosystem
compatibility (joblib, hyperparameter search, model inspection tools) with no
extra code; and significantly less custom code to maintain long-term. See
`outputs_v2/code_documentation_v2.pdf` for the full reasoning and a complete
file-by-file reference.
