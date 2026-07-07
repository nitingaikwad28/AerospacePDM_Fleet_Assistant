# agent.py
# The conversational layer: figures out what the user is asking (routing), calls the
# right read-only tool (tools.py) to get real numbers, then phrases an answer - using a
# free local LLM if one is reachable, or a deterministic template if not. The numbers
# themselves NEVER come from the LLM; the LLM (when used) only rewords data the tools
# already computed, the same "deterministic core + optional generative explanation
# layer" pattern used throughout this project.
#
# GUARDRAIL: this agent only ever proposes/explains. It has no tool that sends a work
# order, changes a schedule, or writes to any file - see tools.draft_work_order(), which
# is read-only by design. A human still has to act on anything it surfaces.

import re

import llm_client
import tools
from data_access import DataUnavailable

COMPONENTS = ["engine", "landing_gear", "brakes", "bogie"]
UNIT_ID_PATTERN = re.compile(r"\bAC-\d+(?:-[A-Za-z_]+-\d+)?\b", re.IGNORECASE)

WORK_ORDER_WORDS = ("work order", "ticket", "schedule", "dispatch")
SIMILAR_WORDS = ("similar", "before", "seen this", "other aircraft", "pattern", "happened")
PRIORITY_WORDS = ("top", "priority", "urgent", "worst", "attention", "which aircraft",
                   "which unit", "most critical")
SUMMARY_WORDS = ("summary", "overview", "how's the fleet", "how is the fleet", "status", "health")


def _extract_component(message):
    lower = message.lower()
    for c in COMPONENTS:
        if c.replace("_", " ") in lower or c in lower:
            return c
    return None


def _extract_unit_ref(message):
    m = UNIT_ID_PATTERN.search(message)
    return m.group(0) if m else None


def route(message):
    # Returns (intent_name, tool_result_dict). Order matters: a specific unit reference
    # takes priority over generic keyword intents, since "why is AC-014 flagged" should
    # answer about AC-014, not fall through to a generic fleet summary.
    lower = message.lower()
    unit_ref = _extract_unit_ref(message)

    if unit_ref and any(w in lower for w in WORK_ORDER_WORDS):
        return "work_order", tools.draft_work_order(unit_ref)
    if unit_ref:
        return "unit_detail", tools.get_unit_detail(unit_ref)
    if any(w in lower for w in SIMILAR_WORDS):
        component = _extract_component(message)
        if not component:
            return "similar_needs_component", None
        return "similar", tools.find_similar_incidents(component)
    if any(w in lower for w in WORK_ORDER_WORDS):
        return "work_order_needs_unit", None
    if any(w in lower for w in PRIORITY_WORDS):
        return "top_priority", tools.get_top_priority()
    if any(w in lower for w in SUMMARY_WORDS):
        return "summary", tools.get_fleet_summary()
    return "summary", tools.get_fleet_summary()


# ---------------------------------------------------------------- fallback templates
# Used whenever no LLM is reachable - deterministic, always available, no external call.

def _template_summary(data):
    counts = data["status_counts"]
    parts = [f"{v} {k}" for k, v in counts.items()]
    return (
        f"Fleet snapshot: {data['total_units']} tracked parts ({', '.join(parts)}). "
        f"Model benchmarks: RUL error {data['model_rul_mae_cycles']} cycles (MAE), "
        f"anomaly recall {data['model_anomaly_recall']}, mean detection lead time "
        f"{data['model_mean_detection_lead_time_cycles']} cycles."
    )


def _template_top_priority(rows):
    if not rows:
        return "No units are currently flagged Warning or Critical."
    lines = [f"{i+1}. {r['unit_id']} ({r['component']}) - {r['health_status']}, "
             f"predicted RUL {r['predicted_RUL']:.1f} cycles, anomaly score "
             f"{r['anomaly_score']:.3f}" for i, r in enumerate(rows)]
    return "Top priority units:\n" + "\n".join(lines)


def _template_unit_detail(data):
    if not data["found"]:
        return (f"I couldn't find a unit or aircraft matching \"{data['query']}\". "
                f"Try an aircraft_id like AC-014 or a full unit_id.")
    lines = []
    for u in data["units"]:
        line = (f"{u['unit_id']} ({u['component']}, cycle {u['cycle']}): "
                f"{u['health_status']}, predicted RUL {u['predicted_RUL']} cycles, "
                f"anomaly score {u['anomaly_score']}.")
        if u["trend"]:
            movers = [f"{k} {v['direction']}" for k, v in u["trend"].items() if v["direction"] != "flat"]
            if movers:
                line += f" Over the last {u['trend_window_cycles']} cycles: {', '.join(movers)}."
        lines.append(line)
    return "\n".join(lines)


def _template_similar(data):
    if data["similar_flagged_count"] == 0:
        return f"No other {data['component']} units are currently flagged."
    units = ", ".join(u["unit_id"] for u in data["similar_units"])
    return (f"{data['similar_flagged_count']} other {data['component']} unit(s) are "
            f"currently flagged too: {units}.")


def _template_work_order(data):
    if not data["has_draft"]:
        return (f"No work order exists for \"{data['query']}\" - either it's healthy, "
                f"or the ID wasn't recognized.")
    lines = []
    for wo in data["work_orders"]:
        lines.append(f"{wo['work_order_id']}: {wo['priority']} priority, "
                      f"recommended action: {wo['recommended_action']}. "
                      f"(Already drafted by the automated pipeline - not yet sent; "
                      f"this assistant does not send or modify work orders.)")
    return "\n".join(lines)


_TEMPLATES = {
    "summary": _template_summary,
    "top_priority": _template_top_priority,
    "unit_detail": _template_unit_detail,
    "similar": _template_similar,
    "work_order": _template_work_order,
}

_CLARIFICATIONS = {
    "similar_needs_component": (
        "Which component? (engine, landing_gear, brakes, or bogie) - "
        "e.g. \"has this happened before on other engines?\""),
    "work_order_needs_unit": (
        "Which aircraft or unit? e.g. \"work order for AC-014\"."),
}

_LLM_SYSTEM_PREFIX = (
    "You are a concise aerospace fleet maintenance assistant. Using ONLY the data "
    "below, answer the user's question in 2-4 plain-language sentences a maintenance "
    "technician would understand. Do not invent any number that is not present in the "
    "data. If the data shows nothing noteworthy, say so plainly.\n\nDATA:\n{data}\n\n"
    "QUESTION: {question}\n\nANSWER:"
)


def _summarize_for_llm(intent, result):
    # A local, CPU-run model needs meaningfully more time to process a longer PROMPT,
    # separately from how long its OUTPUT is (that's bounded by num_predict in
    # llm_client.py). Multi-unit / multi-work-order results can get long enough to add
    # several seconds of extra prompt-processing time for no real benefit to the answer,
    # so the LLM gets a trimmed version here - the fallback template still always uses
    # the FULL, untrimmed result, so no information is lost when no LLM is available.
    if intent == "unit_detail" and result.get("found") and len(result["units"]) > 1:
        worst_first = sorted(result["units"], key=lambda u: u["predicted_RUL"])
        trimmed = worst_first[0].copy()
        if trimmed.get("trend"):
            trimmed["trend"] = {k: v for k, v in trimmed["trend"].items() if v["direction"] != "flat"}
        return {"found": True, "units": [trimmed],
                "note": f"{len(result['units'])} units matched; showing the most urgent one"}
    if intent == "work_order" and result.get("has_draft") and len(result["work_orders"]) > 1:
        return {"has_draft": True, "work_orders": result["work_orders"][:1],
                "note": f"{len(result['work_orders'])} work orders matched; showing 1"}
    if intent == "similar" and len(result.get("similar_units", [])) > 5:
        trimmed = dict(result)
        trimmed["similar_units"] = result["similar_units"][:5]
        return trimmed
    if intent == "top_priority" and len(result) > 3:
        return result[:3]
    return result


def answer(message, use_llm=True):
    try:
        intent, result = route(message)
    except DataUnavailable as e:
        return {"text": str(e), "source": "error", "intent": "error"}

    if result is None:
        return {"text": _CLARIFICATIONS[intent], "source": "clarification", "intent": intent}

    fallback_text = _TEMPLATES[intent](result)

    if use_llm and llm_client.is_available():
        llm_data = _summarize_for_llm(intent, result)
        prompt = _LLM_SYSTEM_PREFIX.format(data=llm_data, question=message)
        llm_text = llm_client.generate(prompt)
        if llm_text:
            return {"text": llm_text, "source": "llm", "intent": intent, "raw_data": result}

    return {"text": fallback_text, "source": "template", "intent": intent, "raw_data": result}
