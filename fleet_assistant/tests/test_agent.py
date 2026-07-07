# test_agent.py
# Tests the keyword-based intent router and the deterministic (no-LLM) fallback
# response path - the part of the assistant that must always work, with or without
# Ollama installed.
#
#   python -m unittest tests.test_agent -v      (run from fleet_assistant/)

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import agent


class RouterTestCase(unittest.TestCase):
    def test_unit_reference_routes_to_unit_detail(self):
        intent, _ = agent.route("why is AC-014 flagged")
        self.assertEqual(intent, "unit_detail")

    def test_extract_unit_ref_matches_aircraft_id(self):
        self.assertEqual(agent._extract_unit_ref("what about AC-014?"), "AC-014")

    def test_extract_unit_ref_matches_full_unit_id(self):
        ref = agent._extract_unit_ref("tell me about AC-014-landing_gear-54 please")
        self.assertEqual(ref, "AC-014-landing_gear-54")

    def test_extract_unit_ref_none_when_absent(self):
        self.assertIsNone(agent._extract_unit_ref("give me a fleet summary"))

    def test_extract_component_detects_landing_gear(self):
        self.assertEqual(agent._extract_component("other landing_gear units"), "landing_gear")

    def test_priority_keywords_route_to_top_priority(self):
        intent, result = agent.route("what needs attention this week")
        self.assertEqual(intent, "top_priority")

    def test_similar_without_component_asks_for_clarification(self):
        intent, result = agent.route("has this happened before")
        self.assertEqual(intent, "similar_needs_component")
        self.assertIsNone(result)

    def test_work_order_without_unit_asks_for_clarification(self):
        intent, result = agent.route("create a work order")
        self.assertEqual(intent, "work_order_needs_unit")
        self.assertIsNone(result)

    def test_default_routes_to_summary(self):
        intent, _ = agent.route("hello")
        self.assertEqual(intent, "summary")


class AnswerFallbackTestCase(unittest.TestCase):
    def test_clarification_answer_has_clarification_source(self):
        result = agent.answer("has this happened before", use_llm=False)
        self.assertEqual(result["source"], "clarification")

    def test_answer_never_calls_llm_when_disabled(self):
        # use_llm=False must short-circuit before any Ollama reachability check -
        # this test would hang/fail if that guarantee were broken on a machine
        # without Ollama installed.
        result = agent.answer("give me a fleet summary", use_llm=False)
        self.assertIn(result["source"], ("template", "error"))


if __name__ == "__main__":
    unittest.main()
