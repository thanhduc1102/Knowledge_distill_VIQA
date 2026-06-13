import unittest
import json
import os
import tempfile

from pipeline.evaluate import evaluate_predictions, programs_match
from pipeline.program_executor import execute_program_steps
from pipeline.reward import compute_ecrl_reward, compute_grpo_reward, compute_pcpo_reward


def formatted(program: str, answer: str) -> str:
    return (
        "**Phân tích lập luận:**\n"
        "Tính theo chương trình.\n"
        "**Chương trình tính toán:**\n"
        f"{program}\n"
        "**Đáp án cuối cùng:**\n"
        f"{answer}\n"
    )


class RewardTests(unittest.TestCase):
    def test_execute_program_steps_returns_intermediates(self):
        steps = execute_program_steps("add(1, 2), multiply(#0, const_100)")
        self.assertEqual(steps, [3.0, 300.0])

    def test_pcpo_backward_compatible_reward(self):
        ground_truth = {"program": "add(1, 2)", "answer": "3.0", "table": []}
        reward = compute_pcpo_reward(formatted("add(1, 2)", "3.0"), ground_truth)
        self.assertAlmostEqual(reward, 0.95)

    def test_grpo_dispatcher_defaults_to_pcpo(self):
        ground_truth = {"program": "add(1, 2)", "answer": "3.0", "table": []}
        reward = compute_grpo_reward(formatted("add(1, 2)", "3.0"), ground_truth)
        self.assertAlmostEqual(reward, 0.95)

    def test_ecrl_rewards_symbolically_equivalent_program(self):
        ground_truth = {"program": "add(1, 2)", "answer": "3.0", "table": []}
        reward = compute_ecrl_reward(formatted("add(2, 1)", "3.0"), ground_truth)
        self.assertGreater(reward, 0.90)

    def test_symbolic_program_match_handles_commutativity(self):
        self.assertTrue(programs_match("add(1, 2)", "add(2, 1)"))
        self.assertTrue(programs_match("multiply(3, 4)", "multiply(4, 3)"))
        self.assertFalse(programs_match("subtract(5, 2)", "subtract(2, 5)"))

    def test_ecrl_gives_limited_signal_to_invalid_program(self):
        ground_truth = {"program": "add(1, 2)", "answer": "3.0", "table": []}
        reward = compute_ecrl_reward(formatted("add(1, unknown)", "3.0"), ground_truth)
        self.assertGreater(reward, 0.0)
        self.assertLess(reward, 0.50)

    def test_evaluate_predictions_supports_mixed_benchmark_families(self):
        payload = [
            {
                "id": "finqa-1",
                "benchmark": "finqa",
                "metric_family": "program",
                "predicted_program": "add(1, 2)",
                "predicted_answer": "3.0",
                "gold_program": "add(1, 2)",
                "gold_answer": "3.0",
                "gold_answer_raw": "3.0",
                "gold_steps": [],
            },
            {
                "id": "tatqa-1",
                "benchmark": "tatqa",
                "metric_family": "answer_only",
                "predicted_program": "",
                "predicted_answer": "A | B",
                "gold_program": "",
                "gold_answer": "A | B",
                "gold_answer_raw": ["A", "B"],
                "gold_steps": [],
            },
        ]

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            temp_path = handle.name

        try:
            results = evaluate_predictions(temp_path)
        finally:
            os.unlink(temp_path)

        self.assertEqual(results["total"], 2)
        self.assertAlmostEqual(results["answer_accuracy"], 1.0)
        self.assertAlmostEqual(results["program_accuracy"], 1.0)
        self.assertIn("finqa", results["benchmark_breakdown"])
        self.assertIn("tatqa", results["benchmark_breakdown"])


if __name__ == "__main__":
    unittest.main()