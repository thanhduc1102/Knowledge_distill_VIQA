import unittest

from pipeline.evaluate import programs_match
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


if __name__ == "__main__":
    unittest.main()