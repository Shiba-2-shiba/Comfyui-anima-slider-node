from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from anima_slider_node import prompt_util


class PromptUtilTests(unittest.TestCase):
    def test_loads_bundled_prompt_yaml(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-age_slider_v2.yaml")

        self.assertGreater(len(prompts), 0)
        self.assertEqual(prompts[0].action, "enhance")
        self.assertGreater(prompts[0].width, 0)

    def test_defaults_optional_prompts(self):
        prompt = prompt_util._prompt_from_dict({"target": "safe adult portrait"})

        self.assertEqual(prompt.positive, "safe adult portrait")
        self.assertEqual(prompt.neutral, "safe adult portrait")
        self.assertEqual(prompt.unconditional, "")

    def test_validate_prompts_rejects_child_terms_by_default(self):
        prompt = prompt_util.PromptSettings(
            target="safe adult portrait",
            positive="safe older adult portrait",
            unconditional="safe young girl child portrait",
            neutral="safe adult portrait",
        )

        errors = prompt_util.validate_prompts([prompt])

        self.assertIn("unsafe age term", errors[0])


if __name__ == "__main__":
    unittest.main()
