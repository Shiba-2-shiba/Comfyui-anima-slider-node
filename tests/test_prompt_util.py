from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from anima_slider_node import prompt_util


class PromptUtilTests(unittest.TestCase):
    def _prompt(self, target: str, width: int, height: int):
        return prompt_util.PromptSettings(
            target=target,
            positive=target,
            unconditional="",
            neutral=target,
            width=width,
            height=height,
        )

    def test_loads_bundled_prompt_yaml(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-age_slider_fullbody_v2.yaml")

        self.assertGreater(len(prompts), 0)
        self.assertEqual(prompts[0].action, "enhance")
        self.assertGreater(prompts[0].width, 0)

    def test_defaults_optional_prompts(self):
        prompt = prompt_util._prompt_from_dict({"target": "safe adult portrait"})

        self.assertEqual(prompt.positive, "safe adult portrait")
        self.assertEqual(prompt.neutral, "safe adult portrait")
        self.assertEqual(prompt.unconditional, "")

    def test_resolve_training_resolution_preserves_explicit_node_values(self):
        prompts = [
            self._prompt("safe adult portrait", width=896, height=1152),
        ]

        resolution = prompt_util.resolve_training_resolution(prompts, [0], width=512, height=512)

        self.assertEqual(resolution, (512, 512))

    def test_resolve_training_resolution_uses_prompt_yaml_when_node_values_are_zero(self):
        prompts = [
            self._prompt("safe adult portrait", width=896, height=1152),
            self._prompt("safe adult full body", width=896, height=1152),
        ]

        resolution = prompt_util.resolve_training_resolution(prompts, [0, 1], width=0, height=0)

        self.assertEqual(resolution, (896, 1152))

    def test_resolve_training_resolution_rejects_mixed_yaml_resolutions(self):
        prompts = [
            self._prompt("safe adult portrait", width=896, height=1152),
            self._prompt("safe adult full body", width=1024, height=1024),
        ]

        with self.assertRaisesRegex(ValueError, "resolutions differ"):
            prompt_util.resolve_training_resolution(prompts, [0, 1], width=0, height=0)

    def test_age_fullbody_v3_keeps_body_framing_in_every_role(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-age_slider_fullbody_v3.yaml")

        for prompt in prompts:
            for text in (prompt.target, prompt.positive, prompt.unconditional, prompt.neutral):
                self.assertIn("uncropped full body", text)
                self.assertIn("head-to-toe visible", text)
                self.assertIn("floor visible", text)

    def test_age_fullbody_v3_positive_avoids_horror_aging_terms(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-age_slider_fullbody_v3.yaml")
        rejected_terms = {
            "very elderly",
            "80 years old",
            "deeply wrinkled",
            "sagging jowls",
            "hollow cheeks",
            "frail elderly build",
        }

        for prompt in prompts:
            positive = prompt.positive.lower()
            for term in rejected_terms:
                self.assertNotIn(term, positive)

    def test_age_fullbody_v4_preserves_bright_background_in_every_role(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-age_slider_fullbody_v4.yaml")

        for prompt in prompts:
            for text in (prompt.target, prompt.positive, prompt.unconditional, prompt.neutral):
                self.assertIn("uncropped full body", text)
                self.assertIn("head-to-toe visible", text)
                self.assertIn("bright floor", text)
                self.assertIn("no dark background", text)
                self.assertIn("background unchanged", text)

    def test_age_fullbody_v4_uses_intentionally_strong_age_terms(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-age_slider_fullbody_v4.yaml")

        errors = prompt_util.validate_prompts(prompts)

        self.assertTrue(errors)
        self.assertFalse(prompt_util.validate_prompts(prompts, allow_unsafe_age_terms=True))
        for prompt in prompts:
            positive = prompt.positive.lower()
            unconditional = prompt.unconditional.lower()
            self.assertIn("very elderly woman", positive)
            self.assertIn("deep facial wrinkles", positive)
            self.assertIn("young girl", unconditional)
            self.assertIn("toddler", unconditional)

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
