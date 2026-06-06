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

    def _assert_fixed_phase2_context(self, prompts: list[prompt_util.PromptSettings]):
        self.assertEqual(len(prompts), 8)
        self.assertFalse(prompt_util.validate_prompts(prompts))
        for prompt in prompts:
            self.assertEqual(prompt.neutral, prompt.target)
            for text in (prompt.target, prompt.positive, prompt.unconditional, prompt.neutral):
                self.assertIn("adult woman", text)
                self.assertIn("fully clothed", text)
                self.assertIn("nonsexual", text)
                self.assertIn("solo", text)
                self.assertIn("dark brown hair", text)
                self.assertIn("neutral colors", text)
                self.assertIn("clean bright neutral studio", text)
                self.assertIn("no dark background", text)
                self.assertIn("background unchanged", text)
                self.assertIn("pose unchanged", text)

    def _assert_terms_absent(self, text: str, terms: set[str]):
        lowered = text.lower()
        for term in terms:
            self.assertNotIn(term, lowered)

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

        for index, prompt in enumerate(prompts):
            for text in (prompt.target, prompt.positive, prompt.unconditional, prompt.neutral):
                if index in {4, 5}:
                    self.assertIn("upper body", text)
                    self.assertIn("face clearly visible", text)
                else:
                    self.assertIn("uncropped full body", text)
                    self.assertIn("head-to-toe visible", text)
                    self.assertIn("bright floor", text)
                self.assertIn("no dark background", text)
                self.assertIn("background unchanged", text)
                self.assertIn("dark brown hair", text)
                self.assertIn("neutral colors", text)

    def test_age_fullbody_v4_uses_intentionally_strong_age_terms(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-age_slider_fullbody_v4.yaml")

        errors = prompt_util.validate_prompts(prompts)

        self.assertTrue(errors)
        self.assertFalse(prompt_util.validate_prompts(prompts, allow_unsafe_age_terms=True))
        for prompt in prompts:
            positive = prompt.positive.lower()
            unconditional = prompt.unconditional.lower()
            self.assertIn("very elderly woman", positive)
            self.assertIn("feminine elderly face", positive)
            self.assertIn("elderly female face", positive)
            self.assertIn("deep facial wrinkles", positive)
            self.assertIn("elderly feminine body shape", positive)
            self.assertIn("thin elderly arms", positive)
            self.assertIn("young girl", unconditional)
            self.assertIn("toddler", unconditional)
            self.assertIn("petite youthful build", unconditional)
            self.assertIn("slim youthful arms", unconditional)
            self.assertNotIn("chibi", unconditional)

    def test_age_fullbody_v4_avoids_color_axis_terms(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-age_slider_fullbody_v4.yaml")
        rejected_terms = {"pink hair", "red hair", "blonde hair", "silver hair", "gray hair", "white hair"}

        for prompt in prompts:
            joined = " ".join([prompt.target, prompt.positive, prompt.unconditional, prompt.neutral]).lower()
            for term in rejected_terms:
                self.assertNotIn(term, joined)

    def test_mature_face_slider_keeps_face_axis_clean(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-mature_face_slider.yaml")

        self.assertEqual(len(prompts), 8)
        self.assertFalse(prompt_util.validate_prompts(prompts, allow_unsafe_age_terms=True))
        for index, prompt in enumerate(prompts):
            self.assertEqual(prompt.neutral, prompt.target)
            for text in (prompt.target, prompt.positive, prompt.unconditional, prompt.neutral):
                self.assertIn("dark brown hair", text)
                self.assertIn("neutral colors", text)
                self.assertIn("background unchanged", text)
                if index in {4, 5}:
                    self.assertIn("upper body", text)
                    self.assertIn("face clearly visible", text)
                else:
                    self.assertIn("portrait", text)
                    self.assertIn("readable face", text)
            positive = prompt.positive.lower()
            unconditional = prompt.unconditional.lower()
            self.assertIn("mature adult feminine face", positive)
            self.assertIn("defined cheekbones", positive)
            self.assertIn("young girl", unconditional)
            self.assertIn("soft youthful facial features", unconditional)
            self.assertNotIn("chibi", unconditional)

    def test_face_roundness_slider_keeps_shape_axis_clean(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-face_roundness_slider.yaml")

        self.assertEqual(len(prompts), 8)
        self.assertFalse(prompt_util.validate_prompts(prompts))
        for index, prompt in enumerate(prompts):
            self.assertEqual(prompt.neutral, prompt.target)
            for text in (prompt.target, prompt.positive, prompt.unconditional, prompt.neutral):
                self.assertIn("dark brown hair", text)
                self.assertIn("neutral colors", text)
                self.assertIn("background unchanged", text)
                if index in {4, 5}:
                    self.assertIn("upper body", text)
                    self.assertIn("face clearly visible", text)
                else:
                    self.assertIn("portrait", text)
                    self.assertIn("readable face", text)
            positive = prompt.positive.lower()
            unconditional = prompt.unconditional.lower()
            self.assertIn("round face", positive)
            self.assertIn("soft cheeks", positive)
            self.assertIn("angular face", unconditional)
            self.assertIn("sharp jawline", unconditional)
            for term in {"young girl", "child", "chibi"}:
                self.assertNotIn(term, positive)
                self.assertNotIn(term, unconditional)

    def test_eye_openness_slider_keeps_expression_axis_clean(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-eye_openness_slider.yaml")

        self.assertEqual(len(prompts), 8)
        self.assertFalse(prompt_util.validate_prompts(prompts))
        for index, prompt in enumerate(prompts):
            self.assertEqual(prompt.neutral, prompt.target)
            for text in (prompt.target, prompt.positive, prompt.unconditional, prompt.neutral):
                self.assertIn("dark brown hair", text)
                self.assertIn("neutral colors", text)
                self.assertIn("background unchanged", text)
                self.assertIn("neutral expression", text)
                if index in {4, 5}:
                    self.assertIn("upper body", text)
                    self.assertIn("face clearly visible", text)
                else:
                    self.assertIn("portrait", text)
                    self.assertIn("readable face", text)
            positive = prompt.positive.lower()
            unconditional = prompt.unconditional.lower()
            self.assertIn("wide open eyes", positive)
            self.assertIn("raised eyelids", positive)
            self.assertIn("relaxed half-open eyes", unconditional)
            self.assertIn("lowered eyelids", unconditional)
            for term in {"surprised", "sleepy", "angry", "blush", "pink hair", "red hair", "blonde hair"}:
                self.assertNotIn(term, positive)
                self.assertNotIn(term, unconditional)

    def test_smile_intensity_slider_keeps_mouth_axis_clean(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-smile_intensity_slider.yaml")

        self.assertEqual(len(prompts), 8)
        self.assertFalse(prompt_util.validate_prompts(prompts))
        for index, prompt in enumerate(prompts):
            self.assertEqual(prompt.neutral, prompt.target)
            for text in (prompt.target, prompt.positive, prompt.unconditional, prompt.neutral):
                self.assertIn("dark brown hair", text)
                self.assertIn("neutral colors", text)
                self.assertIn("background unchanged", text)
                self.assertIn("calm neutral eyes", text)
                if index in {4, 5}:
                    self.assertIn("upper body", text)
                    self.assertIn("face clearly visible", text)
                else:
                    self.assertIn("portrait", text)
                    self.assertIn("readable face", text)
            positive = prompt.positive.lower()
            unconditional = prompt.unconditional.lower()
            self.assertIn("clear smiling mouth", positive)
            self.assertIn("upturned mouth corners", positive)
            self.assertIn("relaxed closed mouth", unconditional)
            self.assertIn("straight mouth line", unconditional)
            for term in {"cheerful", "happy", "joyful", "blush", "pink hair", "red hair", "blonde hair"}:
                self.assertNotIn(term, positive)
                self.assertNotIn(term, unconditional)

    def test_blush_intensity_slider_keeps_cheek_color_axis_clean(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-blush_intensity_slider.yaml")

        self.assertEqual(len(prompts), 8)
        self.assertFalse(prompt_util.validate_prompts(prompts))
        for index, prompt in enumerate(prompts):
            self.assertEqual(prompt.neutral, prompt.target)
            for text in (prompt.target, prompt.positive, prompt.unconditional, prompt.neutral):
                self.assertIn("dark brown hair", text)
                self.assertIn("neutral colors", text)
                self.assertIn("background unchanged", text)
                self.assertIn("overall palette unchanged", text)
                self.assertIn("neutral expression", text)
                if index in {4, 5}:
                    self.assertIn("upper body", text)
                    self.assertIn("face clearly visible", text)
                else:
                    self.assertIn("portrait", text)
                    self.assertIn("readable face", text)
            positive = prompt.positive.lower()
            unconditional = prompt.unconditional.lower()
            self.assertIn("localized cheek blush", positive)
            self.assertIn("blush limited to cheeks", positive)
            self.assertIn("no cheek blush", unconditional)
            self.assertIn("unflushed cheeks", unconditional)
            for term in {"embarrassed", "shy", "pink hair", "red hair", "blonde hair", "pink background", "pink clothes"}:
                self.assertNotIn(term, positive)
                self.assertNotIn(term, unconditional)

    def test_emotion_valence_slider_keeps_expression_context_clean(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-emotion_valence_slider.yaml")

        self.assertEqual(len(prompts), 8)
        self.assertFalse(prompt_util.validate_prompts(prompts))
        for index, prompt in enumerate(prompts):
            self.assertEqual(prompt.neutral, prompt.target)
            for text in (prompt.target, prompt.positive, prompt.unconditional, prompt.neutral):
                self.assertIn("dark brown hair", text)
                self.assertIn("neutral colors", text)
                self.assertIn("background unchanged", text)
                self.assertIn("pose unchanged", text)
                if index in {4, 5}:
                    self.assertIn("upper body", text)
                    self.assertIn("face clearly visible", text)
                else:
                    self.assertIn("portrait", text)
                    self.assertIn("readable face", text)
            positive = prompt.positive.lower()
            unconditional = prompt.unconditional.lower()
            self.assertIn("bright happy expression", positive)
            self.assertIn("cheerful facial expression", positive)
            self.assertIn("gloomy expression", unconditional)
            self.assertIn("melancholic facial expression", unconditional)
            for term in {"crying", "tears", "angry", "rage", "black background", "dim lighting", "warm lighting", "pink hair", "red hair", "blonde hair"}:
                self.assertNotIn(term, positive)
                self.assertNotIn(term, unconditional)

    def test_breast_size_slider_keeps_body_axis_clothed_and_nonsexual(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-breast_size_slider.yaml")
        rejected_terms = {"cleavage", "nude", "bikini", "lingerie", "seductive", "revealing", "pink hair", "red hair", "blonde hair"}

        self._assert_fixed_phase2_context(prompts)
        for prompt in prompts:
            joined = " ".join([prompt.target, prompt.positive, prompt.unconditional, prompt.neutral])
            self.assertIn("covered torso visible", joined)
            self.assertIn("opaque modest top", joined)
            self.assertIn("larger covered bust silhouette", prompt.positive.lower())
            self.assertIn("fuller bust volume under clothing", prompt.positive.lower())
            self.assertIn("smaller covered bust silhouette", prompt.unconditional.lower())
            self.assertIn("flatter chest volume under clothing", prompt.unconditional.lower())
            self._assert_terms_absent(joined, rejected_terms)

    def test_waist_width_slider_keeps_waist_axis_clean(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-waist_width_slider.yaml")
        rejected_terms = {"pink hair", "red hair", "blonde hair", "child", "young girl", "cleavage", "seductive"}

        self._assert_fixed_phase2_context(prompts)
        for prompt in prompts:
            joined = " ".join([prompt.target, prompt.positive, prompt.unconditional, prompt.neutral])
            self.assertIn("waist visible", joined)
            self.assertIn("base outfit unchanged", joined)
            self.assertIn("narrow waist", prompt.positive.lower())
            self.assertIn("cinched waistline", prompt.positive.lower())
            self.assertIn("wide waist", prompt.unconditional.lower())
            self.assertIn("straight torso silhouette", prompt.unconditional.lower())
            self._assert_terms_absent(joined, rejected_terms)

    def test_skirt_length_slider_keeps_hemline_axis_clean(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-skirt_length_slider.yaml")
        rejected_terms = {"pink hair", "red hair", "blonde hair", "school uniform", "student", "revealing", "seductive"}

        self._assert_fixed_phase2_context(prompts)
        for prompt in prompts:
            joined = " ".join([prompt.target, prompt.positive, prompt.unconditional, prompt.neutral])
            self.assertIn("uncropped full body", joined)
            self.assertIn("head-to-toe visible", joined)
            self.assertIn("same skirt outfit", joined)
            self.assertIn("ankle-length skirt", prompt.positive.lower())
            self.assertIn("hem near ankles", prompt.positive.lower())
            self.assertIn("knee-length skirt", prompt.unconditional.lower())
            self.assertIn("hem near knees", prompt.unconditional.lower())
            self._assert_terms_absent(joined, rejected_terms)

    def test_clothing_fit_slider_keeps_fit_axis_clean(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-clothing_fit_slider.yaml")
        rejected_terms = {"pink hair", "red hair", "blonde hair", "cleavage", "seductive", "revealing"}

        self._assert_fixed_phase2_context(prompts)
        for prompt in prompts:
            joined = " ".join([prompt.target, prompt.positive, prompt.unconditional, prompt.neutral])
            self.assertIn("body shape unchanged", joined)
            self.assertIn("same garment type", joined)
            self.assertIn("closely fitted clothing", prompt.positive.lower())
            self.assertIn("tailored fit", prompt.positive.lower())
            self.assertIn("loose clothing fit", prompt.unconditional.lower())
            self.assertIn("extra fabric folds", prompt.unconditional.lower())
            self._assert_terms_absent(joined, rejected_terms)

    def test_collar_height_slider_keeps_neckline_axis_clean(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-collar_height_slider.yaml")
        rejected_terms = {"pink hair", "red hair", "blonde hair", "cleavage", "seductive", "revealing"}

        self._assert_fixed_phase2_context(prompts)
        for prompt in prompts:
            joined = " ".join([prompt.target, prompt.positive, prompt.unconditional, prompt.neutral])
            self.assertIn("same blouse garment", joined)
            self.assertIn("collar area visible", joined)
            self.assertIn("high collar", prompt.positive.lower())
            self.assertIn("neck-covering collar", prompt.positive.lower())
            self.assertIn("low collar", prompt.unconditional.lower())
            self.assertIn("open neckline", prompt.unconditional.lower())
            self._assert_terms_absent(joined, rejected_terms)

    def test_outfit_decoration_slider_keeps_decoration_axis_clean(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-outfit_decoration_slider.yaml")
        rejected_terms = {"pink hair", "red hair", "blonde hair", "gold background", "glowing", "sparkle background"}

        self._assert_fixed_phase2_context(prompts)
        for prompt in prompts:
            joined = " ".join([prompt.target, prompt.positive, prompt.unconditional, prompt.neutral])
            self.assertIn("outfit clearly visible", joined)
            self.assertIn("base outfit unchanged", joined)
            self.assertIn("ornate outfit decoration", prompt.positive.lower())
            self.assertIn("decorative embroidery", prompt.positive.lower())
            self.assertIn("plain outfit decoration", prompt.unconditional.lower())
            self.assertIn("minimal trim", prompt.unconditional.lower())
            self._assert_terms_absent(joined, rejected_terms)

    def test_anger_intensity_slider_keeps_face_tension_axis_clean(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-anger_intensity_slider.yaml")
        rejected_terms = {"pink hair", "red hair", "blonde hair", "fire", "aura", "action pose", "black background", "dim lighting"}

        self._assert_fixed_phase2_context(prompts)
        for prompt in prompts:
            joined = " ".join([prompt.target, prompt.positive, prompt.unconditional, prompt.neutral])
            self.assertIn("face clearly visible", joined)
            self.assertIn("facial expression only", joined)
            self.assertIn("angry expression", prompt.positive.lower())
            self.assertIn("furrowed eyebrows", prompt.positive.lower())
            self.assertIn("tense mouth", prompt.positive.lower())
            self.assertIn("calm expression", prompt.unconditional.lower())
            self.assertIn("relaxed eyebrows", prompt.unconditional.lower())
            self._assert_terms_absent(joined, rejected_terms)

    def test_hair_messiness_slider_keeps_hair_identity_clean(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-hair_messiness_slider.yaml")
        rejected_terms = {"pink hair", "red hair", "blonde hair", "silver hair", "short hair", "long hair", "wet hair", "wind"}

        self._assert_fixed_phase2_context(prompts)
        for prompt in prompts:
            joined = " ".join([prompt.target, prompt.positive, prompt.unconditional, prompt.neutral])
            self.assertIn("same hair length", joined)
            self.assertIn("same hairstyle", joined)
            self.assertIn("medium hair", joined)
            self.assertIn("messy hair", prompt.positive.lower())
            self.assertIn("loose stray strands", prompt.positive.lower())
            self.assertIn("neat hair", prompt.unconditional.lower())
            self.assertIn("controlled strands", prompt.unconditional.lower())
            self._assert_terms_absent(joined, rejected_terms)

    def test_pose_dynamism_slider_keeps_subject_and_background_clean(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-pose_dynamism_slider.yaml")
        rejected_terms = {"pink hair", "red hair", "blonde hair", "motion blur", "speed lines", "battle", "weapon"}

        self.assertEqual(len(prompts), 8)
        self.assertFalse(prompt_util.validate_prompts(prompts))
        for prompt in prompts:
            self.assertEqual(prompt.neutral, prompt.target)
            joined = " ".join([prompt.target, prompt.positive, prompt.unconditional, prompt.neutral])
            for text in (prompt.target, prompt.positive, prompt.unconditional, prompt.neutral):
                self.assertIn("adult woman", text)
                self.assertIn("fully clothed", text)
                self.assertIn("nonsexual", text)
                self.assertIn("solo", text)
                self.assertIn("dark brown hair", text)
                self.assertIn("neutral colors", text)
                self.assertIn("clean bright neutral studio", text)
                self.assertIn("no dark background", text)
                self.assertIn("background unchanged", text)
            self.assertIn("uncropped full body", joined)
            self.assertIn("same outfit", joined)
            self.assertIn("arms and legs visible", joined)
            self.assertIn("dynamic pose", prompt.positive.lower())
            self.assertIn("energetic body movement", prompt.positive.lower())
            self.assertIn("relaxed standing pose", prompt.unconditional.lower())
            self.assertIn("minimal gesture", prompt.unconditional.lower())
            self._assert_terms_absent(joined, rejected_terms)

    def test_posture_uprightness_slider_keeps_age_and_body_axis_clean(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-posture_uprightness_slider.yaml")
        rejected_terms = {"pink hair", "red hair", "blonde hair", "old", "young girl", "child", "confident", "timid"}

        self._assert_fixed_phase2_context(prompts)
        for prompt in prompts:
            joined = " ".join([prompt.target, prompt.positive, prompt.unconditional, prompt.neutral])
            self.assertIn("adult body unchanged", joined)
            self.assertIn("same standing pose base", joined)
            self.assertIn("upright posture", prompt.positive.lower())
            self.assertIn("straight back", prompt.positive.lower())
            self.assertIn("slouched posture", prompt.unconditional.lower())
            self.assertIn("hunched shoulders", prompt.unconditional.lower())
            self._assert_terms_absent(joined, rejected_terms)

    def test_pastel_tone_slider_keeps_subject_and_composition_clean(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-pastel_tone_slider.yaml")
        rejected_terms = {"pink hair", "red hair", "blonde hair", "silver hair", "glow", "dreamy background", "soft focus"}

        self.assertEqual(len(prompts), 8)
        self.assertFalse(prompt_util.validate_prompts(prompts))
        for prompt in prompts:
            self.assertEqual(prompt.neutral, prompt.target)
            joined = " ".join([prompt.target, prompt.positive, prompt.unconditional, prompt.neutral])
            for text in (prompt.target, prompt.positive, prompt.unconditional, prompt.neutral):
                self.assertIn("adult woman", text)
                self.assertIn("dark brown hair", text)
                self.assertIn("same subject", text)
                self.assertIn("same composition", text)
                self.assertIn("background unchanged", text)
            self.assertIn("pale pastel colors", prompt.positive.lower())
            self.assertIn("low saturation palette", prompt.positive.lower())
            self.assertIn("vivid saturated colors", prompt.unconditional.lower())
            self.assertIn("high saturation palette", prompt.unconditional.lower())
            self._assert_terms_absent(joined, rejected_terms)

    def test_line_weight_slider_keeps_line_axis_clean(self):
        prompts = prompt_util.load_prompts_from_yaml(REPO_ROOT / "prompts" / "prompts-anima-line_weight_slider.yaml")
        rejected_terms = {"pink hair", "red hair", "blonde hair", "silver hair", "sketch", "monochrome", "detail density"}

        self._assert_fixed_phase2_context(prompts)
        for prompt in prompts:
            joined = " ".join([prompt.target, prompt.positive, prompt.unconditional, prompt.neutral])
            self.assertIn("same color palette", joined)
            self.assertIn("same detail level", joined)
            self.assertIn("bold outlines", prompt.positive.lower())
            self.assertIn("thick lineart", prompt.positive.lower())
            self.assertIn("thin lineart", prompt.unconditional.lower())
            self.assertIn("fine contour lines", prompt.unconditional.lower())
            self._assert_terms_absent(joined, rejected_terms)

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
