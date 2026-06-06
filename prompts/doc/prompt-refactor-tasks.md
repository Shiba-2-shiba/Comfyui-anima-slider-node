# Prompt Refactor Task List

This task list is the execution checklist for prompt refactoring.
Work top to bottom unless a later item is explicitly prioritized.

## Global Setup

- [x] Create prompt refactor specification.
- [x] Create progress tracker.
- [x] Create category task list.
- [x] Add shared prompt-audit helper tests or utilities if repeated checks become noisy.

## Phase 1: Face And Expression Prompt Sets

Goal: apply Age slider lessons to face/expression sliders.

Age-slider note: when editing age-related files, assume `allow_unsafe_age_terms=True`
is available. Do not weaken old/young wording or shrink the age difference unless
an observed failure points to a specific term. Remove only terms that introduce a
separate unwanted axis, such as style or chibi drift.

### 1. Mature Face

- [x] Inspect `prompts-anima-mature_face_slider.yaml`.
- [x] Identify current nuisance axes: hair color, camera, background, age, body.
- [x] Add tests for mature-face invariants.
- [x] Refactor prompt text.
- [x] Run `python -m pytest tests/test_prompt_util.py`.
- [x] Run `python -m pytest`.
- [x] Update `prompt-refactor-progress.md`.

### 2. Face Roundness

- [x] Inspect `prompts-anima-face_roundness_slider.yaml`.
- [x] Add tests that prevent age/chibi/body drift.
- [x] Refactor prompt text.
- [x] Verify tests.
- [x] Update progress.

### 3. Eye Openness

- [x] Inspect `prompts-anima-eye_openness_slider.yaml`.
- [x] Add tests that keep expression/mood stable except eye openness.
- [x] Refactor prompt text.
- [x] Verify tests.
- [x] Update progress.

### 4. Smile Intensity

- [x] Inspect `prompts-anima-smile_intensity_slider.yaml`.
- [x] Add tests that prevent blush/lighting/background drift.
- [x] Refactor prompt text.
- [x] Verify tests.
- [x] Update progress.

### 5. Blush Intensity

- [x] Inspect `prompts-anima-blush_intensity_slider.yaml`.
- [x] Add tests that isolate cheek redness from global pink palette.
- [x] Refactor prompt text.
- [x] Verify tests.
- [x] Update progress.

### 6. Emotion Valence

- [x] Inspect `prompts-anima-emotion_valence_slider.yaml`.
- [x] Define allowed emotion-axis changes.
- [x] Add tests for fixed pose/background/palette.
- [x] Refactor prompt text.
- [x] Verify tests.
- [x] Update progress.

## Phase 2: Body And Clothing Prompt Sets

Goal: isolate body/clothing silhouette while preserving adult, fully clothed, nonsexual context.

### 7. Breast Size

- [x] Inspect `prompts-anima-breast_size_slider.yaml`.
- [x] Add tests for `adult`, `fully clothed`, and `nonsexual` wording.
- [x] Refactor prompt text to keep outfit/camera stable.
- [x] Verify tests.
- [x] Update progress.

### 8. Waist Width

- [x] Inspect `prompts-anima-waist_width_slider.yaml`.
- [x] Add tests that isolate waist silhouette from body identity and clothing style.
- [x] Refactor prompt text.
- [x] Verify tests.
- [x] Update progress.

### 9. Skirt Length

- [x] Inspect `prompts-anima-skirt_length_slider.yaml`.
- [x] Add tests that keep outfit category stable while changing hemline.
- [x] Refactor prompt text.
- [x] Verify tests.
- [x] Update progress.

### 10. Clothing Fit

- [x] Inspect `prompts-anima-clothing_fit_slider.yaml`.
- [x] Add tests that keep body shape stable.
- [x] Refactor prompt text.
- [x] Verify tests.
- [x] Update progress.

### 11. Collar Height

- [x] Inspect `prompts-anima-collar_height_slider.yaml`.
- [x] Add tests that keep garment type stable.
- [x] Refactor prompt text.
- [x] Verify tests.
- [x] Update progress.

### 12. Outfit Decoration

- [x] Inspect `prompts-anima-outfit_decoration_slider.yaml`.
- [x] Add tests that isolate ornament density from palette/background.
- [x] Refactor prompt text.
- [x] Verify tests.
- [x] Update progress.

## Phase 3: Pose, Hair, And Style Prompt Sets

### 13. Anger Intensity

- [ ] Inspect `prompts-anima-anger_intensity_slider.yaml`.
- [ ] Add tests that keep pose/background stable.
- [ ] Refactor prompt text.
- [ ] Verify tests.
- [ ] Update progress.

### 14. Hair Messiness

- [ ] Inspect `prompts-anima-hair_messiness_slider.yaml`.
- [ ] Add tests that keep hair color and length stable within each mapping.
- [ ] Refactor prompt text.
- [ ] Verify tests.
- [ ] Update progress.

### 15. Pose Dynamism

- [ ] Inspect `prompts-anima-pose_dynamism_slider.yaml`.
- [ ] Add tests that keep identity/outfit/background stable.
- [ ] Refactor prompt text.
- [ ] Verify tests.
- [ ] Update progress.

### 16. Posture Uprightness

- [ ] Inspect `prompts-anima-posture_uprightness_slider.yaml`.
- [ ] Add tests that keep age/body axis stable.
- [ ] Refactor prompt text.
- [ ] Verify tests.
- [ ] Update progress.

### 17. Pastel Tone

- [ ] Inspect `prompts-anima-pastel_tone_slider.yaml`.
- [ ] Add tests that allow palette changes but prevent subject/camera changes.
- [ ] Refactor prompt text.
- [ ] Verify tests.
- [ ] Update progress.

### 18. Line Weight

- [ ] Inspect `prompts-anima-line_weight_slider.yaml`.
- [ ] Add tests that isolate line thickness from detail density/palette.
- [ ] Refactor prompt text.
- [ ] Verify tests.
- [ ] Update progress.

### 19. Chibi Style

- [ ] Inspect `prompts-anima-chibi_style_slider.yaml`.
- [ ] Decide whether this is a style/proportion slider or an age slider.
- [ ] Add tests that prevent unintended age/identity drift.
- [ ] Refactor prompt text.
- [ ] Verify tests.
- [ ] Update progress.

## Phase 4: Effects And Environment Prompt Sets

Goal: audit and clean only the effect sets that show obvious coupling.

- [ ] Audit `prompts-anima-aura_intensity_slider.yaml` for dark-background coupling.
- [ ] Audit `prompts-anima-particle_amount_slider.yaml` for dark-background coupling.
- [ ] Audit glow/electric/spark sets for lighting bleed.
- [ ] Audit weather/effect sets for pose or mood coupling.
- [ ] Refactor one effect family per commit only when a concrete coupling is found.

## Per-Task Done Definition

Each task is done only when:

- Prompt YAML parses.
- Relevant invariant tests exist.
- `python -m pytest tests/test_prompt_util.py` passes.
- `python -m pytest` passes.
- Progress tracker row is updated.
- Remaining risks are documented.
