# Prompt Refactor Specification

## Purpose

This document defines the refactor rules for Anima slider prompt YAML files.
The goal is to make each prompt set learn one clean slider axis while reducing
style bleed, background drift, color drift, camera drift, and identity drift.

This refactor is behavior-preserving at the schema level: every YAML file must
remain a list of prompt mappings compatible with the current trainer.

## Scope

In scope:

- Prompt YAML files under `prompts/`.
- Prompt documentation and validation tests that lock prompt invariants.
- Refactoring prompt text to isolate the intended slider axis.

Out of scope unless explicitly requested:

- Trainer algorithm changes.
- Node UI changes.
- New dependencies.
- Changing the 8-prompt train/eval convention.

## Cleanup Plan

1. Add or update tests before editing a prompt family.
2. Refactor one prompt category at a time.
3. Keep prompt schema and file names stable unless a migration is intentional.
4. Preserve the train/eval split: indices `0-5` are training prompts, `6-7` are eval prompts.
5. Run prompt tests after each prompt family.
6. Run the full test suite before commit.

## Prompt Invariants

Every standard 8-prompt YAML should satisfy:

- Exactly 8 mappings.
- Required keys: `target`, `positive`, `unconditional`, `neutral`, `guidance_scale`, `action`, `width`, `height`, `batch_size`.
- `neutral == target` unless a specific exception is documented.
- `action: enhance`.
- `batch_size: 1`.
- `guidance_scale` is consistent within the prompt set unless a reason is documented.
- One resolution per file.
- Indices `6,7` remain eval prompts and should not be the only prompts that express the core concept.

## Axis Isolation Rules

Within a single prompt mapping:

- Keep quality tags identical across all roles.
- Keep subject count identical across all roles.
- Keep gender/age/body identity fixed unless it is the slider axis.
- Keep camera framing identical across all roles.
- Keep background and lighting identical across all roles unless background or lighting is the slider axis.
- Keep hair color, clothing color, and outfit category identical unless they are the slider axis.
- Put only the intended concept difference in `positive` and `unconditional`.

Bad pattern:

```yaml
positive: "old woman, gray hair, dark room, long dress"
unconditional: "young girl, pink hair, bright room, short dress"
```

Better pattern:

```yaml
positive: "same room, same dark hair, same outfit, elderly woman, wrinkles"
unconditional: "same room, same dark hair, same outfit, young girl, smooth skin"
```

## Category Rules

### Face And Expression Sliders

Examples: mature face, face roundness, eye openness, smile, blush, emotion valence, anger.

- Include 1-2 upper-body/readable-face training prompts at indices `4,5` when facial detail matters.
- Keep hair color and clothing color fixed within each mapping.
- Avoid changing background tone between positive and unconditional.
- Avoid adding camera changes only to one role.

### Age Sliders

Age sliders are allowed to use direct age terms when the workflow explicitly
uses `allow_unsafe_age_terms=True`. Do not weaken the age axis merely to satisfy
the default prompt validator.

- Keep strong old-age terms when they are needed for visible slider strength.
- Keep strong young-direction terms when they are needed for a usable negative direction.
- Do not replace the young side with weak adult-only wording if that makes the learned direction ineffective.
- Do not add safety euphemisms that collapse the positive/unconditional difference.
- Remove terms only when they create a demonstrated unwanted axis, such as `chibi` causing style/proportion drift.
- Record any intentional age-risk terms in tests or progress notes so future edits do not remove them accidentally.

### Body And Clothing Sliders

Examples: breast size, waist width, clothing fit, collar height, skirt length.

- Include `adult`, `fully clothed`, and `nonsexual` where the concept is body-related.
- Keep camera framing identical across roles.
- Keep clothing type identical across roles unless the clothing item itself is the slider axis.
- Prefer silhouette terms over sexualized descriptors.

### Pose And Posture Sliders

Examples: pose dynamism, posture uprightness.

- Keep subject, outfit, background, and style fixed.
- Use body-position terms as the only major axis.
- Avoid changing facial expression unless expression is part of the concept.

### Style And Color Sliders

Examples: pastel tone, line weight.

- Color/style terms are allowed as the axis.
- Keep subject, clothing category, pose, and background layout stable.
- Use eval prompts that are not identical to training backgrounds.

### Effects And Environment Sliders

Examples: flame, smoke, fog, rain, debris, laser, splash.

- The effect may change the environment, but the base scene should remain comparable.
- Avoid changing character identity or pose as part of the effect.
- Use clear low/high effect pairs.

## Validation Requirements

Prompt refactors should be locked with tests for:

- Required schema and resolution.
- Category-specific invariants.
- No unintended color/camera/background axis for face/body prompt sets.
- Intentional unsafe age terms only when the training workflow requires `allow_unsafe_age_terms=True`.
- No deprecated terms for the current design, such as `chibi` in the age slider young direction.

## Recommended Refactor Workflow

For each prompt family:

1. Read the current YAML.
2. Identify the intended slider axis.
3. List nuisance axes currently mixed into positive/unconditional.
4. Add or update tests for the intended invariant.
5. Edit the prompt set.
6. Run `python -m pytest tests/test_prompt_util.py`.
7. Run `python -m pytest`.
8. Record the result in `prompt-refactor-progress.md`.

## Acceptance Criteria

A prompt family is considered refactored when:

- The intended slider axis can be described in one sentence.
- Each prompt mapping changes only that axis.
- Tests lock the family-specific invariants.
- Full test suite passes.
- Any remaining risk is recorded in the progress file.
