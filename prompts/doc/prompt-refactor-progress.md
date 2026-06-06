# Prompt Refactor Progress

Status values:

- `todo`: not started.
- `audit`: inspected, needs decisions.
- `in_progress`: tests or prompt edits started.
- `done`: refactored and verified.
- `defer`: intentionally postponed.

## Current Baseline

Generated from repository inspection on 2026-06-06.

Known global risks:

- Many character/face/body prompt sets vary hair color (`pink hair`, `red hair`, `blonde hair`) across mappings.
- Some sets include lighting/background terms that can become part of the slider axis.
- Face-detail concepts need upper-body/readable-face coverage in training prompts.
- The standard split is training `0-5`, eval `6-7`.
- Age sliders intentionally use direct age terms under `allow_unsafe_age_terms=True`; do not weaken the age difference just to satisfy default validation.

## Completed

| File | Status | Notes | Verification |
| --- | --- | --- | --- |
| `prompts-anima-age_slider_fullbody_v4.yaml` | done | Refactored around age-only direction: fixed dark hair/neutral clothes/background, added upper-body training prompts at `4,5`, removed only `chibi` because it caused style drift, and kept strong old/young age terms under `allow_unsafe_age_terms=True`. Added feminine elderly and body-shape cues. | `python -m pytest` passed after latest edits |

## High Priority Queue

| File | Status | Main Risk | Planned Fix |
| --- | --- | --- | --- |
| `prompts-anima-mature_face_slider.yaml` | done | Face axis previously mixed with hair color and camera variation. | Fixed dark hair/neutral clothes/background across roles, added readable-face upper-body training prompts at `4,5`, and locked mature-face-only contrast with tests. |
| `prompts-anima-face_roundness_slider.yaml` | done | Face shape previously varied hair colors and camera phrasing. | Fixed dark hair/neutral clothes/background across roles, added readable-face upper-body training prompts at `4,5`, and locked round-vs-angular contrast with tests. |
| `prompts-anima-eye_openness_slider.yaml` | done | Eye openness previously varied hair color, clothing, and background terms across mappings. | Fixed dark hair/neutral clothes/background across roles, kept `neutral expression`, added readable-face upper-body training prompts at `4,5`, and locked open-vs-relaxed-eyelid contrast with tests. |
| `prompts-anima-smile_intensity_slider.yaml` | done | Smile previously mixed hair color/background variation and `cheerful expression` mood terms. | Fixed dark hair/neutral clothes/background across roles, kept calm neutral eyes, removed mood/blush/color terms, and locked mouth-only smile contrast with tests. |
| `prompts-anima-blush_intensity_slider.yaml` | done | Blush previously varied hair/background and included pink/red hair mappings that could teach global palette drift. | Fixed dark hair/neutral clothes/background across roles, added `overall palette unchanged`, kept neutral expression, and locked localized cheek-only blush contrast with tests. |
| `prompts-anima-emotion_valence_slider.yaml` | done | Emotion valence previously varied hair/background and included `warm face`, which could leak into palette or lighting. | Defined allowed axis as happy-vs-melancholic facial expression, eyes, brows, and mouth; fixed pose/background/palette; banned crying/anger/dim-lighting drift with tests. |

## Body And Clothing Queue

| File | Status | Main Risk | Planned Fix |
| --- | --- | --- | --- |
| `prompts-anima-breast_size_slider.yaml` | done | Previously lacked adult/fully clothed/nonsexual safeguards and used `big/huge breasts` plus hair/background drift. | Fixed adult woman, fully clothed, nonsexual, dark hair, neutral background, and pose; changed axis to larger-vs-smaller covered bust silhouette under opaque modest clothing. |
| `prompts-anima-waist_width_slider.yaml` | done | Waist axis varied hair/background/outfit and could mix body identity with clothing style. | Fixed adult identity, base outfit, pose, and background; contrasted narrow/cinched waist against wide/straight torso silhouette only. |
| `prompts-anima-skirt_length_slider.yaml` | done | Skirt length varied hair/background and the short side could pull toward exposure or youth-coded styling. | Fixed same skirt outfit, full-body framing, pose, and background; contrasted ankle-length against knee-length hemline. |
| `prompts-anima-clothing_fit_slider.yaml` | done | Fit axis used body-hugging language and varied hair/background, risking body-shape drift. | Fixed body shape, garment type, pose, and background; contrasted tailored fitted clothing against loose clothing fit and extra fabric folds. |
| `prompts-anima-collar_height_slider.yaml` | done | Collar height varied garment types and hair/background. | Fixed same blouse garment, collar area visibility, pose, and background; contrasted high/neck-covering collar against low/open neckline. |
| `prompts-anima-outfit_decoration_slider.yaml` | done | Decoration axis varied hair/background and could leak into palette/detail density. | Fixed base outfit, pose, background, and neutral colors; contrasted ornate outfit decoration against plain decoration/minimal trim. |

## Pose, Hair, And Style Queue

| File | Status | Main Risk | Planned Fix |
| --- | --- | --- | --- |
| `prompts-anima-anger_intensity_slider.yaml` | todo | Anger can mix with pose, lighting, and action effects. | Keep pose/background fixed; contrast brows/eyes/mouth tension only. |
| `prompts-anima-hair_messiness_slider.yaml` | todo | Hair messiness can change hair length/color/style. | Fix hair color/length per mapping; contrast neat vs messy only. |
| `prompts-anima-pose_dynamism_slider.yaml` | todo | Dynamic pose can alter camera/framing/background. | Keep subject and background fixed; contrast body action only. |
| `prompts-anima-posture_uprightness_slider.yaml` | todo | Posture can overlap age/body confidence. | Keep age/body terms fixed; contrast spine/shoulders stance only. |
| `prompts-anima-pastel_tone_slider.yaml` | todo | Color slider intentionally changes palette but must not change subject. | Lock subject/background layout; contrast palette only. |
| `prompts-anima-line_weight_slider.yaml` | todo | Line weight can mix with style and detail level. | Lock palette/subject/background; contrast outline thickness only. |
| `prompts-anima-chibi_style_slider.yaml` | todo | Chibi affects proportions, age impression, and style. | Decide whether chibi is allowed as style axis; isolate proportions/style from age terms. |

## Lower Priority Effects Queue

These are lower priority because many already have clearer low/high effect pairs.

| File | Status | Main Risk |
| --- | --- | --- |
| `prompts-anima-aura_intensity_slider.yaml` | audit | `dark background` appears and may couple aura with background. |
| `prompts-anima-energy_aura_intensity_slider.yaml` | todo | Aura may couple with character power pose. |
| `prompts-anima-magic_glow_intensity_slider.yaml` | todo | Glow may couple with palette/background brightness. |
| `prompts-anima-particle_amount_slider.yaml` | audit | `dark background` appears and may couple particles with background. |
| `prompts-anima-electric_arc_intensity_slider.yaml` | todo | Electric arcs may couple with pose/action. |
| `prompts-anima-spark_amount_slider.yaml` | todo | Sparks may couple with lighting. |
| `prompts-anima-beam_thickness_slider.yaml` | todo | Beam thickness seems relatively clean. |
| `prompts-anima-laser_beam_intensity_slider.yaml` | todo | Beam intensity may couple with background brightness. |
| `prompts-anima-flame_intensity_slider.yaml` | todo | Flame intensity may couple with smoke/background. |
| `prompts-anima-smoke_density_slider.yaml` | todo | Smoke density may couple with darkness. |
| `prompts-anima-fog_density_slider.yaml` | todo | Fog density may intentionally affect background clarity. |
| `prompts-anima-rain_intensity_slider.yaml` | todo | Rain may couple with dark mood. |
| `prompts-anima-water_splash_intensity_slider.yaml` | todo | Splash intensity may couple with action pose. |

## Latest Verification

- `python -m pytest tests\test_prompt_util.py`: passed after eye-openness prompt refactor.
- `python -m pytest`: passed after eye-openness prompt refactor.
- `python -m pytest tests\test_prompt_util.py`: passed after smile-intensity prompt refactor.
- `python -m pytest`: passed after smile-intensity prompt refactor.
- `python -m pytest tests\test_prompt_util.py`: passed after blush-intensity prompt refactor.
- `python -m pytest`: passed after blush-intensity prompt refactor.
- `python -m pytest tests\test_prompt_util.py`: passed after emotion-valence prompt refactor.
- `python -m pytest`: passed after emotion-valence prompt refactor.
- `python -m pytest tests\test_prompt_util.py -k "breast_size or waist_width or skirt_length or clothing_fit or collar_height or outfit_decoration"`: passed after Phase 2 prompt refactor.
- `python -m pytest tests\test_prompt_util.py`: passed after Phase 2 prompt refactor.
- `python -m pytest`: passed after Phase 2 prompt refactor.

## Notes For Future Refactors

- Do not optimize every YAML in one commit.
- Prefer one category per commit.
- When a prompt set intentionally uses a risky term, document the reason in the progress row.
- If a trained LoRA shows background/color drift, update this progress file with the observed symptom before editing the next prompt.
