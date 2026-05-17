# Concept Slider Research Notes

Date: 2026-05-18

Scope: ComfyUI Anima Slider Node の概念方向スライダー強化に向けた調査メモ。今回はスクリプト変更なし。

## Current Implementation Notes

現在の実装は Concept Sliders 系の骨格に近い。

- `prompt_util.PromptSettings` は `target`, `positive`, `unconditional`, `neutral`, `guidance_scale`, `action` を持つ。
- `slider_loss.flow_slider_teacher()` は `positive - unconditional` を方向ベクトルとして使い、`target +/- eta * direction` を教師にする。
- `training.flow_loss_for_record()` は frozen base model から `target_base`, `positive_base`, `unconditional_base` を得て、LoRA 有効時の `target` 予測を教師に近づける。
- `direction_loss=bidirectional` は LoRA multiplier `+1` と `-1` の両方向を同時に学習する。
- `guidance_scale` は YAML から読み込まれて `AnimaPromptConds` に保持されるが、現時点では教師生成や loss weight に使われていない。

Relevant files:

- `anima_slider_node/prompt_util.py`
- `anima_slider_node/slider_loss.py`
- `anima_slider_node/training.py`
- `anima_slider_node/lora_network.py`
- `nodes.py`

## External References

- Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models
  - Paper: https://arxiv.org/abs/2311.12092
  - Project: https://sliders.baulab.info/
  - Code: https://github.com/rohitgandikota/sliders
- SliderSpace: Decomposing the Visual Capabilities of Diffusion Models
  - Paper: https://arxiv.org/abs/2502.01639
  - Project: https://sliderspace.baulab.info/
  - Code: https://github.com/rohitgandikota/sliderspace
- Text Slider: Efficient and Plug-and-Play Continuous Concept Control for Image/Video Synthesis via LoRA Adapters
  - Paper: https://arxiv.org/abs/2509.18831
  - Project: https://textslider.github.io/
  - Code: https://github.com/aiiu-lab/TextSlider
- Prompt Sliders for Fine-Grained Control, Editing and Erasing of Concepts in Diffusion Models
  - Paper: https://arxiv.org/abs/2409.16535
- Erasing Concepts from Diffusion Models / LECO
  - Paper: https://arxiv.org/abs/2303.07345
  - LECO code: https://github.com/p1atdev/LECO
- FLUX / Rectified Flow LoRA training references
  - diffusers FLUX LoRA: https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md
  - Scaling Rectified Flow Transformers: https://arxiv.org/abs/2403.03206
- FreeSliders
  - Paper: https://arxiv.org/abs/2511.00103

## Teacher Signal Enhancement

ここでの「教師信号の強化」は、単純な `target + eta * (positive - unconditional)` だけでなく、方向の強さ、保持したい属性、反対方向、neutral/preservation 条件を loss に明示的に入れることを指す。

Possible extensions:

1. Use `guidance_scale`
   - Current YAML already carries `guidance_scale`.
   - Teacher direction could use `eta * record.guidance_scale` or a bounded transform of it.
   - This makes prompt set metadata actually affect training.

2. Preservation / neutral loss
   - Train target concept movement while also keeping neutral or preservation prompts close to the frozen base model.
   - Example: for neutral prompts, LoRA output should remain close to base output.
   - This is the likely path to reduce style bleed and unrelated attribute drift.

3. Disentanglement attributes
   - Concept Sliders official implementation uses attribute lists to reduce unwanted coupling, such as age changing gender.
   - Node-side schema could add `preservation_prompts` or `disentangle_attributes`.
   - YAML-side schema could add prompt groups such as `preserve: [...]`.

4. Symmetric / bidirectional objective
   - Current `bidirectional` mode already moves in this direction.
   - Further strengthening would make the `+scale` and `-scale` branches report separate metrics and possibly use different weights.

5. Scale sweep evaluation
   - Evaluate LoRA multipliers such as `-1.0`, `-0.5`, `0`, `0.5`, `1.0`.
   - This does not directly improve training, but it reveals whether the learned direction is smooth, saturated, or unstable.

### Benefits

- Stronger targeted control: the trained LoRA is more likely to move the intended concept rather than merely learning a generic prompt bias.
- Lower interference: preservation/disentanglement losses can reduce unwanted changes to identity, gender, style, composition, or background.
- More predictable slider strength: using `guidance_scale` and scale-sweep evaluation makes the output easier to tune.
- Better use of existing YAML: current prompt files already contain `guidance_scale`; using it reduces dead metadata.
- More paper-aligned behavior: Concept Sliders is not just ordinary LoRA training; it is a teacher-guided low-rank direction. Stronger teacher design moves this implementation closer to that method.
- Better diagnostics: separate metrics for enhance, erase, neutral, and preservation failures make bad prompt sets easier to debug.

### Demerits

- More forward passes: preservation and disentanglement losses require additional frozen-model and LoRA-model evaluations. This increases runtime and VRAM pressure.
- More hyperparameters: weights for direction, neutral preservation, disentanglement, bidirectional balance, and scale sweep can make the node harder to use.
- Risk of underpowered sliders: if preservation is too strong, the LoRA may learn to do almost nothing.
- Prompt sensitivity: preservation attributes are only as good as the prompts. Bad preservation prompts can preserve the wrong thing or suppress the desired edit.
- Harder debugging in ComfyUI: current training already handles inference-mode tensors, frozen model paths, LoRA injection, and DynamicVRAM behavior. Extra branches increase the surface area for detached-loss or device/dtype issues.
- Report size and UI complexity: richer evaluation produces more data, but not all of it should become node inputs. Some parameters should likely stay advanced or YAML-only.

Pragmatic recommendation: implement teacher enhancement in small steps. First make `guidance_scale` effective and reported. Then add optional neutral preservation. Only after that add full disentanglement attributes.

## Automatic Direction Discovery

SliderSpace is interesting because it discovers multiple semantic directions from a broad concept instead of requiring the user to manually write every `positive/unconditional` pair.

The shape is different from the current trainer, so it should probably be a separate node rather than folded into `Train Anima Slider LoRA`.

Recommended node split:

1. `Discover Anima Slider Directions`
   - Inputs: base concept prompt, sample count, seed range, number of candidate directions, generation resolution, optional CLIPVision/image-feature provider.
   - Work: generate or receive sample images, extract image features, run PCA or clustering, produce candidate semantic directions.
   - Outputs: candidate report JSON, candidate prompt/YAML suggestions, optional preview image grid paths.

2. `Train Anima Slider LoRA`
   - Keep this node focused on training a LoRA from an explicit prompt spec.
   - It can consume the YAML or JSON produced by the discovery node.

3. Optional `Evaluate Anima Slider Sweep`
   - Inputs: trained LoRA, prompt set, scale list, fixed seeds.
   - Outputs: preview grid and metrics for strength/smoothness/interference.

Why separate:

- Discovery is exploratory and image/feature-heavy; training is deterministic and optimizer-heavy.
- SliderSpace-style discovery likely needs image features. ComfyUI text `CLIP` is not necessarily enough; a CLIPVision-like input or an existing image-embedding path may be needed.
- Discovery may create several candidate directions, while the trainer should train one clear direction at a time.
- Keeping discovery separate avoids making the existing trainer UI too large.

Potential technical routes:

- Lowest-risk route: discovery node only proposes candidate YAML from generated text/prompt variations, without CLIP image PCA. This is simpler but less like SliderSpace.
- SliderSpace-like route: generate images from a concept prompt, encode images with CLIPVision or another image embedding model, PCA the embeddings, then ask the user or a lightweight captioning step to label directions.
- Hybrid route: use image PCA to find candidate clusters, then generate prompt suggestions for each cluster and pass those into the existing trainer.

Main dependency risk:

- True SliderSpace-style discovery needs an image embedding model. Adding `open_clip`, transformers CLIP image models, or similar would be a new dependency unless the implementation can reuse ComfyUI's existing CLIPVision model inputs. Given the repo rule of no new dependencies without explicit request, a separate node with an optional `CLIP_VISION` input is the cleaner design.

## Refactor Direction

If implementation proceeds later, keep the change sequence small:

1. Extract teacher construction behind a `TeacherStrategy`.
2. Make `guidance_scale` active in the existing teacher path.
3. Add neutral preservation as an optional loss branch.
4. Add richer evaluation/reporting before adding more node inputs.
5. Create automatic direction discovery as a separate node only after the training path has stable evaluation.

Do not start with SliderSpace integration inside the trainer. It is a different workflow and will make the training node harder to maintain.
