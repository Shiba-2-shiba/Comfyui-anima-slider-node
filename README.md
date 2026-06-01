# ComfyUI Anima Slider Node

Anima/Cosmos RFlow 向けの experimental text-only slider LoRA training を ComfyUI の custom node として実行するためのリポジトリです。

## 使い方

このフォルダを ComfyUI の `custom_nodes` 配下に置くか、シンボリックリンクしてください。

```powershell
cd C:\path\to\ComfyUI\custom_nodes
git clone https://github.com/Shiba-2-shiba/Comfyui-anima-slider-node Comfyui-anima-slider-node
```

ComfyUI を再起動すると、`training/anima slider` に `Train Anima Slider LoRA` が追加されます。

## ノードの入出力

入力:

- `MODEL`: ComfyUI でロードした diffusion model
- `CLIP`: ComfyUI でロードした text encoder
- `VAE`: ワークフロー互換用の入力。現在の text-only loss では画像 encode には使いません
- `prompt_yaml`: 同梱 `prompts/*.yaml` から選択
- `custom_prompt_yaml_path`: 任意の YAML を直接指定する場合に使用
- `steps`, `lr`, `rank`, `alpha`, `network_preset`, `model_residency`, `lora_weight_dtype`, `width`, `height` などの学習設定

出力:

- `lora`: ComfyUI の `LORA_MODEL` 型
- `report_json`: 学習 report JSON 文字列
- `lora_path`: 保存済み `.safetensors`
- `report_path`: 保存済み `.json`

LoRA と report は ComfyUI の output directory 配下に保存されます。`output_lora_prefix` の既定値は `loras/anima_slider` です。

## 16GB VRAM向けの確認手順

16GB VRAMで高解像度学習を狙う場合は、`network_preset=attn_mlp` を維持し、`model_residency=prefer_cuda` と `gradient_checkpointing=True` を基本設定にしてください。`dynamic` はCUDA常駐が失敗する場合の最後の手段です。

`lora_weight_dtype` は既定の `fp32` を推奨します。`auto` も fp32 の trainable LoRA weight を使います。VRAM をさらに削りたい場合だけ `base` または `bf16` を試してください。ただし bf16 LoRA weight は小さい学習率の更新が丸めで消えやすく、品質確認が必須です。

推奨する切り分け順:

1. `width=512`, `height=512`, `steps=1`, `prompt_indices=0`
2. `width=768`, `height=768`, `steps=1`, `prompt_indices=0`
3. `width=1024`, `height=1024`, `steps=1`, `prompt_indices=0`, `skip_initial_eval=True`, `skip_final_eval=True`
4. 1024x1024の学習本体が通った後で、`skip_initial_eval=False`, `skip_final_eval=False` に戻してeval込みを確認

`skip_initial_eval` と `skip_final_eval` はOOM phaseを分けるための診断用です。品質評価の代替ではありません。report JSONには `gradient_checkpointing`, eval skip設定、setup後とtext adapter precompute後のCUDA memory diagnostics、各stepのphase timingsが記録されます。

## Prompt YAML

YAML は list 形式です。

```yaml
- target: "base prompt"
  positive: "base prompt, direction to enhance"
  unconditional: "base prompt, opposite direction"
  neutral: "base prompt"
  guidance_scale: 2.0
  action: enhance
  width: 1024
  height: 1024
  batch_size: 1
```

`positive` / `unconditional` / `neutral` は省略可能です。`positive` と `neutral` は `target` に、`unconditional` は空文字にフォールバックします。

## 注意

- この実装は `anima-slider-experiment` の flow slider trainer を ComfyUI 入力モデル向けに移植したものです。
- `MODEL` に LoRA wrapper を一時注入しますが、学習終了時に元の linear module へ戻します。
- `steps` の既定値は `600` です。短い smoke 確認だけ行う場合は、一時的に `steps=3`, `width=512`, `height=512`, `prompt_indices=0,1,2,3` 程度まで下げてください。
- `model_residency=prefer_cuda` は ComfyUI のロード後に base model を CUDA へ寄せる best-effort 設定です。OOM になる環境では `dynamic` に戻してください。
- `lora_weight_dtype=base` は旧挙動に近く、base model が bf16 なら LoRA weight も bf16 になります。sd-scripts の通常の Anima LoRA 学習に寄せるなら `fp32` を使ってください。
- 16GB VRAMで1024x1024を狙う場合も、LoRA対象を `attn_only` へ削るのではなく、まず `attn_mlp` と `gradient_checkpointing=True` の組み合わせで確認してください。
- bundled prompt のうち年齢語を含む YAML は `allow_unsafe_age_terms=True` が必要な場合があります。
