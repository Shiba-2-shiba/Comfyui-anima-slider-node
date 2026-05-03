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
- `steps`, `lr`, `rank`, `alpha`, `network_preset`, `width`, `height` などの学習設定

出力:

- `lora`: ComfyUI の `LORA_MODEL` 型
- `report_json`: 学習 report JSON 文字列
- `lora_path`: 保存済み `.safetensors`
- `report_path`: 保存済み `.json`

LoRA と report は ComfyUI の output directory 配下に保存されます。`output_lora_prefix` の既定値は `loras/anima_slider` です。

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
- bundled prompt のうち年齢語を含む YAML は `allow_unsafe_age_terms=True` が必要な場合があります。
