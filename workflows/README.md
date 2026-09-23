# Anima 2.9Bの正式採用学習ワークフロー

- 通常使用：[anima_2_9b_slider_training.json](anima_2_9b_slider_training.json)
- 提供原本：[image_anima_preview_anima_2_9b_original.json](archive/image_anima_preview_anima_2_9b_original.json)

ユーザー提供の`Downloads/image_anima_preview.json`を、Anima 2.9B用の正式採用パラメーターとして追加した。原本はバイト単位で同一。SHA-256は`bc65fa3a2ad65ea89184e848bca672cdff290723cc4ff8fd00b366cd79764566`。

通常使用版は**選択YAMLを旧v5から正規の老齢化へ更新し、MarkdownNoteを現在の説明に置き換えた**。数値、学習方式、seedと実行後制御、モデル選択、接続、出力prefixなどの実行設定は原本を維持している。リポジトリ直下の旧`image_anima_preview.json`は別の過去資料で、この正式採用版とは異なる。

## 読み込みと使い分け

ComfyUIに通常使用版JSONをドロップするか、ワークフローとして開く。これはUI用のグラフJSONであり、REST `/prompt`にそのまま送るAPI JSONではない。

`UNETLoader`・`CLIPLoader`・`VAELoader`から、このパックの出力ノード`ComfyuiAnimaSliderTrainLora`へ接続されている。学習ノード自体がLoRAとレポートを保存する。元ファイル名にpreviewとあるが、KSamplerや画像保存ノードによる**生成画像プレビューは含まない**。

| 用途 | 選択する`prompt_yaml` |
| --- | --- |
| 老齢化（通常使用版の選択値） | `prompts-anima-aging_slider_fullbody.yaml` |
| 幼齢化 | `prompts-anima-deaging_slider_fullbody.yaml` |
| 胸部サイズ：中程度→増大（正式版） | `prompts-anima-breast_size_slider_v4.yaml` |
| 既存衣服フィットの調査 | `prompts-anima-clothing_fit_slider.yaml` |
| 衣服：標準→密着 | `prompts-anima-clothing_fit_slider_v2.yaml` |
| 衣服：標準→ゆったり | `prompts-anima-clothing_fit_slider_v3.yaml` |

胸部はLora10の評価を受けてv4のみ正式採用し、初版・v2・縮小用v3は[archive](../prompts/archive/README.md)へ移動した。衣服v2/v3は方向別の学習例で、ゆったりv3はLora9で評価済み、密着v2は未評価。胸部v4・衣服v2/v3は`enhance_only`で別LoRAとして学習し、正の強度で適用する。設定と出力prefix例は[プロンプトREADME](../prompts/README.md)を参照。これらもguidance2.0なので実効係数3.0となる。

`custom_prompt_yaml_path`は空欄。入力すると`prompt_yaml`より優先される。ComfyUIが改名前の一覧を保持している場合は再起動してから開く。原本を再現する場合、旧v5はarchiveにあるため、この入力へ移動先YAMLの絶対パスを指定する。

## 保存されているパラメーター

| 項目 | 保存値 |
| --- | --- |
| diffusion model | `miaomiaoHarem_29BBETA10.safetensors`、weight_dtype=`default` |
| CLIP | `qwen_3_06b_base.safetensors`、type=`stable_diffusion`、device=`default` |
| VAE | `qwen_image_vae.safetensors` |
| train / eval indices | `0,1,2,3,4,5` / `6,7` |
| steps / 共通lr | `900` / `1.5e-5` |
| rank / alpha | `16` / `16` |
| network_preset / network_reg_dims | `attn_mlp` / 空欄 |
| module別lr | self-attention `8e-6`、cross-attention `2.5e-5`、MLP `1.5e-6` |
| model_residency / lora_weight_dtype | `prefer_cuda` / `fp32` |
| gradient_checkpointing | `true` |
| skip_initial_eval / skip_final_eval | `false` / `false` |
| width / height | **`1024` / `1024`を明示** |
| num_inference_steps | `20`（学習内の軌道計算） |
| timestep_sampling / sigmoid_scale | `shift` / `1` |
| discrete_flow_shift / loss_weighting_scheme | `3` / `none` |
| direction_loss | **`enhance_only`** |
| teacher_guidance_scale / teacher_norm_reference | `1` / `target` |
| min_step_index / max_step_index | `-1` / `-1` |
| eval_step_indices | 空欄（現実装では中間時刻） |
| eta | `1.5` |
| seed / control_after_generate | `583650396415791` / **`randomize`** |
| eval_seed / vary_seed | `961218314523996` / `true` |
| allow_unsafe_age_terms | `true`（幼齢化でも使用できる原本設定） |
| output_lora_prefix | `loras/anima_age_slider` |

現実装のこのノードはAdamWを使用し、学習内schedulerは`simple`。これらはJSONで選択するウィジェットではなく実装側の設定である。学習ノードの入力順と、seed直後の実行後制御ウィジェットを区別して上表へ対応付けた。

モデル名は提供グラフの選択値で、ファイルそのものは同梱しない。モデルを変更する場合は対応するAnima 2.9B / 40 blocksを選び、変更後の名前とハッシュを実験記録へ残す。このグラフのモデル名を、過去の全LoRAの学習元を証明する記録にはしない。

## 他スライダーの比較で変更・確認する項目

- 年齢正規版のYAML guidanceは1.0なので実効係数は`1.5 × 1.0 × 1.0 = 1.5`。既存の胸部サイズ・衣服フィットは2.0なので**3.0**になる。比較途中でguidanceやetaを変える場合は方式変更とは別実験として記録する。
- width/heightの明示値1024×1024がYAMLの896×1152より優先される。解像度変更も同時に混ぜない。
- A/B比較では`randomize`を`fixed`に変更し、実行seedを照合する。`vary_seed=true`は学習中にstepごとにseedを変える別設定。さらにLoRA初期化乱数は学習seedと別なので、[調査手順](../prompts/doc/slider-direction-investigation.md)の初期化の留保を適用する。
- 正式採用版は単方向。方式比較のA条件のみ`bidirectional`に変更し、同じYAMLのB条件は`enhance_only`とする。
- 胸部サイズ・衣服フィットの実験では`output_lora_prefix`を概念・方向・run IDが分かる名前へ変更する。元のage名のまま成果物を混ぜない。

本文・設定の読み取りと接続の静的検証は、実学習成功の検証とは別である。この追加作業では学習ジョブを実行していない。確認時にlocalhost / 127.0.0.1の8000・8188番へ接続できなかったため、起動中サーバーのモデル候補・ノード登録・UI読込の確認は未実施。グラフのモデル名は提供値として保存している。
