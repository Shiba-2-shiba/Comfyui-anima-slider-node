# QPOLA Anima Slider LoRA ノード実装仕様書

- 文書状態: 実装前仕様（初稿確定）
- 仕様バージョン: 0.1
- 対象リポジトリ: `Comfyui-anima-slider-node`
- 対象ノード: `Train Anima Slider LoRA (QPOLA)`
- QPOLA参照実装: v1.0.4 (`QPOLA/optimizer`)

## 1. 目的

現行の text-only Anima/Cosmos flow-slider LoRA 学習に、QPOLA のモーメントフリーな空間協調型パラメータ更新を導入する。

Slider の教師信号、損失、timestep sampling、LoRA 注入対象は変更しない。変更対象は `loss.backward()` 後の optimizer 更新だけとし、既存 AdamW ノードとの同条件比較を可能にする。

## 2. 背景と現行動作

現行ノード `ComfyuiAnimaSliderTrainLora` は、次の学習フローを持つ。

```text
Prompt YAML
  -> CLIP conditioning
  -> target / positive / unconditional のモデル出力
  -> target +/- eta * (positive - unconditional) の teacher
  -> teacher と LoRA 有効時出力の MSE
  -> backward
  -> AdamW.step
  -> LoRA safetensors + report JSON
```

QPOLA はデータセット、教師信号、損失関数を提供する学習パイプラインではなく、`torch.optim.Optimizer` 互換の更新器である。このため、QPOLA の導入によって `slider_loss.flow_slider_teacher()` の意味や既存 Prompt YAML の契約を変えてはならない。

なお、現行の `scheduler_name="simple"` は diffusion sigma 列の生成設定であり、optimizer の learning-rate scheduler ではない。QPOLA 導入後も維持する。

## 3. スコープ

### 3.1 対象

- QPOLA 専用の新しい ComfyUI V3 backend node を追加する。
- 既存 AdamW ノードと学習ロジックを共有する。
- QPOLA v1.0.4 の Python loader、PTX、対応する CUDA source、ライセンスをパッケージ内へ同梱する。
- QPOLA の optimizer 設定と実装バージョンを report JSON および safetensors metadata に記録する。
- CPUだけの環境でも、既存 AdamW ノードの登録・検索・実行を阻害しない。
- 単体テスト、パッケージテスト、NVIDIA CUDA 実機 smoke test を用意する。

### 3.2 対象外

- Slider teacher または MSE loss の再設計
- 画像データセットを使用する LoRA 学習への変更
- int8、fp8、fp4 LoRA weight の導入
- AMD ROCm、Apple MPS、Intel GPU、CPU-only 向け QPOLA 移植
- QPOLA の数式または CUDA kernel の独自改変
- frontend JavaScript、独自widget、進捗DOMの追加
- QPOLA失敗時に AdamW へ自動的に切り替える処理

## 4. 設計上の決定事項

### 4.1 別ノードとして追加する

既存ノードへ optimizer 選択欄を追加せず、次の独立ノードを登録する。

| 項目 | 値 |
|---|---|
| Python class | `AnimaSliderTrainLoraQpolaNode` |
| `node_id` | `ComfyuiAnimaSliderTrainLoraQpola` |
| `display_name` | `Train Anima Slider LoRA (QPOLA)` |
| `category` | `training/anima slider` |
| `description` | `Train an experimental Anima/Cosmos flow-slider LoRA with the QPOLA optimizer. NVIDIA CUDA is required.` |
| `search_aliases` | `anima slider`, `flow slider`, `train lora`, `qpola`, `moment free optimizer` |
| `is_experimental` | `True` |
| `is_output_node` | `True` |
| `not_idempotent` | `True` |

理由:

- 既存workflowのschemaと既定動作を変更しない。
- AdamW版とQPOLA版を同じworkflow内で明確に区別できる。
- CUDA/PTX制約をQPOLAノードだけに閉じ込められる。
- 出力がどのoptimizerで学習されたかUI上でも判別できる。

### 4.2 QPOLA の自動フォールバックは禁止する

QPOLA のロード、CUDA preflight、kernel launch のいずれかが失敗した場合は学習を停止し、原因を含む例外を返す。AdamW への暗黙フォールバックは行わない。

QPOLAノードからAdamWモデルが出力されると、結果の再現性とreportの信頼性が失われるためである。

### 4.3 Phase 1 は fp32 LoRA に限定する

初期実装では `lora_weight_dtype` を `fp32` のみに固定する。QPOLA 自体が対応する bf16/fp16/int8/fp8 と、ComfyUIで保存・再適用するLoRA weight dtypeの対応は別の検証項目とする。

Base model の dtype はこの制限の対象外である。QPOLAが更新する trainable LoRA parameter が fp32 かつ CUDA 上に存在することをpreflightで検証する。

## 5. ノード入出力仕様

### 5.1 共通入力

次の入力は既存 `Train Anima Slider LoRA` と同じ名前、型、意味、並び順を維持する。

- `model`, `clip`, `vae`
- `prompt_yaml`, `custom_prompt_yaml_path`
- `prompt_indices`, `eval_prompt_indices`
- `steps`
- `rank`, `alpha`
- `network_preset`, `network_reg_dims`, `network_reg_lrs`
- `model_residency`
- `gradient_checkpointing`
- `skip_initial_eval`, `skip_final_eval`
- `width`, `height`
- `num_inference_steps`
- `timestep_sampling`, `sigmoid_scale`, `discrete_flow_shift`
- `loss_weighting_scheme`, `direction_loss`
- `teacher_guidance_scale`, `teacher_norm_reference`
- `min_step_index`, `max_step_index`, `eval_step_indices`
- `eta`
- `seed`, `eval_seed`, `vary_seed`
- `allow_unsafe_age_terms`

共通入力のvalidation、tooltip、YAML解決規則は既存ノードと同じとする。

### 5.2 QPOLAノード固有または変更される入力

| 入力 | ComfyUI型 | 既定値 | 制約 | 説明 |
|---|---|---:|---|---|
| `lr` | `FLOAT` | `0.0001` | `0.0..1.0`, step `0.0000001` | QPOLAの最大更新幅。AdamWの既定値を流用しない。 |
| `lora_weight_dtype` | `COMBO` | `fp32` | options: `fp32` のみ | Phase 1ではQPOLA更新対象をfp32へ限定する。 |
| `qpola_eps` | `FLOAT` | `1e-8` | `1e-12..1e-2` | 勾配の局所スケール正規化に使うepsilon。 |
| `qpola_low_vram` | `BOOLEAN` | `True` | - | `True` の場合、QPOLA upstream挙動に従い各step後にCUDA cacheを解放する。速度低下の可能性をtooltipに明記する。 |
| `output_lora_prefix` | `STRING` | `loras/anima_slider_qpola` | 空文字不可 | ComfyUI output directory配下の保存prefix。 |

`network_reg_lrs` によるmodule別learning rateは維持する。QPOLAはPyTorch parameter groupごとの `lr` を使用するため、既存のgroup構築結果をそのまま渡す。

初期の推奨比較値は `3e-5`, `1e-4`, `3e-4` とする。`1e-3` はQPOLA upstreamの一般的なLoRA推奨値であり、このslider lossに対する安全な既定値とは見なさない。

### 5.3 出力

既存ノードと同じ4出力を同じ順序で返す。

| 出力 | 型 | 内容 |
|---|---|---|
| `lora` | `LORA_MODEL` | 学習済みLoRA state dict |
| `report_json` | `STRING` | 学習reportのJSON文字列 |
| `lora_path` | `STRING` | 保存したsafetensorsの絶対パス |
| `report_path` | `STRING` | 保存したreport JSONの絶対パス |

## 6. Backend構成

### 6.1 ファイル構成

実装時の予定構成は次のとおり。

```text
Comfyui-anima-slider-node/
  nodes.py
  pyproject.toml
  README.md
  QPOLA_NODE_SPEC.md
  anima_slider_node/
    training.py
    optimizer_factory.py
    third_party/
      __init__.py
      qpola/
        __init__.py
        optimizer.py
        qpola.cu
        qpola_kernel.ptx
        LICENSE
        NOTICE.md
  tests/
    test_optimizer_factory.py
    test_qpola_node_schema.py
    test_qpola_packaging.py
```

`optimizer_factory.py` はoptimizer選択とpreflightだけを担当し、slider loss、model forward、LoRA injectionを持たない。

### 6.2 `TrainRequest` の拡張

`training.TrainRequest` の末尾へ次を追加する。

```python
optimizer_type: str = "adamw"
optimizer_eps: float = 1e-8
optimizer_low_vram: bool = False
```

既定値を設け、既存test fixtureや内部呼び出しを壊さない。既存ノードは明示的に `optimizer_type="adamw"`、新ノードは `optimizer_type="qpola"` を渡す。

`optimizer_low_vram` はAdamWでは参照しない。QPOLAノードからは `qpola_low_vram` の値をこのfieldへ渡す。

### 6.3 Optimizer factory

公開境界は次の形を基本とする。

```python
def build_optimizer(
    param_groups: list[dict],
    *,
    optimizer_type: str,
    lr: float,
    eps: float = 1e-8,
    low_vram: bool = False,
) -> tuple[torch.optim.Optimizer, dict[str, object]]:
    ...
```

`optimizer_factory.py` から `training.TrainRequest` をimportしてはならない。factoryへ必要値を明示的に渡し、`training.py` との循環importを避ける。

挙動:

1. `optimizer_type == "adamw"`
   - 現行と同じ `torch.optim.AdamW(param_groups, lr=lr)` を生成する。
   - 現行の暗黙既定値 `weight_decay=0.01` を変更しない。
2. `optimizer_type == "qpola"`
   - CUDA preflightを実行する。
   - vendored QPOLAを関数内でlazy importする。
   - `QPOLA(param_groups, lr=lr, eps=eps, low_vram=low_vram)` を生成する。
3. その他
   - `ValueError` とし、未知のoptimizerを黙って受理しない。

戻り値のmetadataには最低限、次を含める。

```json
{
  "type": "qpola",
  "implementation_version": "1.0.4",
  "lr": 0.0001,
  "eps": 1e-8,
  "low_vram": true,
  "weight_decay_applied": false,
  "moment_state": false
}
```

### 6.4 Lazy load境界

次のmodule import時には、CUDA driverまたはPTXをロードしてはならない。

- package root `__init__.py`
- `nodes.py`
- `anima_slider_node.training`
- `anima_slider_node.optimizer_factory`

QPOLAの実装importは、QPOLAノードの `execute()` が学習要求を組み立てた後、optimizer factoryのQPOLA分岐へ到達した時だけ行う。

これにより、CPU-only環境またはPTX非対応環境でも、ComfyUI起動と既存AdamWノードの登録を成功させる。

### 6.5 CUDA preflight

QPOLA optimizer生成前に次を検証する。

- `torch.cuda.is_available()` が `True`
- trainable LoRA parameterが1個以上ある
- 全trainable LoRA parameterがCUDA device上にある
- 全trainable LoRA parameterが `torch.float32`
- 同梱 `qpola_kernel.ptx` が存在する
- CUDA driver loaderを初期化できる

失敗メッセージには、失敗条件、検出device/dtype、推奨修正を含める。例:

```text
QPOLA requires fp32 LoRA parameters on NVIDIA CUDA, but found device=cpu.
Use model_residency=prefer_cuda and lora_weight_dtype=fp32, or use the AdamW node.
```

### 6.6 学習ループへの接続

`training.train_lora_from_records()` の次の行だけをfactory呼び出しへ置き換える。

```python
optimizer = torch.optim.AdamW(optimizer_param_groups, lr=request.lr)
```

置換後:

```python
optimizer, optimizer_metadata = build_optimizer(
    optimizer_param_groups,
    optimizer_type=request.optimizer_type,
    lr=request.lr,
    eps=request.optimizer_eps,
    low_vram=request.optimizer_low_vram,
)
```

以下は変更しない。

- `optimizer.zero_grad(set_to_none=True)`
- `loss.backward()`
- `optimizer.step()`
- prompt cycling
- timestep sampling
- teacher計算
- initial/final eval
- LoRA module restoreと`mp.cleanup()`の`finally`境界

QPOLAの例外が発生しても、既存のcleanup境界によってbase linear moduleが必ず復元されなければならない。

## 7. ComfyUI登録仕様

`AnimaSliderExtension.get_node_list()` は次の両方を返す。

```python
return [
    AnimaSliderTrainLoraNode,
    AnimaSliderTrainLoraQpolaNode,
]
```

V3登録を正とする。現在提供しているlegacy compatibility mappingにも新node IDを追加する。

```python
NODE_CLASS_MAPPINGS = {
    "ComfyuiAnimaSliderTrainLora": AnimaSliderTrainLoraNode,
    "ComfyuiAnimaSliderTrainLoraQpola": AnimaSliderTrainLoraQpolaNode,
}
```

frontend JavaScriptは追加しない。schema metadataだけで `/api/object_info` と Nodes 2.0検索に表示できる実装とする。

## 8. 共通処理の再利用

`nodes.py` 内で長大なschemaとexecute処理を複製しない。次の小さいhelperへ抽出する。

- 共通input schemaを毎回新規構築するhelper
- Prompt YAML解決、validation、conditioning encode、`TrainRequest`構築、保存を行う共通execute helper

各node classは次だけを決定する。

- node metadata
- `lr` の既定値
- 許可する `lora_weight_dtype`
- optimizer固有input
- optimizer type
- output prefix既定値

helper抽出前後で既存AdamW nodeのschema、既定値、出力順、保存形式を変えてはならない。

## 9. Reportとsafetensors metadata

### 9.1 Report JSON

既存fieldに加えて次を保存する。

```json
{
  "trainer_type": "comfyui_flow_slider_qpola",
  "optimizer": {
    "type": "qpola",
    "implementation_version": "1.0.4",
    "lr": 0.0001,
    "eps": 1e-8,
    "low_vram": true,
    "weight_decay_applied": false,
    "moment_state": false
  }
}
```

既存AdamW nodeにも、後方互換な追加fieldとして次を記録してよい。

```json
{
  "optimizer": {
    "type": "adamw",
    "lr": 0.000005,
    "weight_decay": 0.01,
    "moment_state": true
  }
}
```

既存fieldの削除・改名は禁止する。

### 9.2 Safetensors metadata

metadata valueは文字列として次を保存する。

```json
{
  "format": "pt",
  "trainer_type": "comfyui_flow_slider_qpola",
  "optimizer_type": "qpola",
  "optimizer_version": "1.0.4"
}
```

`_save_lora_and_report()` のhard-coded `trainer_type` はreportまたは明示引数から取得する。既存AdamW出力では従来値 `comfyui_flow_slider` を維持する。

## 10. Vendoringとライセンス

QPOLAは外部の兄弟directory `../QPOLA` からruntime importしない。custom node単体で配布・移動できるよう、必要ファイルをpackage内に固定して同梱する。

同梱物:

- QPOLA Python optimizer loader
- `qpola_kernel.ptx`
- PTXに対応する `qpola.cu`
- Apache License 2.0全文
- upstream URL、version、vendoring日、ローカル変更の有無を記載した `NOTICE.md`

vendored sourceを変更した場合、Apache-2.0の条件に従って変更箇所を明記する。PTXだけでなく対応するsourceを含める。

新しいPyPI依存は追加しない。

`pyproject.toml` はsource tree全体をinstallable package `comfyui_anima_slider_node` へ割り当て、node entrypoint、Prompt YAML、sub-package、non-Python dataをwheelへ含める。最低限、生成wheel内に次が存在することをtestで確認する。

- `comfyui_anima_slider_node/__init__.py`
- `comfyui_anima_slider_node/nodes.py`
- `comfyui_anima_slider_node/prompts/*.yaml`
- `comfyui_anima_slider_node/anima_slider_node/third_party/qpola/optimizer.py`
- `comfyui_anima_slider_node/anima_slider_node/third_party/qpola/qpola_kernel.ptx`
- `comfyui_anima_slider_node/anima_slider_node/third_party/qpola/qpola.cu`
- `comfyui_anima_slider_node/anima_slider_node/third_party/qpola/LICENSE`
- `comfyui_anima_slider_node/anima_slider_node/third_party/qpola/NOTICE.md`

## 11. エラー処理

| 状況 | 要求動作 |
|---|---|
| CUDAが利用不能 | 実行時エラー。ComfyUI/node importは成功させる。 |
| PTXが欠落 | 実行時エラー。期待パスを表示する。 |
| CUDA driver load失敗 | 元のdriver errorを保持した説明付きエラー。 |
| LoRA parameterがCPU上 | 実行前に停止し、`model_residency=prefer_cuda`を案内する。 |
| LoRA parameterがfp32以外 | 実行前に停止し、`lora_weight_dtype=fp32`を案内する。 |
| QPOLA kernel launch失敗 | device、kernel名、CUDA result codeを含むエラー。 |
| lossがNaN/Inf | 既存diagnosticsへoptimizer metadataを加えて停止する。 |
| 未知のoptimizer type | `ValueError`。フォールバック禁止。 |

エラー後もLoRA wrapperの復元、Comfy model cleanup、参照解放が実行されること。

## 12. テスト仕様

### 12.1 既存回帰テスト

- 現在の全テストが引き続き成功すること。
- 既存AdamW nodeのnode ID、display name、input名、既定値、出力順が変わらないこと。
- optimizer factory導入後もAdamWが `weight_decay=0.01` を使用すること。

### 12.2 単体テスト

1. CPU-onlyをmockした状態でpackage、`nodes.py`、`training.py`をimportできる。
2. AdamW分岐はQPOLA moduleをimportしない。
3. QPOLA分岐は `lr`, parameter-group LR, `eps`, `low_vram` を正しく渡す。
4. 未知のoptimizer typeを拒否する。
5. QPOLA初期化失敗時にAdamWへフォールバックしない。
6. preflightがCPU parameterと非fp32 parameterを拒否する。
7. reportにoptimizer metadataが記録される。
8. safetensors metadataのtrainer/optimizer識別子が正しい。
9. `AnimaSliderExtension.get_node_list()` が2ノードを返す。
10. `/api/object_info` 相当のschemaにnode metadataと全入出力が現れる。
11. QPOLA失敗後も注入したLoRA moduleがbase linearへ復元される。

QPOLA unit testでは実CUDA kernelを起動せず、loaderとoptimizer classをmockしてoptimizer factory境界を検証する。

### 12.3 パッケージテスト

- wheelをbuildできる。
- wheelを展開し、PTX、CUDA source、LICENSE、NOTICEの同梱を確認する。
- wheelからimportした場合もPTX解決が相対パスで成功する。

### 12.4 NVIDIA CUDA実機 smoke test

最低条件:

- `lora_weight_dtype=fp32`
- `width=512`, `height=512`
- `steps=1`
- `prompt_indices=0`
- `skip_initial_eval=True`
- `skip_final_eval=True`
- `lr=1e-4`

確認項目:

- QPOLA kernelが1回以上成功する。
- lossがfiniteである。
- LoRA parameterにfiniteな非ゼロ差分が生じる。
- `.safetensors` とreport JSONが保存される。
- reportとsafetensors metadataがQPOLAを示す。
- 実行終了後にbase modelのlinear moduleが復元される。

次に `steps=3`、`prompt_indices=0,1,2` で連続更新を確認する。

## 13. A/B評価仕様

実装完了判定とは別に、品質評価として同一base model、prompt YAML、prompt indices、seed、rank、alpha、resolutionでAdamWとQPOLAを比較する。

QPOLAの候補LR:

- `3e-5`
- `1e-4`
- `3e-4`

記録項目:

- initial/final eval mean loss
- stepごとのloss
- NaN/Inf発生有無
- optimizer step時間
- 全体の秒/step
- setup後、text precompute後、step後のCUDA memory diagnostics
- LoRAファイルサイズ
- slider multiplier `-1, 0, +1` での生成結果
- concept以外の構図、背景、画風の保持

QPOLAの優位性は upstream README の主張だけでは判定せず、この比較結果で判断する。

## 14. 受け入れ条件

次をすべて満たした時に実装完了とする。

- ComfyUI起動時にAdamW版とQPOLA版の両ノードが登録される。
- CPU-onlyまたはQPOLA非対応環境でも既存AdamWノードを使用できる。
- QPOLAノードは非対応環境で明確に失敗し、AdamWへ切り替わらない。
- 現行slider teacher、MSE loss、prompt YAML契約が変更されていない。
- 既存テストと新規単体テストがすべて成功する。
- build artifactへQPOLA runtime、PTX、source、license、noticeが含まれる。
- NVIDIA CUDA smoke testが成功する。
- reportとsafetensorsからoptimizer種別とQPOLA versionを識別できる。
- 実行成功・失敗の両方でbase modelが復元される。
- READMEにGPU要件、QPOLA設定、A/B比較方法、ライセンス帰属が記載される。

## 15. 実装順序

1. 現行node schemaとoptimizer生成の回帰テストを追加する。
2. 共通schema/execute helperを抽出し、AdamW挙動が不変であることを確認する。
3. `TrainRequest` とoptimizer factoryを追加し、AdamWをfactory経由へ移す。
4. QPOLAをlicense/notice付きでvendorし、package data設定を追加する。
5. QPOLA preflightとlazy importを実装する。
6. QPOLA V3 nodeを追加し、extensionとcompatibility mappingへ登録する。
7. reportとsafetensors metadataを拡張する。
8. 単体、lint、type/static、build/packageテストを実行する。
9. NVIDIA CUDA環境で1-step、3-step smoke testを実行する。
10. A/B評価用workflowとREADMEを整備する。

## 16. 既知のリスク

- 同梱PTXはCUDA 12.4 toolchain生成であり、すべてのdriver/GPU組み合わせでの互換性は未確認。
- QPOLAはparameter tensorをflattenし、32/256要素の局所単位で空間整合性を計算する。LoRA matrixの論理軸とflatten後の局所境界が一致するとは限らない。
- 現行AdamWは一次・二次momentとweight decayを使い、QPOLAは使わないため、同じLRでの比較は公平ではない。
- QPOLAのmoment state削減効果はtrainable LoRA parameterに限定され、base modelとactivationが支配的な総VRAMでは効果が小さい可能性がある。
- `qpola_low_vram=True` の毎step cache解放はOOM耐性よりも速度低下として現れる可能性がある。
- QPOLAの数値的品質とsliderの概念分離性能は未検証であり、実装成功と品質優位性を分けて評価する必要がある。

## 17. 変更予定ファイル

| ファイル | 変更内容 |
|---|---|
| `nodes.py` | 共通helper、QPOLA node schema/execute、V3/legacy登録、metadata保存 |
| `anima_slider_node/training.py` | `TrainRequest`拡張、optimizer factory呼び出し、report拡張 |
| `anima_slider_node/optimizer_factory.py` | optimizer選択、QPOLA preflight、lazy import |
| `anima_slider_node/third_party/qpola/*` | vendored QPOLA runtime、PTX、source、license、notice |
| `pyproject.toml` | sub-package探索とpackage data同梱 |
| `tests/test_training_util.py` | 既存学習回帰の追加 |
| `tests/test_optimizer_factory.py` | optimizer分岐とpreflight単体テスト |
| `tests/test_qpola_node_schema.py` | V3 schemaと登録テスト |
| `tests/test_qpola_packaging.py` | wheel同梱物テスト |
| `README.md` | QPOLAノードの要件、使用方法、比較方法、帰属 |

## 18. 参照実装上の主要シンボル

- `nodes.AnimaSliderTrainLoraNode`
- `training.TrainRequest`
- `training.build_lora_optimizer_param_groups`
- `training.train_lora_from_records`
- `training.flow_loss_for_record`
- `slider_loss.flow_slider_teacher`
- `lora_network.inject_lora_linear_modules`
- `lora_network.restore_linear_modules`
- `QPOLA.optimizer.qpola.QPOLA`
- `QPOLA.optimizer.qpola.cu::qpola_kernel_impl`
