# 全身年齢スライダー旧版

v1～v8の学習用YAMLを本文そのままで保存しています。正規版は[老齢化（旧v9）](../prompts-anima-aging_slider_fullbody.yaml)と[幼齢化（旧v10）](../prompts-anima-deaging_slider_fullbody.yaml)です。

| 版 | 方向・用途 | ファイル |
| --- | --- | --- |
| v1 | 初期の両方向 | [prompts-anima-age_slider_fullbody.yaml](prompts-anima-age_slider_fullbody.yaml) |
| v2 | 両方向 | [prompts-anima-age_slider_fullbody_v2.yaml](prompts-anima-age_slider_fullbody_v2.yaml) |
| v3 | 両方向 | [prompts-anima-age_slider_fullbody_v3.yaml](prompts-anima-age_slider_fullbody_v3.yaml) |
| v4 | 両方向 | [prompts-anima-age_slider_fullbody_v4.yaml](prompts-anima-age_slider_fullbody_v4.yaml) |
| v5 | 単方向・老齢化 | [prompts-anima-age_slider_fullbody_v5.yaml](prompts-anima-age_slider_fullbody_v5.yaml) |
| v6 | 単方向・幼齢化 | [prompts-anima-age_slider_fullbody_v6.yaml](prompts-anima-age_slider_fullbody_v6.yaml) |
| v7 | 単方向・老齢化、しわと表情の調整 | [prompts-anima-age_slider_fullbody_v7.yaml](prompts-anima-age_slider_fullbody_v7.yaml) |
| v8 | 単方向・幼齢化、描画と衣服の調整 | [prompts-anima-age_slider_fullbody_v8.yaml](prompts-anima-age_slider_fullbody_v8.yaml) |

通常の`prompt_yaml`候補は`prompts/`直下だけを列挙するため、旧版は一覧に表示されません。再現する場合は`custom_prompt_yaml_path`にこのフォルダー内のYAMLの絶対パスを指定してください。以前のファイル名・絶対パスは自動変換されません。YAML自体は学習方式を切り替えません。

v9/v10は上記の正規名へ改名しました。本文は同じで、ここに重複した版名ファイルは置いていません。現在の設定と評価は[親README](../README.md)、他スライダーへの展開は[調査手順](../doc/slider-direction-investigation.md)を参照してください。

以下は旧版作成時の設定・判断を残した履歴です。「未検証」「新YAML」などは当時時点の記述で、現在の推奨設定ではありません。特にv5/v6のeta=1.0・896×1152は初期提案で、後の実学習条件eta=1.5・1024×1024とは異なります。

## v7 / v8作成時の記録

- [v7: 老齢化](prompts-anima-age_slider_fullbody_v7.yaml): v5のしわ列挙を整理し、目元の細い線・ほうれい線・顔のたるみ・首・手の加齢を残しました。全4ロールの表情を`neutral expression, relaxed brow`に揃えています。追加生成で笑顔寄りの結果も出たため、`relaxed closed mouth`は採用していません。直立姿勢・背景・衣装・画角はv5を引き継ぎます。
- [v8: 幼齢化](prompts-anima-age_slider_fullbody_v8.yaml): v6の`positive`から`smooth skin`と`large clear eyes`を除き、`tiny nose, small mouth`、大きな頭・短い体・短い手足・小さな手は維持しました。全4ロールで2Dアニメ・セル塗り・控えめなハイライト・マットな肌の描画条件を揃え、既存衣装に合う襟・袖・裾・靴の仕様を具体化しました。衣装の種類を維持し、裾入れは適するトップスだけに指定しています。

両方とも8項目で、0～3が全身学習、4～5が上半身学習、6～7が全身評価用です。各項目で`target = unconditional = neutral`が成人基準、`positive`だけが目的年齢です。v7とv8の成人基準は、表情・描画・衣装の変更方針が異なるため同一ではありません。**両方ともプラスのLoRA強度で目的方向へ適用**します。

これらは生成時のポジティブ文ではなく、**Slider学習用YAML**です。作成済みv5/v6 LoRAへ自動反映されるものではなく、新たな学習が必要です。その後、提供されたv7/v8の学習結果と生成画像を評価しました。現在の正規版と評価は[親README](../README.md)を参照してください。共通の文章指定は、背景・衣服・画風の固定を保証しません。特に+3は学習時の+1を超える適用であり、副作用の解消を保証できません。

初回は、提供されたv5/v6の**実際の学習条件**に合わせて比較します。下の旧v5/v6節にある初期提案のeta=1.0・896×1152とは異なります。

| ノード設定 | v7 / v8の初回比較値 |
| --- | --- |
| `prompt_yaml` | 対応するv7またはv8のYAML |
| `custom_prompt_yaml_path` | 空欄。絶対パスで指定する場合は対応する新YAMLに更新（こちらが優先） |
| `direction_loss` | **`enhance_only`** |
| `teacher_norm_reference` | `target` |
| `eta` / `teacher_guidance_scale` | **`1.5` / `1.0`** |
| YAMLの`guidance_scale` | `1.0`（設定済み。実効係数は1.5） |
| `width` / `height` | **`0` / `0` → YAMLの1024×1024**。ノード側で1024 / 1024の明示も可 |
| `steps` / `rank` / `alpha` | `900` / `16` / `16` |
| optimizer / 共通lr | AdamW / `1.5e-5` |
| module別lr | self-attention `8e-6`、cross-attention `2.5e-5`、MLP `1.5e-6`（共通lrより優先） |
| `num_inference_steps` / `scheduler_name` | `20` / `simple`（学習内の軌道計算） |
| `timestep_sampling` / `discrete_flow_shift` | `shift` / `3.0` |
| `loss_weighting_scheme` | `none` |
| `prompt_indices` / `eval_prompt_indices` | `0,1,2,3,4,5` / `6,7` |
| `skip_initial_eval` / `skip_final_eval` | **両方`False`** |
| `allow_unsafe_age_terms` | v7は`False`で可、v8は`True`が必要 |
| `output_lora_prefix` | `loras/anima_age_slider_v7` / `loras/anima_age_slider_v8` |

YAMLは`eta`・`direction_loss`・学習率などのノード設定を変更しません。学習元チェックポイント名も記録してください。過去の提供レポートには学習元ファイル名がないため、生成先モデルと同じだったとは確認できていません。

生成比較はv7対v5、v8対v6で、元の生成文と追加生成文、seed 1 / 10000、強度0～3（0.5刻み）を揃えます。v7は老齢感を維持してしわの過密・険しさが減るかを確認し、笑顔化だけで改善と判定しません。v8は低頭身・短い手足を残して光沢や服の仕様変更が減るかを確認します。同じ強度に加え、同程度の年齢変化で副作用を比較し、年齢効果が弱まっただけの結果と区別します。

新YAMLを候補一覧に表示するにはComfyUIを再起動するか、`custom_prompt_yaml_path`に対象ファイルの絶対パスを指定してください。

## v5 / v6作成時の記録

- [v5: 老齢化](prompts-anima-age_slider_fullbody_v5.yaml): 成人女性を基準に、顔・首・手の加齢表現を増やします。猫背、こけた頬、細い腕、過度な痩身の指定を外し、両側で直立姿勢を指定しています。
- [v6: 幼齢化](prompts-anima-age_slider_fullbody_v6.yaml): 同じ成人女性を基準に幼齢化します。幼い顔に加え、頭が体に対して大きくなる変化、低頭身、短い手足、小さな手を意図した変化として残しています。

v4の8項目の衣装・画角バリエーションを引き継ぎ、学習用は全身4件＋上半身2件、評価用は全身2件です。足元を見やすくするため全身項目の裾は足首丈にし、靴を指定しました。髪型・背景・照明・表情・姿勢・衣装は各項目の全ロールで揃え、繰り返しの年齢語や背景保持の命令文を整理しています。これらの文章は背景・人物を固定する機能ではなく、副作用の低減は未検証です。

両ファイルとも `target = unconditional = neutral` が成人の基準文、`positive` が目的年齢です。`unconditional` は空欄にせず、差し引く成人の参照として使います。**v5もv6もプラスのLoRA重みで目的方向へ適用**します。マイナス方向の動作は学習対象にしません。

初期比較用のノード設定:

| 設定 | 値 |
| --- | --- |
| `prompt_yaml` | v5またはv6のファイルを選択 |
| `custom_prompt_yaml_path` | 空欄（設定済みなら選択したYAMLより優先されます） |
| `direction_loss` | **`enhance_only`** |
| `teacher_norm_reference` | `target` |
| `eta` / `teacher_guidance_scale` | `1.0` / `1.0` |
| YAMLの`guidance_scale` | `1.0`（各ファイルに設定済み） |
| `width` / `height` | `0` / `0`（YAMLの896×1152を使用） |
| `prompt_indices` | `0,1,2,3,4,5` |
| `eval_prompt_indices` | `6,7` |
| `skip_initial_eval` / `skip_final_eval` | 評価を行う場合は両方`False` |
| `allow_unsafe_age_terms` | v5は`False`で可、v6は`True`が必要 |
| `output_lora_prefix` | 例: `loras/anima_age_slider_v5` / `loras/anima_age_slider_v6` |

YAMLは `direction_loss` や `eta` を切り替えません。保存済みワークフローからv4の設定を使う場合は上記をノード側で変更してください。この組み合わせの実効係数は `eta × guidance_scale × teacher_guidance_scale = 1.0` です。既存v4レポートの6.25より弱い初期候補であり、推奨値の品質検証はまだ行っていません。

rank、alpha、steps、学習率などは初回は既存設定を引き継ぎ、学習元と生成先のチェックポイントを揃えて比較します。YAML以外にも片方向学習・係数・解像度を変更するため、v4との単一要因の比較ではありません。各変更の寄与を調べる場合は一要因ずつ比較してください。

生成時はv5とv6を別々に、同じ全身プロンプト・同じシードで `0, 0.25, 0.5, 0.75, 1.0` から確認します。重みが同じ場合に加えて、同程度の年齢変化が得られた場合の背景・画風・衣装・髪型・姿勢を比較してください。v6では頭身と手足の比率の変化を目的の効果に含めます。学習時の数値評価だけでは画像品質や副作用は判定できません。

追加YAMLを候補一覧に表示するにはComfyUIを再起動してください。`custom_prompt_yaml_path` に対象ファイルの絶対パスを指定して読み込むこともできます。
