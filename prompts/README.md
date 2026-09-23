# Anima Slider prompt YAML set

Anima Sliderの学習用YAMLです。全身年齢スライダーは、評価したv9/v10を効果名で正規版として採用しています。

## 全身年齢スライダーの正規版

| 効果 | 正規ファイル | 由来 |
| --- | --- | --- |
| 老齢化 | [prompts-anima-aging_slider_fullbody.yaml](prompts-anima-aging_slider_fullbody.yaml) | 旧v9 |
| 幼齢化 | [prompts-anima-deaging_slider_fullbody.yaml](prompts-anima-deaging_slider_fullbody.yaml) | 旧v10 |

- **老齢化**：成人女性を基準に、顔・首・手の加齢を強めます。v7へ`facial wrinkles`を追加したv9の学習文を維持しています。猫背を目的とせず、直立を指定しています。
- **幼齢化**：成人女性を基準に、幼い顔、大きな頭、小さい体、短い手足へ変化させます。単なる成人の若返りではなく、幼児寄りの全身比率を意図しています。v8から`flat color fills, simple cel shading`を除いたv10を維持し、2D・控えめなハイライト・マットな肌と衣服仕様を残しています。

今回の正規化は改名と説明更新のみで、v9/v10の学習文・YAML設定は変更していません。**両方とも単方向学習で、プラスのLoRA強度を使います。** 1本をマイナス適用してもう一方の代用にする設計ではありません。

各8項目で、0～3は全身学習、4～5は上半身学習、6～7は全身評価用。`target = unconditional = neutral`は成人基準、`positive`だけが目的年齢です。YAMLは学習用で、既存LoRAの重みを更新するものではありません。

### 学習設定

| ノード設定 | 評価時に使用した値 |
| --- | --- |
| `prompt_yaml` | 上表の正規YAMLを選択 |
| `custom_prompt_yaml_path` | 空欄、または正規YAMLの絶対パス（こちらが優先） |
| `direction_loss` / `teacher_norm_reference` | **`enhance_only`** / `target` |
| `eta` / `teacher_guidance_scale` | `1.5` / `1.0` |
| YAMLの`guidance_scale` | `1.0`（実効係数1.5） |
| `width` / `height` | `0` / `0`でYAMLの1024×1024を使用。1024 / 1024の明示も可 |
| `steps` / `rank` / `alpha` | `900` / `16` / `16` |
| optimizer / 共通lr / weight decay | AdamW / `1.5e-5` / `0.01` |
| module別lr | self-attention `8e-6`、cross-attention `2.5e-5`、MLP `1.5e-6`（共通lrより優先） |
| `num_inference_steps` / `scheduler_name` | `20` / `simple`（学習内の軌道計算） |
| `timestep_sampling` / `discrete_flow_shift` | `shift` / `3.0` |
| `loss_weighting_scheme` | `none` |
| `prompt_indices` / `eval_prompt_indices` | `0,1,2,3,4,5` / `6,7` |
| `skip_initial_eval` / `skip_final_eval` | 両方`False` |
| `allow_unsafe_age_terms` | 老齢化は`False`で可、**幼齢化は`True`が必要** |
| `output_lora_prefix` | 例：`loras/anima_aging_slider` / `loras/anima_deaging_slider`。モデル名・実験IDも付記 |

YAMLは`direction_loss`・`eta`・学習率を自動変更しません。上記は評価品の出発点であり、すべてのモデルで最適な設定という意味ではありません。

### 評価で確認した範囲

`oneObsession_anima29BV1.safetensors`、共通全身文、seed 1、840×1280、30 steps、CFG4、er_sde/betaで各9枚を評価しました。正しいLoRAの選択と強度はPNGの実行入力で確認しています。

| 正規版 | 使用候補と残る課題 |
| --- | --- |
| 老齢化（旧v9） | +1で明確な老齢化。ただし既にしわが強く、+1.5以上は過密なしわ・表情・衣服変更が目立つ。+0.6～+0.9は次の確認候補で、未検証 |
| 幼齢化（旧v10） | 保持重視なら+1～+1.5、強い幼齢比率なら+2も候補。+2の太い白縁は旧v8より軽減したが、+2.5で再発。高強度で描画・服・靴の変化が残る |

正規採用は副作用解消の宣言ではありません。頭身変化は幼齢化の目的として残します。学習元モデル名は未記録で、方式・文章・係数などが変わった経過なので、改善のすべてを単方向化だけの因果効果とは断定しません。

## 旧版と保存済みワークフローの移行

- v1～v8は[archive](archive/README.md)へ保存し、通常の候補一覧から外しています。初版v1は接尾辞なしの`prompts-anima-age_slider_fullbody.yaml`です。
- `prompts-anima-age_slider_fullbody_v9.yaml` → `prompts-anima-aging_slider_fullbody.yaml`
- `prompts-anima-age_slider_fullbody_v10.yaml` → `prompts-anima-deaging_slider_fullbody.yaml`

ComfyUIを再起動して候補一覧を更新し、保存済みワークフローでは`prompt_yaml`を再選択してください。`custom_prompt_yaml_path`を設定している場合はそちらも新しい絶対パスへ更新します。旧名・旧パスは自動変換されません。旧版の再現はarchive内の絶対パスを指定します。既存の学習済みLoRAファイル名はこのYAML改名では変わりません。

## 他のスライダーへの展開

[単方向化・プロンプト調整の調査手順](doc/slider-direction-investigation.md)に、年齢スライダーの経過、方式だけの対照、方向別の基準文、文章調整、画像比較、採用条件と記録テンプレートをまとめています。

## 胸部サイズ・衣服フィット v2 / v3（方向別の学習例）

検討したC案（基準文の統一）を各8項目のYAMLにしました。**v2は増加／密着用、v3は減少／ゆったり用という別方向であり、v3がv2を置き換えるものではありません。** 元のバージョン接尾辞なしYAMLは比較用に維持しています。

| 概念・方向 | YAML | 成人・着衣の基準 | 出力prefix例 |
| --- | --- | --- | --- |
| 胸部サイズ v2：増加 | [breast_size_slider_v2](prompts-anima-breast_size_slider_v2.yaml) | `moderate bust volume under clothing` | `loras/anima_breast_increase_v2` |
| 胸部サイズ v3：減少 | [breast_size_slider_v3](prompts-anima-breast_size_slider_v3.yaml) | 同じ中程度の胸部 | `loras/anima_breast_decrease_v3` |
| 衣服フィット v2：密着 | [clothing_fit_slider_v2](prompts-anima-clothing_fit_slider_v2.yaml) | `regular clothing fit` | `loras/anima_clothing_fitted_v2` |
| 衣服フィット v3：ゆったり | [clothing_fit_slider_v3](prompts-anima-clothing_fit_slider_v3.yaml) | 同じ標準フィット | `loras/anima_clothing_loose_v3` |

各項目は`target = unconditional = neutral = 元の共通文 + 基準句`。v2の`positive`には元の`positive`全文、v3の`positive`には元の`unconditional`全文を使います。目的側の4句、衣服・髪・画角・背景、成人・着衣の指定を保持し、語句削減などのD案はまだ適用していません。

4種類とも**`direction_loss=enhance_only`で別々に学習し、プラスのLoRA強度で適用**します。`action=enhance`、`guidance_scale=2.0`、896×1152、batch_size=1は元YAMLと同じ。学習indicesは0～5、評価は6～7です。`allow_unsafe_age_terms=False`で検証できます。

[Anima 2.9B正式ワークフロー](../workflows/README.md)を使う場合、対象YAMLと出力prefixを選び直してください。eta1.5 × YAML guidance2.0 × teacher guidance1.0で**実効係数3.0**です。ノード側の明示サイズ1024×1024がYAMLの896×1152より優先されます。比較時はseedの実行後制御を`fixed`にし、初期LoRA乱数の制約も[調査手順](doc/slider-direction-investigation.md)に従って記録します。

新しい候補を一覧へ反映するにはComfyUIを再起動するか、`custom_prompt_yaml_path`へ新YAMLの絶対パスを指定します。2026-09-23に胸部増大v2（Lora8）と衣服ゆったりv3（Lora9）の提供画像を評価しました。胸部減少v3・衣服密着v2は未評価です。結果は[画像評価と胸部v4](doc/lora8-lora9-breast-v4-evaluation.md)、当初の実験設計は[検討文書](doc/breast-size-clothing-fit-investigation.md)を参照してください。

## 胸部増大 v4（目的句を簡潔にした候補）

[prompts-anima-breast_size_slider_v4.yaml](prompts-anima-breast_size_slider_v4.yaml)は、Lora8で見られた腰・太ももの幅増加を受けた**未学習の改善候補**です。v2のpositiveにある4つの目的句を`larger bust volume under clothing`だけへ置換し、基準`moderate bust volume under clothing`との差を容量へ絞りました。基準3ロール、8項目の髪・衣服・画角、全数値はv2と同じです。既存v3は胸部減少用として残します。

設定は上記v2と同じ`enhance_only`、train 0～5 / eval 6～7、guidance 2、eta 1.5、teacher guidance 1、900 steps。Lora8との比較ではノードに1024×1024を明示してください。出力prefix例は`loras/anima_breast_increase_v4`。`custom_prompt_yaml_path`が設定済みなら空欄かv4の絶対パスへ更新します。

腰・脚を固定する機能の追加ではなく、文章差分を狭める実験です。上半身学習は維持するため、**再学習後の全身画像で、同程度の胸部効果における腰・太ももの変化を比較**してください。提供Lora8・Lora9は共通の強度0画像を持ちますが、生成文・seedは各1種類だけです。Lora9の`3.4.png`の実強度は4.0でした。詳しい所見・次案・評価条件は[評価文書](doc/lora8-lora9-breast-v4-evaluation.md)にまとめています。

## 8プロンプト構成のYAML

正規の年齢2種と以下の同梱スライダーは、原則として次の構成です:
- 8 prompts
- neutral == target
- action: enhance
- batch_size: 1
- positive/unconditional differ by the intended slider axis

## ファイル一覧

### 年齢・顔立ち・体型

- `prompts-anima-mature_face_slider.yaml` — `mature_face_slider`。期待される効果: 顔立ちを柔らかく幼い印象から、頬骨や輪郭が明確な成熟した成人顔へ寄せる。
- `prompts-anima-face_roundness_slider.yaml` — `face_roundness_slider`。期待される効果: 顔の輪郭や頬の丸みを強め、角張った輪郭から柔らかく丸い顔立ちへ寄せる。
- `prompts-anima-breast_size_slider.yaml` — `breast_size_slider`。期待される効果: 服を着た上半身の胸部ボリュームを、小さめのシルエットから大きめのシルエットへ調整する。
- `prompts-anima-waist_width_slider.yaml` — `waist_width_slider`。期待される効果: 服を着た状態の腰まわりの幅を、太めの直線的なシルエットから細めでくびれのある形へ調整する。

### 表情・髪

- `prompts-anima-anger_intensity_slider.yaml` — `anger_intensity_slider`。期待される効果: 眉、目つき、口元の緊張を強め、穏やかな表情から怒りの強い表情へ変化させる。
- `prompts-anima-blush_intensity_slider.yaml` — `blush_intensity_slider`。期待される効果: 頬の赤みや血色感を強め、自然な肌色から照れ・高揚・温かみのある表情へ変化させる。
- `prompts-anima-emotion_valence_slider.yaml` — `emotion_valence_slider`。期待される効果: 表情全体の感情トーンを、沈んだ雰囲気から明るく快活な雰囲気へ寄せる。
- `prompts-anima-eye_openness_slider.yaml` — `eye_openness_slider`。期待される効果: まぶたの開き具合を強め、半目やリラックスした目元から大きく開いた alert な目元へ変化させる。
- `prompts-anima-smile_intensity_slider.yaml` — `smile_intensity_slider`。期待される効果: 口元と表情の笑顔感を強め、落ち着いた表情から明るい笑顔へ寄せる。
- `prompts-anima-hair_messiness_slider.yaml` — `hair_messiness_slider`。期待される効果: 髪の乱れ、ほつれ毛、ばらつきを増やし、整った髪からラフで動きのある髪へ寄せる。

### ポーズ・姿勢

- `prompts-anima-pose_dynamism_slider.yaml` — `pose_dynamism_slider`。期待される効果: 身体のジェスチャーや動きを強め、静かな立ち姿からアクション性のあるポーズへ寄せる。
- `prompts-anima-posture_uprightness_slider.yaml` — `posture_uprightness_slider`。期待される効果: 背筋や肩の水平感を整え、猫背気味の姿勢からまっすぐ立った姿勢へ変化させる。

### 衣服・衣装デザイン

- `prompts-anima-clothing_fit_slider.yaml` — `clothing_fit_slider`。期待される効果: 衣服のフィット感を、ゆったりした形から体に沿ったすっきりしたシルエットへ寄せる。
- `prompts-anima-collar_height_slider.yaml` — `collar_height_slider`。期待される効果: 襟の高さや首元の覆い方を強め、開いた襟元から高く構造的な襟へ変化させる。
- `prompts-anima-outfit_decoration_slider.yaml` — `outfit_decoration_slider`。期待される効果: 衣装の刺繍、縁取り、柄、装飾密度を増やし、簡素な服から華やかな衣装へ寄せる。
- `prompts-anima-skirt_length_slider.yaml` — `skirt_length_slider`。期待される効果: スカート丈を短めから長めへ寄せ、裾の位置や衣装全体の落ち着いた印象を調整する。

### 画風・背景ディテール

- `prompts-anima-background_detail_slider.yaml` — `background_detail_slider`。期待される効果: 背景の装飾、棚、壁面情報などを増やし、シンプルな空間から情報量の多い環境へ寄せる。
- `prompts-anima-line_weight_slider.yaml` — `line_weight_slider`。期待される効果: 線画の太さや輪郭線の存在感を強め、繊細な線から太くはっきりしたアニメ調の線へ寄せる。
- `prompts-anima-pastel_tone_slider.yaml` — `pastel_tone_slider`。期待される効果: 色調を淡く柔らかいパステル寄りにし、濃く彩度の高い色から軽い印象へ変化させる。

### エネルギー・魔法・SFエフェクト

- `prompts-anima-aura_intensity_slider.yaml` — `aura_intensity_slider`。期待される効果: 汎用的なオーラの明るさや輪郭を強め、人物の存在感、神秘性、威圧感を高める。
- `prompts-anima-energy_aura_intensity_slider.yaml` — `energy_aura_intensity_slider`。期待される効果: キャラクターや物体を包むエネルギーオーラの発光と密度を強め、力の解放感を出す。
- `prompts-anima-magic_glow_intensity_slider.yaml` — `magic_glow_intensity_slider`。期待される効果: 魔法陣、杖、手元、エフェクトの発光を強め、ファンタジー的な発動感を明確にする。
- `prompts-anima-electric_arc_intensity_slider.yaml` — `electric_arc_intensity_slider`。期待される効果: 電撃の弧、稲妻状の放電、発光を強め、雷・機械・魔法エネルギーの鋭さを増す。
- `prompts-anima-spark_amount_slider.yaml` — `spark_amount_slider`。期待される効果: 火花の数を増やし、金属衝突、電気的なショート、魔法発動時の瞬間的な勢いを加える。
- `prompts-anima-particle_amount_slider.yaml` — `particle_amount_slider`。期待される効果: 浮遊粒子や光点を増やし、魔法、空気感、余韻、画面密度を自然に足す。
- `prompts-anima-laser_beam_intensity_slider.yaml` — `laser_beam_intensity_slider`。期待される効果: レーザー光の明るさ、発光、エネルギー感を強め、ビーム攻撃やSF的演出を派手にする。
- `prompts-anima-beam_thickness_slider.yaml` — `beam_thickness_slider`。期待される効果: ビームの太さと存在感を増やし、細い光線から重量感のある強力な放射へ寄せる。

### 炎・爆発・熱・破壊エフェクト

- `prompts-anima-explosion_intensity_slider.yaml` — `explosion_intensity_slider`。期待される効果: 爆発の火球、衝撃波、破片、煙を強め、小さな炸裂から大きな爆発演出へ寄せる。
- `prompts-anima-shockwave_strength_slider.yaml` — `shockwave_strength_slider`。期待される効果: 衝撃波の圧力リングや空気の歪み、巻き上がる砂塵を強め、攻撃や爆発後のインパクトを大きく見せる。
- `prompts-anima-debris_amount_slider.yaml` — `debris_amount_slider`。期待される効果: 破片や瓦礫の量を増やし、崩壊・衝突・爆発シーンの荒々しさとスケール感を強調する。
- `prompts-anima-dust_cloud_density_slider.yaml` — `dust_cloud_density_slider`。期待される効果: 土煙や粉塵の濃度を上げ、地面への衝撃、移動、破壊の余韻をより重く演出する。
- `prompts-anima-flame_intensity_slider.yaml` — `flame_intensity_slider`。期待される効果: 炎の高さ、明るさ、燃焼感を強め、鍛冶場・焚き火・魔法炎などの熱量を増やす。
- `prompts-anima-ember_amount_slider.yaml` — `ember_amount_slider`。期待される効果: 火の粉や燃えさしを増やし、炎まわりの空気感、余熱、ドラマ性を追加する。
- `prompts-anima-heat_haze_strength_slider.yaml` — `heat_haze_strength_slider`。期待される効果: 熱による背景の揺らぎや空気の屈折を強め、高温環境や強い炎の存在感を出す。
- `prompts-anima-smoke_density_slider.yaml` — `smoke_density_slider`。期待される効果: 煙の濃さや広がりを増やし、火災後・爆発後・工業的な場面の視覚的な重さを高める。

### 水・天候・空気感エフェクト

- `prompts-anima-fog_density_slider.yaml` — `fog_density_slider`。期待される効果: 霧の量と奥行き感を強め、森・湿地・夜明けの場面をより幻想的または不穏に見せる。
- `prompts-anima-rain_intensity_slider.yaml` — `rain_intensity_slider`。期待される効果: 雨粒の密度や降り方を強め、天候の厳しさ、湿った空気、ドラマチックな雰囲気を追加する。
- `prompts-anima-droplet_amount_slider.yaml` — `droplet_amount_slider`。期待される効果: 表面や空中の水滴を増やし、雨上がり、濡れ感、水辺の細かな質感を補強する。
- `prompts-anima-steam_amount_slider.yaml` — `steam_amount_slider`。期待される効果: 蒸気や湯気を増やし、温泉、機械、冷却、魔法効果などの湿度と熱気を表現する。
- `prompts-anima-water_splash_intensity_slider.yaml` — `water_splash_intensity_slider`。期待される効果: 水しぶきの高さ、広がり、勢いを強め、水場での衝突や動作を派手に見せる。

### 動き・漫画的強調エフェクト

- `prompts-anima-motion_line_intensity_slider.yaml` — `motion_line_intensity_slider`。期待される効果: 動線やスピード線を強め、キャラクターや物体の移動速度、勢い、アニメ的な躍動感を増す。
- `prompts-anima-slash_trail_intensity_slider.yaml` — `slash_trail_intensity_slider`。期待される効果: 斬撃の軌跡や光の残像を強め、剣・爪・魔法斬撃の方向性と迫力を明確にする。
- `prompts-anima-impact_line_density_slider.yaml` — `impact_line_density_slider`。期待される効果: 集中線や衝撃線の密度を上げ、打撃や爆発の焦点、漫画的なインパクトを強める。

## 過去の全身年齢YAML

旧v1～v8の一覧と作成時の記録は[archive](archive/README.md)を参照してください。
