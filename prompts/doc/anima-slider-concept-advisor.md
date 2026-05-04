# Anima Slider LoRA Concept Advisor

この文書は、Anima/Cosmos RFlow 向け text-only Slider LoRA の「作るべき概念」を提案・評価するための Local LLM 用ガイドである。目的は、思いついた単語をそのまま prompt 化することではなく、学習差分がきれいで、8 prompt 程度でも効果を確認しやすい slider 候補へ変換することである。

想定する使い方:

```text
この文書を参考に、次のお題から Slider LoRA 候補を提案してください。
お題: <ユーザーのお題>
目的: <任意。生成で何を調整したいか>
制約: <任意。避けたい方向、対象、構図など>
出力数: <任意。通常は3-5候補>
```

LLM は、曖昧なお題でも質問を返す前に安全で実用的な解釈を1つ置き、複数の候補へ分解して提案する。危険・曖昧・広すぎる概念は、そのまま採用せず、より狭い視覚属性へ置き換える。

---

## 1. 最重要原則

Slider LoRA で学習したいのは、`positive` と `unconditional` の差分である。良い概念とは、次の質問にすべて「はい」と言いやすい概念である。

1. 画像を並べたとき、A/B の違いを人間が見て判断できるか。
2. 弱い、中間、強いという連続量として想像できるか。
3. 両端をそれぞれ 2-4 個程度の短い英語タグで書けるか。
4. 差分が顔、髪、服、背景、画風、構図などに分裂しないか。
5. `target` と `neutral` から目的語を抜いても自然な prompt になるか。
6. 8 prompts の中で髪色・構図・服装を変えても意味が崩れないか。
7. 反対側を `not ...`, `no ...`, `bad ...`, `low quality` で作らず、同品質の自然な反対属性で書けるか。

迷ったら「1枚の画像で視覚的に測れる単一軸か？」を優先する。

---

## 2. 採用判断スコア

各候補を 0-2 点で採点する。合計 14 点以上は優先候補、10-13 点は要注意候補、9 点以下は後回しまたは分解対象とする。

| 項目 | 0 点 | 1 点 | 2 点 |
| --- | --- | --- | --- |
| 視覚明瞭性 | 見ても判定しづらい | 条件次第で見える | A/B で明確に見える |
| 連続性 | 強弱にしづらい | 2値なら可能 | slider 強度で段階変化しそう |
| 語彙の明確さ | 両端のタグが曖昧 | 片側だけ明確 | 両端を 2-4 語で書ける |
| 概念純度 | 複数属性が必ず混ざる | 混ざりやすいが制御可能 | 差分を単一概念にしやすい |
| 安全性 | 性的・未成年・侮辱的文脈に寄る | 注意語で制御が必要 | adult / fully clothed / nonsexual で安定 |
| 汎化性 | 特定構図専用になりやすい | 構図を選べば可能 | 複数構図で成立する |
| 評価容易性 | 成否判断が主観的 | 代表例なら見られる | 失敗症状を具体的に分類できる |
| 実用性 | 使い道が狭い | 実験として有用 | 生成調整で繰り返し使える |

採点は楽観しない。`cute`, `beautiful`, `cool`, `sexy`, `quality` のような広い概念は、視覚明瞭性や実用性が高そうに見えても、概念純度と語彙の明確さを低くする。

---

## 3. お題を Slider 候補へ変換する手順

### Step 1: お題を分解する

ユーザーのお題が広い場合は、まず視覚的に独立した軸へ分解する。

例:

| 元のお題 | そのまま採用しない理由 | 分解後の候補 |
| --- | --- | --- |
| かわいい | 年齢、目、顔形状、表情、服装へ分散する | `eye_size_slider`, `face_roundness_slider`, `soft_smile_slider`, `pastel_color_palette_slider` |
| 美人 | 顔立ち、画風、品質、年齢、メイクへ分散する | `facial_contour_sharpness_slider`, `clean_lineart_slider`, `color_saturation_slider` |
| セクシー | 性的文脈に寄りやすく、服装・体型・ポーズが混ざる | `clothing_fit_slider`, `confident_pose_slider`, `formal_glamour_outfit_slider` |
| 迫力 | ポーズ、カメラ、表情、光、背景密度へ分散する | `dynamic_pose_slider`, `intense_expression_slider`, `dramatic_lighting_slider` |
| 大人っぽい | 年齢・体型・服装・表情が混ざる | `older_adult_face_slider`, `formal_outfit_slider`, `calm_expression_slider` |
| 高品質 | 低品質側を作ると汚くなる | `rendering_detail_slider`, `line_weight_slider`, `clean_shading_slider` |

### Step 2: 目的概念を1文で定義する

悪い定義:

```text
かわいさを上げる。
```

良い定義:

```text
成人キャラクターの年齢感や服装を固定したまま、目の大きさだけを連続的に調整する。
```

### Step 3: 両端の英語タグを作る

原則として、positive / unconditional はそれぞれ 2-4 語にする。長くしすぎると差分が濁る。

良い例:

```text
positive: wide open eyes, alert eyes, visible eyelids
unconditional: half-closed eyes, relaxed eyelids, sleepy eyes
```

悪い例:

```text
positive: very cute beautiful young energetic shiny eyes, happy smile, colorful outfit
unconditional: not cute, plain, bad face, tired, low quality
```

### Step 4: 混ざりやすい属性を先に書く

候補ごとに、何へ吸着しやすいかを明記する。

例:

```text
eye_openness_slider は、表情、眠そうな雰囲気、目の大きさ、口元へ混ざりやすい。
対策: neutral expression, relaxed mouth, adult face を共通条件に入れる。
```

### Step 5: 構図を決める

概念が見える構図を優先する。構図そのものを slider にしたい場合以外、positive/unconditional で構図を変えない。

| 概念タイプ | 推奨構図 |
| --- | --- |
| 表情・目・眉・顔形状 | portrait / close-up / upper body |
| 髪 | portrait / upper body / waist up |
| 胸・ウエスト・肩・筋肉 | waist up / thigh-up / full body |
| 服の fit / 布 / 装飾 | upper body / waist up / thigh-up |
| ポーズ | upper body / full body、ただし構図変化に注意 |
| 背景 | upper body などを固定し、背景だけを変える |
| 画風・色・線 | 構図を固定し、複数の被写体条件で確認する |

---

## 4. 優先カテゴリ別の候補辞書

### 4.1 表情・感情

| 候補名 | positive | unconditional | 推奨度 | 主なリスク |
| --- | --- | --- | --- | --- |
| `smile_intensity_slider` | `big smile, cheerful expression, smiling eyes` | `neutral expression, relaxed mouth, calm face` | 高 | 口の開き、歯見せ、幼さ |
| `emotion_valence_slider` | `bright happy expression, cheerful mood, lively eyes` | `gloomy expression, melancholic mood, subdued face` | 高 | 怒り、涙、背景の暗さ |
| `anger_calmness_slider` | `angry expression, intense glare, furrowed brows` | `calm expression, relaxed eyes, soft brows` | 中 | 眉・目・口が同時に動く |
| `embarrassment_slider` | `embarrassed expression, blushing, shy eyes` | `composed expression, calm face, steady gaze` | 中 | blush slider へ吸着 |
| `confidence_expression_slider` | `confident expression, steady gaze, slight smirk` | `uncertain expression, hesitant eyes, timid face` | 中 | ポーズや服装へ吸着 |

推奨条件: `adult woman, fully clothed, nonsexual, solo, portrait, simple background`。

### 4.2 顔パーツ・顔形状

| 候補名 | positive | unconditional | 推奨度 | 主なリスク |
| --- | --- | --- | --- | --- |
| `eye_openness_slider` | `wide open eyes, alert eyes, visible eyelids` | `half-closed eyes, relaxed eyelids, sleepy eyes` | 高 | 感情、眠気、目の大きさ |
| `eye_size_slider` | `large eyes, big eyes, wide iris` | `small eyes, narrow eyes, compact iris` | 中 | 年齢、かわいさ |
| `face_roundness_slider` | `round face, soft cheeks, gentle facial contour` | `angular face, sharp jawline, defined facial contour` | 中高 | 年齢、性別印象 |
| `jawline_definition_slider` | `defined jawline, sharp chin, angular lower face` | `soft jawline, rounded chin, smooth lower face` | 中 | 顔の年齢・性別印象 |
| `eyebrow_angle_slider` | `arched eyebrows, sharp eyebrows, angled brows` | `soft eyebrows, relaxed eyebrows, gentle brows` | 中 | 表情へ吸着 |

対策: 顔形状では `adult face`, `neutral expression`, `hair away from face` を共通条件に置く。髪型で輪郭が隠れる prompt を増やしすぎない。

### 4.3 髪

| 候補名 | positive | unconditional | 推奨度 | 主なリスク |
| --- | --- | --- | --- | --- |
| `hair_length_slider` | `long hair, flowing hair` | `short hair, cropped hair` | 高 | 構図依存、髪型変化 |
| `hair_volume_slider` | `voluminous hair, fluffy hair, thick hair` | `flat hair, sleek hair, low volume hair` | 高 | 髪質、髪型 |
| `hair_messiness_slider` | `messy hair, tousled hair, stray hair` | `neat hair, tidy hair, smooth hair` | 中 | 状況描写、キャラ性 |
| `hair_curliness_slider` | `curly hair, wavy hair, ringlets` | `straight hair, smooth straight hair` | 中 | 髪型そのもの |

対策: 髪色は nuisance 条件として分散してよい。ただし hair color slider でない限り、髪色を positive/unconditional の差分に入れない。

### 4.4 体型・身体比率

身体関連は実用性が高い一方で、性的文脈、年齢感、服装、ポーズへ吸着しやすい。必ず `adult woman, fully clothed, nonsexual` を全フィールドに入れる。露出を増やさず、視認性のある安全な服装で固定する。

| 候補名 | positive | unconditional | 推奨度 | 主なリスク |
| --- | --- | --- | --- | --- |
| `bust_size_slider` | `large breasts, fuller bust, natural breast volume` | `small breasts, slender chest, modest bust` | 中 | 性的文脈、服装変化 |
| `waist_width_slider` | `narrow waist, slim waistline, defined waist` | `wide waist, straight waistline, broad waist` | 中 | 腰幅、服シルエット |
| `muscle_tone_slider` | `toned body, athletic build, defined arms` | `soft body, non-muscular build, smooth arms` | 中 | ポーズ、服装、性別印象 |
| `shoulder_width_slider` | `broad shoulders, structured shoulders` | `narrow shoulders, delicate shoulders` | 中 | 性別印象、服装 |
| `height_impression_slider` | `tall body proportions, long limbs` | `short body proportions, compact proportions` | 低中 | 単体画像では評価困難 |

避ける語: `child`, `teen`, `minor`, `young girl`, `schoolgirl`, `sexy`, `nude`, `skimpy`, `erotic`, `fat`, `obese`, `skinny`。

### 4.5 服装・布の状態

服装系は身体 slider より安全で、prompt 差分を単一化しやすい。服カテゴリを `target` に置き、positive/unconditional では fit、厚み、装飾量などの修飾だけを変える。

| 候補名 | positive | unconditional | 推奨度 | 主なリスク |
| --- | --- | --- | --- | --- |
| `clothing_fit_slider` | `fitted clothing, tailored fit, close fitting outfit` | `loose clothing, oversized fit, relaxed fit outfit` | 高 | 体型 slider 化 |
| `fabric_weight_slider` | `thick fabric, heavy fabric, structured cloth` | `light fabric, thin fabric, soft drape` | 中 | 透け表現、季節感 |
| `ornament_density_slider` | `ornate outfit, detailed trim, decorative accents` | `plain outfit, simple design, minimal trim` | 中 | 画風や背景密度 |
| `formality_slider` | `formal outfit, tailored jacket, polished clothes` | `casual outfit, relaxed clothes, simple casual wear` | 中 | 職業、場面 |

### 4.6 ポーズ・構図

ポーズ系は効果が出ても、手足破綻や構図変化を誘発しやすい。最初は単純な姿勢や視線から始める。

| 候補名 | positive | unconditional | 推奨度 | 主なリスク |
| --- | --- | --- | --- | --- |
| `gaze_direction_slider` | `looking at viewer, direct gaze` | `looking away, averted gaze` | 中高 | 顔向き、構図 |
| `posture_uprightness_slider` | `upright posture, straight back` | `slouched posture, relaxed shoulders` | 中 | 感情、年齢 |
| `dynamic_pose_slider` | `dynamic pose, action pose, energetic pose` | `relaxed standing pose, still pose, calm pose` | 低中 | 構図変化、手足破綻 |
| `head_tilt_slider` | `tilted head, slight head tilt` | `straight head, level head position` | 中 | 表情・可愛さ |

カメラ距離や構図そのもの、例えば `close-up` と `full body` の対比は、通常 prompt や workflow 側で制御した方が安定しやすい。

### 4.7 背景・環境

背景系は人物属性から分離しやすい。人物、服、構図、画風を固定し、背景語だけを差分にする。

| 候補名 | positive | unconditional | 推奨度 | 主なリスク |
| --- | --- | --- | --- | --- |
| `background_detail_slider` | `detailed background, decorated room, rich background elements` | `plain background, simple background, minimal background` | 高 | 人物ディテールへ波及 |
| `natural_urban_background_slider` | `natural background, trees, greenery` | `urban background, buildings, city scenery` | 中 | 色調、光 |
| `indoor_outdoor_slider` | `indoor room, interior background` | `outdoor scenery, open sky background` | 中 | 光と場面が変わる |
| `background_depth_slider` | `deep perspective background, layered depth` | `flat simple background, shallow background` | 中 | 構図 |

### 4.8 画風・レンダリング

画風系は強く効きやすいが、モデル本来の画風や品質へ波及しやすい。低品質を反対側にしない。

| 候補名 | positive | unconditional | 推奨度 | 主なリスク |
| --- | --- | --- | --- | --- |
| `line_weight_slider` | `bold outlines, thick lineart, strong contour lines` | `thin lineart, delicate outlines, fine contour lines` | 中 | 画風全体 |
| `color_saturation_slider` | `vivid colors, high saturation, rich colors` | `muted colors, low saturation, subdued colors` | 中高 | 光、背景 |
| `lighting_key_slider` | `bright lighting, high key lighting, soft bright light` | `dim lighting, low key lighting, subdued light` | 中 | 場面、感情 |
| `rendering_detail_slider` | `detailed rendering, intricate shading, refined details` | `simple rendering, flat shading, minimal details` | 中 | 品質差 |

避ける語: `low quality`, `bad anatomy`, `blurry`, `worst quality`。

---

## 5. 後回し・禁止に近い概念

次の概念は、そのまま slider 化しない。必ず分解または置換する。

| 元概念 | 理由 | 代替 |
| --- | --- | --- |
| `beauty_slider` | 美しさが顔、画風、年齢、服装へ分散する | 顔輪郭、線、色、表情に分解 |
| `cute_slider` | 年齢・目・顔丸さ・服装に混ざる | 目の大きさ、顔の丸さ、笑顔、色調に分解 |
| `sexy_slider` | 性的文脈に寄りやすい | clothing fit、formal glamour、confident pose |
| `young_slider` | 未成年方向へ流れるリスク | adult-only の older/younger adult face。ただし慎重に |
| `quality_slider` | 反対側が低品質化しやすい | rendering detail、line weight、shading style |
| `camera_distance_slider` | 通常 prompt 制御向き | workflow / prompt 側で制御 |
| `style_of_specific_artist_slider` | 権利・作家性・画風全体の混入リスク | 一般的な線、色、塗り、構図語に分解 |

---

## 6. 出力フォーマット

LLM は必ず次の順番で出力する。冗長な説明や長い雑談はしない。

```text
## 結論
推奨: <最も良い候補名>
理由: <1-2文>
前提: <曖昧なお題をどう解釈したか>

## 候補一覧
| 優先 | 候補名 | 目的 | positive | unconditional | スコア | 判定 |
| --- | --- | --- | --- | --- | ---: | --- |
| 高 | ... | ... | ... | ... | 15/16 | 採用 |

## 個別提案
### 1. <candidate_name>
目的: <何を連続調整するか>
positive: <2-4語>
unconditional: <2-4語>
推奨構図: <portrait 4, close-up 2, upper body 2 のように8枠で>
固定したい条件: <共通条件>
避ける語: <混入しやすい語>
想定リスク: <何に吸着しやすいか>
対策: <targetやprompt設計でどう抑えるか>
初回検証: 8 prompts / guidance_scale 2.0 / 896x1152 / action enhance
採用スコア: <0-16>
判定: 採用 / 要注意 / 後回し

## 後回しにした案
- <候補>: <理由>

## 次に作る prompt set 用の概念仕様
concept_name: <candidate_name>
category: <expression | face | hair | body | clothing | pose | background | rendering>
positive_terms: <comma-separated English tags>
unconditional_terms: <comma-separated English tags>
common_conditions: <comma-separated English tags>
recommended_layout: <8 prompt の構図配分>
avoid_terms: <comma-separated terms>
```

最後の「次に作る prompt set 用の概念仕様」は、8 prompt set generator に渡せる形にする。

---

## 7. 良い出力例

入力:

```text
お題: キャラをもっとかわいく調整できる slider
目的: 汎用キャラ生成で使いたい
```

出力例:

```text
## 結論
推奨: face_roundness_slider
理由: 「かわいさ」は広すぎるため、そのまま slider 化すると年齢・目・服装・画風に分散する。まず成人キャラの顔の丸さだけを動かす方が、視覚的に評価しやすい。
前提: 未成年方向には寄せず、adult woman の安全なキャラ造形として解釈する。

## 候補一覧
| 優先 | 候補名 | 目的 | positive | unconditional | スコア | 判定 |
| --- | --- | --- | --- | --- | ---: | --- |
| 高 | face_roundness_slider | 顔の丸さを調整する | round face, soft cheeks, gentle facial contour | angular face, sharp jawline, defined facial contour | 14/16 | 採用 |
| 中 | eye_size_slider | 目の大きさを調整する | large eyes, big eyes, wide iris | small eyes, narrow eyes, compact iris | 13/16 | 要注意 |
| 中 | soft_smile_slider | 表情の柔らかさを調整する | soft smile, gentle expression, relaxed eyes | neutral expression, composed face, calm eyes | 13/16 | 要注意 |

## 個別提案
### 1. face_roundness_slider
目的: 成人キャラクターの年齢感や服装を固定したまま、顔の丸さと輪郭の柔らかさを連続的に調整する
positive: round face, soft cheeks, gentle facial contour
unconditional: angular face, sharp jawline, defined facial contour
推奨構図: portrait 4, close-up 2, upper body 2
固定したい条件: adult woman, fully clothed, nonsexual, solo, neutral expression, hair away from face, simple background
避ける語: cute, young, child, teen, schoolgirl, chibi, baby face
想定リスク: 年齢、目の大きさ、かわいさ、性別印象に吸着する
対策: adult face と neutral expression を全 prompt で固定し、目の大きさや笑顔語を差分に入れない
初回検証: 8 prompts / guidance_scale 2.0 / 896x1152 / action enhance
採用スコア: 14/16
判定: 採用

## 後回しにした案
- cute_slider: 年齢・目・顔・服装・画風が同時に動くため概念純度が低い

## 次に作る prompt set 用の概念仕様
concept_name: face_roundness_slider
category: face
positive_terms: round face, soft cheeks, gentle facial contour
unconditional_terms: angular face, sharp jawline, defined facial contour
common_conditions: masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, neutral expression, simple background
recommended_layout: portrait 4, close-up 2, upper body 2
avoid_terms: cute, young, child, teen, schoolgirl, chibi, baby face
```

---

## 8. 失敗症状からの修正提案

学習結果が悪いとき、まず prompt 設計を疑う。rank、steps、learning rate の調整はその後にする。

| 症状 | 推定原因 | 修正 |
| --- | --- | --- |
| slider が効かない | 差分語が弱い、抽象的、target に目的語が残っている | positive/unconditional を 2-4 個の強い対比語へ絞る |
| 別人になる | 顔立ち、髪、服、体型が差分に混ざる | 目的外属性を target に固定し、差分語を削る |
| 服装が変わる | 片側だけ服語が増えている | 服カテゴリは target、fit や装飾だけを差分にする |
| 背景が変わる | 背景語が非対称 | 背景は全フィールドで同一にする。背景 slider 以外では simple background 固定 |
| 年齢が動く | `cute`, `round`, `small`, `young` などが年齢に吸着 | `adult woman`, `adult face`, `fully clothed`, `nonsexual` を全フィールドへ入れる |
| 性的に寄る | 身体特徴語だけが強く、文脈固定が弱い | 安全な服装と nonsexual を全フィールドへ入れる。露出語を使わない |
| 逆方向が汚い | unconditional が低品質・否定・別カテゴリになっている | 反対側も同品質の自然属性で書く |
| 構図が動く | positive/unconditional で構図語が違う | 構図、視点、背景を全フィールドで同じにする |

---

## 9. 最終チェック

出力前に次を確認する。

- 候補名は `<concept>_slider` の形になっている。
- positive と unconditional はどちらも自然な属性で、否定・低品質ではない。
- target に入れるべき固定条件と、差分語が分離されている。
- 8 prompts で検証できる構図配分が書かれている。
- リスクと対策が具体的である。
- 広すぎる概念をそのまま採用していない。
- 身体・年齢・魅力系では `adult woman`, `fully clothed`, `nonsexual` を明示している。
