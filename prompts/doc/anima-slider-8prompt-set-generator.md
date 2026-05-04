# Anima Slider LoRA 8-Prompt Set Generator

この文書は、Anima/Cosmos RFlow 向け text-only Slider LoRA の prompt YAML を、任意の slider 概念から 8 prompts の標準セットとして生成するための Local LLM 用ガイドである。

目的は、自然な文章を書くことではなく、`positive` と `unconditional` の差分だけが目的概念になる YAML を出力することである。

想定する使い方:

```text
この文書を参考に、次の概念仕様から Comfyui-anima-slider-node 用の 8 prompt YAML を作成してください。

concept_name: <例: eye_openness_slider>
category: <expression | face | hair | body | clothing | pose | background | rendering>
positive_terms: <positive 側の英語タグ 2-4個>
unconditional_terms: <unconditional 側の英語タグ 2-4個>
common_conditions: <任意。共通で入れたい条件>
avoid_terms: <任意。使ってはいけない語>
```

入力が日本語のお題だけの場合は、まず概念を1つに狭め、安全な positive/unconditional terms を作る。曖昧な場合は、成人・非性的・服装ありの安全な人物生成として解釈する。

---

## 1. YAML の絶対ルール

出力する YAML は、次の条件を満たす。

1. YAML は list of mappings。つまり先頭は必ず `- target:` から始まる。
2. ちょうど 8 要素を出力する。
3. 各要素は、次の 9 keys をこの順番で持つ。
   - `target`
   - `positive`
   - `unconditional`
   - `neutral`
   - `guidance_scale`
   - `action`
   - `width`
   - `height`
   - `batch_size`
4. `neutral` は `target` と完全に同じ文字列にする。
5. `positive` は `target + ", " + positive_terms` の形にする。
6. `unconditional` は `target + ", " + unconditional_terms` の形にする。
7. `target` と `neutral` には、positive_terms / unconditional_terms / その類義語を入れない。
8. positive/unconditional の差分以外は、同じ prompt 内で完全にそろえる。
9. 反対側に `not ...`, `no ...`, `bad ...`, `low quality`, `worst quality`, `blurry`, `bad anatomy` を使わない。
10. 年齢・身体・魅力・服 fit 関連では、全フィールドに `adult woman, fully clothed, nonsexual` を含める。
11. `action` は通常 `enhance`。
12. `guidance_scale` は初回検証では `2.0`。
13. `width: 896`, `height: 1152`, `batch_size: 1` を標準にする。
14. YAML 内にコメントを入れない。

---

## 2. 出力フォーマット

通常は、短い設計メモと YAML だけを出力する。

````text
## 設計メモ
concept_name: <name>
category: <category>
positive_terms: <terms>
unconditional_terms: <terms>
固定方針: <1-2文>
注意: <1-2文>

## YAML
```yaml
- target: "..."
  positive: "..."
  unconditional: "..."
  neutral: "..."
  guidance_scale: 2.0
  action: enhance
  width: 896
  height: 1152
  batch_size: 1
...
```

## 自己チェック
- 8 prompts: OK
- neutral equals target: OK
- target excludes slider terms: OK
- positive/unconditional differ only by concept terms: OK
- unsafe age/sexual/low-quality terms: none
````

ユーザーが「YAMLだけ」と指定した場合は、設計メモと自己チェックを省略し、YAML code block だけを出力する。

---

## 3. 共通 base tokens

特に指定がない場合、人物 prompt は次の tokens から始める。

```text
masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo
```

表情・顔・髪・身体・服装 slider では、この安全 base を全フィールドに入れる。背景・画風 slider でも、人物を含める場合は同じ base を使う。

追加しやすい固定語:

```text
neutral expression, relaxed mouth, simple background, plain background, clean background, simple studio background
```

避ける語:

```text
child, teen, minor, schoolgirl, schoolboy, young girl, young boy, student, school uniform, nude, naked, erotic, sexy, low quality, bad anatomy, blurry, worst quality
```

---

## 4. Category 別の prompt 設計

### 4.1 expression

用途: 笑顔、感情価、怒り、恥ずかしさ、表情の強弱。

推奨構図: `portrait` 4、`close-up` 2、`upper body` 2。

固定条件:

```text
adult woman, fully clothed, nonsexual, solo, simple background
```

注意:

- 目の開きや口の開きに吸着しやすい。
- expression slider では髪型や服装を変えすぎない。
- 怒り、涙、泣き顔などを混ぜると別 slider になる。

### 4.2 face

用途: 目の開き、目の大きさ、顔の丸さ、輪郭、眉。

推奨構図: `portrait` 4、`close-up` 2、`upper body` 2。

固定条件:

```text
adult woman, fully clothed, nonsexual, solo, neutral expression, relaxed mouth, hair away from face, simple background
```

注意:

- 顔形状は年齢・かわいさ・性別印象へ吸着しやすい。
- `cute`, `young`, `baby face`, `chibi` を入れない。
- 輪郭系では髪が顔を隠さないようにする。

### 4.3 hair

用途: 髪の長さ、量、乱れ、カール感。

推奨構図: `portrait` 2、`upper body` 4、`waist up` 2。

固定条件:

```text
adult woman, fully clothed, nonsexual, solo, neutral expression, simple background
```

注意:

- hair length slider では target に `long hair`, `short hair`, `medium hair` を入れない。
- hair volume / messiness / curliness slider では、target に髪色と必要なら髪長を入れてよい。
- 髪色は nuisance として分散するが、positive/unconditional の差分にしない。

### 4.4 body

用途: 胸、ウエスト、肩幅、筋肉量、身長感。

推奨構図: `waist up` 2、`standing upper body` 2、`thigh-up` 2、`full body` 2。

固定条件:

```text
adult woman, fully clothed, nonsexual, solo, safe clothing, simple background
```

注意:

- 必ず安全な服装で視認性を確保する。
- 露出を増やさない。
- 体型全体語ではなく、目的部位周辺の語に絞る。
- `skinny`, `fat`, `obese`, `curvy`, `sexy` など広い語は避ける。

### 4.5 clothing

用途: フィット感、布の厚み、装飾量、フォーマル度。

推奨構図: `upper body` 2、`waist up` 2、`standing upper body` 2、`thigh-up` 2。

固定条件:

```text
adult woman, fully clothed, nonsexual, solo, simple background
```

注意:

- 服カテゴリは target に置く。例: `turtleneck sweater`, `button-up blouse`, `tailored jacket`, `long sleeve dress`。
- positive/unconditional には fit や texture などの修飾だけを入れる。
- clothing_fit_slider では、体型語を差分に入れない。

### 4.6 pose

用途: 視線、姿勢、頭の傾き、動きの強さ。

推奨構図: `upper body` 2、`waist up` 2、`full body` 4。

固定条件:

```text
adult woman, fully clothed, nonsexual, solo, simple background
```

注意:

- dynamic pose は構図や手足破綻に混ざりやすい。
- カメラ距離の slider はなるべく作らない。
- 構図を変えたい場合は通常 prompt や workflow 側で制御する。

### 4.7 background

用途: 背景情報量、自然/都市、屋内/屋外、奥行き。

推奨構図: `upper body` 8 または `waist up` 8。人物構図は固定する。

固定条件:

```text
adult woman, fully clothed, nonsexual, solo, neutral expression
```

注意:

- 背景 slider では、target に背景語を入れすぎない。
- 人物、服装、構図、視点、画風を固定し、背景語だけを差分にする。
- indoor/outdoor は光や場面も動くため、中程度のリスクがある。

### 4.8 rendering

用途: 線の太さ、彩度、明るさ、描き込み量、塗り。

推奨構図: `portrait` 2、`upper body` 2、`waist up` 2、`full body` 2。

固定条件:

```text
adult woman, fully clothed, nonsexual, solo, simple background
```

注意:

- 反対側を低品質にしない。
- `detailed rendering` / `simple rendering` は品質差に混ざりやすいので慎重にする。
- line weight や saturation のような狭い画風軸を優先する。

---

## 5. 8 prompt の slot table

LLM は category に応じて下表を調整して使う。目的概念と衝突する slot は変更する。

### 5.1 face / expression 用

| index | composition | view | hair | clothing | background |
| --- | --- | --- | --- | --- | --- |
| 0 | portrait | front view | black hair, medium hair, hair away from face | simple blouse | plain gray background |
| 1 | portrait | three-quarter view | brown hair, long hair, hair away from face | knit top | simple studio background |
| 2 | close-up | front view | blonde hair, short hair, hair away from face | collared shirt | plain background |
| 3 | close-up | three-quarter view | silver hair, medium hair, hair away from face | light sweater | clean background |
| 4 | upper body | front view | pink hair, long hair, hair away from face | cardigan over shirt | plain background |
| 5 | upper body | three-quarter view | red hair, short hair, hair away from face | casual jacket | neutral background |
| 6 | portrait | front view | black hair with blue streaks, long hair, hair away from face | dress shirt | simple studio background |
| 7 | upper body | three-quarter view | brown hair, medium hair, hair away from face | turtleneck sweater | soft studio background |

### 5.2 hair 用

| index | composition | view | hair base | clothing | background |
| --- | --- | --- | --- | --- | --- |
| 0 | portrait | front view | black hair | simple blouse | plain gray background |
| 1 | upper body | three-quarter view | brown hair | knit top | simple studio background |
| 2 | upper body | front view | blonde hair | collared shirt | plain background |
| 3 | waist up | three-quarter view | silver hair | light sweater | clean background |
| 4 | upper body | front view | pink hair | cardigan over shirt | plain background |
| 5 | portrait | three-quarter view | red hair | casual jacket | neutral background |
| 6 | waist up | front view | black hair with blue streaks | dress shirt | simple studio background |
| 7 | upper body | three-quarter view | brown hair | turtleneck sweater | soft studio background |

hair length slider の場合は、hair base に髪長を入れない。hair volume / curliness / messiness slider の場合は、hair base に `medium hair` などを足してよい。

### 5.3 body / clothing 用

| index | composition | view | hair | clothing | background |
| --- | --- | --- | --- | --- | --- |
| 0 | upper body | front view | black hair, medium hair | fitted turtleneck sweater | plain gray background |
| 1 | upper body | three-quarter view | brown hair, long hair | button-up blouse | simple studio background |
| 2 | waist up | front view | blonde hair, short hair | simple knit top | plain background |
| 3 | waist up | three-quarter view | silver hair, medium hair | tailored jacket over shirt | clean background |
| 4 | standing upper body | front view | pink hair, long hair | cardigan over blouse | plain background |
| 5 | standing upper body | three-quarter view | red hair, short hair | long sleeve ribbed top | neutral background |
| 6 | thigh-up | front view | black hair with blue streaks, long hair | belted dress | simple studio background |
| 7 | thigh-up | three-quarter view | brown hair, medium hair | long sleeve dress shirt outfit | soft studio background |

clothing_fit_slider の場合は、target の clothing から `fitted`, `loose`, `oversized`, `tailored fit` などの fit 語を消し、服カテゴリだけにする。

### 5.4 background 用

| index | composition | view | hair | clothing | subject background base |
| --- | --- | --- | --- | --- | --- |
| 0 | upper body | front view | black hair, medium hair | simple blouse | neutral background |
| 1 | upper body | three-quarter view | brown hair, long hair | knit top | neutral background |
| 2 | upper body | front view | blonde hair, short hair | collared shirt | neutral background |
| 3 | upper body | three-quarter view | silver hair, medium hair | light sweater | neutral background |
| 4 | upper body | front view | pink hair, long hair | cardigan over shirt | neutral background |
| 5 | upper body | three-quarter view | red hair, short hair | casual jacket | neutral background |
| 6 | upper body | front view | black hair with blue streaks, long hair | dress shirt | neutral background |
| 7 | upper body | three-quarter view | brown hair, medium hair | turtleneck sweater | neutral background |

背景 slider では、positive/unconditional の最後に背景差分語を足す。

### 5.5 rendering 用

| index | composition | view | hair | clothing | background |
| --- | --- | --- | --- | --- | --- |
| 0 | portrait | front view | black hair, medium hair | simple blouse | plain background |
| 1 | upper body | three-quarter view | brown hair, long hair | knit top | simple studio background |
| 2 | waist up | front view | blonde hair, short hair | collared shirt | plain background |
| 3 | full body | three-quarter view | silver hair, medium hair | long sleeve dress | clean background |
| 4 | portrait | front view | pink hair, long hair | cardigan over shirt | plain background |
| 5 | upper body | three-quarter view | red hair, short hair | casual jacket | neutral background |
| 6 | waist up | front view | black hair with blue streaks, long hair | dress shirt | simple studio background |
| 7 | full body | three-quarter view | brown hair, medium hair | turtleneck sweater and skirt | soft studio background |

---

## 6. 生成アルゴリズム

1. 入力から `concept_name`, `category`, `positive_terms`, `unconditional_terms` を決める。
2. positive_terms と unconditional_terms をそれぞれ 2-4 個に圧縮する。
3. avoid_terms と衝突する語を削る。
4. category に合う slot table を選ぶ。
5. 各 index で `target` を作る。
6. `target` には目的概念を入れない。
7. `positive = target + ", " + positive_terms` を作る。
8. `unconditional = target + ", " + unconditional_terms` を作る。
9. `neutral = target` を作る。
10. YAML 8要素を出力する。
11. 自己チェックで、差分以外が揃っているか確認する。

---

## 7. 概念別の追加ルール

### eye_openness_slider

- target に `wide eyes`, `half-closed eyes`, `sleepy`, `alert` を入れない。
- 共通条件に `neutral expression, relaxed mouth` を入れる。
- 目が隠れる髪型を避ける。

### face_roundness_slider

- target に `cute`, `young`, `baby face`, `round face`, `sharp jawline` を入れない。
- 共通条件に `adult face, neutral expression, hair away from face` を入れる。
- 口や目の大きさを差分にしない。

### hair_length_slider

- target に髪長を入れない。
- positive: `long hair, flowing hair`
- unconditional: `short hair, cropped hair`
- upper body / waist up を多めにする。

### clothing_fit_slider

- target に `fitted`, `tight`, `loose`, `oversized` を入れない。
- 服カテゴリだけを target に置く。
- body words を差分に入れない。

### bust_size_slider

- 全フィールドに `adult woman, fully clothed, nonsexual` を入れる。
- 安全な服装を target に入れる。
- 露出語、性的語、ポーズ語を使わない。

### background_detail_slider

- target では背景を `neutral background` 程度にする。
- positive/unconditional の差分は背景情報量だけにする。
- 人物・服・構図・表情は固定する。

### color_saturation_slider

- target に彩度語を入れない。
- positive: `vivid colors, high saturation, rich colors`
- unconditional: `muted colors, low saturation, subdued colors`
- `low quality` を使わない。

---

## 8. 悪い YAML の例

悪い例:

```yaml
- target: "masterpiece, 1girl, cute girl, portrait"
  positive: "masterpiece, 1girl, cute girl, portrait, big eyes, young, pink dress, beautiful background"
  unconditional: "low quality, not cute, bad face, plain background"
  neutral: "masterpiece, 1girl, cute girl, portrait"
  guidance_scale: 2.0
  action: enhance
  width: 896
  height: 1152
  batch_size: 1
```

問題:

- target に `cute` が入り、目的概念が抜けていない。
- positive だけ服装と背景が増えている。
- unconditional が低品質・否定になっている。
- 年齢方向に寄る `young` が入っている。
- 差分が eye / age / clothing / background / quality に分裂している。

---

## 9. 良い YAML の例: eye_openness_slider

```yaml
- target: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, portrait, front view, black hair, medium hair, hair away from face, simple blouse, neutral expression, relaxed mouth, plain gray background"
  positive: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, portrait, front view, black hair, medium hair, hair away from face, simple blouse, neutral expression, relaxed mouth, plain gray background, wide open eyes, alert eyes, visible eyelids"
  unconditional: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, portrait, front view, black hair, medium hair, hair away from face, simple blouse, neutral expression, relaxed mouth, plain gray background, half-closed eyes, relaxed eyelids, sleepy eyes"
  neutral: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, portrait, front view, black hair, medium hair, hair away from face, simple blouse, neutral expression, relaxed mouth, plain gray background"
  guidance_scale: 2.0
  action: enhance
  width: 896
  height: 1152
  batch_size: 1
- target: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, portrait, three-quarter view, brown hair, long hair, hair away from face, knit top, neutral expression, relaxed mouth, simple studio background"
  positive: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, portrait, three-quarter view, brown hair, long hair, hair away from face, knit top, neutral expression, relaxed mouth, simple studio background, wide open eyes, alert eyes, visible eyelids"
  unconditional: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, portrait, three-quarter view, brown hair, long hair, hair away from face, knit top, neutral expression, relaxed mouth, simple studio background, half-closed eyes, relaxed eyelids, sleepy eyes"
  neutral: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, portrait, three-quarter view, brown hair, long hair, hair away from face, knit top, neutral expression, relaxed mouth, simple studio background"
  guidance_scale: 2.0
  action: enhance
  width: 896
  height: 1152
  batch_size: 1
- target: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, close-up, front view, blonde hair, short hair, hair away from face, collared shirt, neutral expression, relaxed mouth, plain background"
  positive: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, close-up, front view, blonde hair, short hair, hair away from face, collared shirt, neutral expression, relaxed mouth, plain background, wide open eyes, alert eyes, visible eyelids"
  unconditional: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, close-up, front view, blonde hair, short hair, hair away from face, collared shirt, neutral expression, relaxed mouth, plain background, half-closed eyes, relaxed eyelids, sleepy eyes"
  neutral: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, close-up, front view, blonde hair, short hair, hair away from face, collared shirt, neutral expression, relaxed mouth, plain background"
  guidance_scale: 2.0
  action: enhance
  width: 896
  height: 1152
  batch_size: 1
- target: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, close-up, three-quarter view, silver hair, medium hair, hair away from face, light sweater, neutral expression, relaxed mouth, clean background"
  positive: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, close-up, three-quarter view, silver hair, medium hair, hair away from face, light sweater, neutral expression, relaxed mouth, clean background, wide open eyes, alert eyes, visible eyelids"
  unconditional: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, close-up, three-quarter view, silver hair, medium hair, hair away from face, light sweater, neutral expression, relaxed mouth, clean background, half-closed eyes, relaxed eyelids, sleepy eyes"
  neutral: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, close-up, three-quarter view, silver hair, medium hair, hair away from face, light sweater, neutral expression, relaxed mouth, clean background"
  guidance_scale: 2.0
  action: enhance
  width: 896
  height: 1152
  batch_size: 1
- target: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, upper body, front view, pink hair, long hair, hair away from face, cardigan over shirt, neutral expression, relaxed mouth, plain background"
  positive: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, upper body, front view, pink hair, long hair, hair away from face, cardigan over shirt, neutral expression, relaxed mouth, plain background, wide open eyes, alert eyes, visible eyelids"
  unconditional: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, upper body, front view, pink hair, long hair, hair away from face, cardigan over shirt, neutral expression, relaxed mouth, plain background, half-closed eyes, relaxed eyelids, sleepy eyes"
  neutral: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, upper body, front view, pink hair, long hair, hair away from face, cardigan over shirt, neutral expression, relaxed mouth, plain background"
  guidance_scale: 2.0
  action: enhance
  width: 896
  height: 1152
  batch_size: 1
- target: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, upper body, three-quarter view, red hair, short hair, hair away from face, casual jacket, neutral expression, relaxed mouth, neutral background"
  positive: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, upper body, three-quarter view, red hair, short hair, hair away from face, casual jacket, neutral expression, relaxed mouth, neutral background, wide open eyes, alert eyes, visible eyelids"
  unconditional: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, upper body, three-quarter view, red hair, short hair, hair away from face, casual jacket, neutral expression, relaxed mouth, neutral background, half-closed eyes, relaxed eyelids, sleepy eyes"
  neutral: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, upper body, three-quarter view, red hair, short hair, hair away from face, casual jacket, neutral expression, relaxed mouth, neutral background"
  guidance_scale: 2.0
  action: enhance
  width: 896
  height: 1152
  batch_size: 1
- target: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, portrait, front view, black hair with blue streaks, long hair, hair away from face, dress shirt, neutral expression, relaxed mouth, simple studio background"
  positive: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, portrait, front view, black hair with blue streaks, long hair, hair away from face, dress shirt, neutral expression, relaxed mouth, simple studio background, wide open eyes, alert eyes, visible eyelids"
  unconditional: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, portrait, front view, black hair with blue streaks, long hair, hair away from face, dress shirt, neutral expression, relaxed mouth, simple studio background, half-closed eyes, relaxed eyelids, sleepy eyes"
  neutral: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, portrait, front view, black hair with blue streaks, long hair, hair away from face, dress shirt, neutral expression, relaxed mouth, simple studio background"
  guidance_scale: 2.0
  action: enhance
  width: 896
  height: 1152
  batch_size: 1
- target: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, upper body, three-quarter view, brown hair, medium hair, hair away from face, turtleneck sweater, neutral expression, relaxed mouth, soft studio background"
  positive: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, upper body, three-quarter view, brown hair, medium hair, hair away from face, turtleneck sweater, neutral expression, relaxed mouth, soft studio background, wide open eyes, alert eyes, visible eyelids"
  unconditional: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, upper body, three-quarter view, brown hair, medium hair, hair away from face, turtleneck sweater, neutral expression, relaxed mouth, soft studio background, half-closed eyes, relaxed eyelids, sleepy eyes"
  neutral: "masterpiece, best quality, score_9, score_8, 1girl, adult woman, fully clothed, nonsexual, solo, upper body, three-quarter view, brown hair, medium hair, hair away from face, turtleneck sweater, neutral expression, relaxed mouth, soft studio background"
  guidance_scale: 2.0
  action: enhance
  width: 896
  height: 1152
  batch_size: 1
```

---

## 10. 最終チェックリスト

YAML 出力前に必ず確認する。

- 8 prompts ちょうどである。
- すべての prompt に `target`, `positive`, `unconditional`, `neutral`, `guidance_scale`, `action`, `width`, `height`, `batch_size` がある。
- `neutral` は `target` と完全一致している。
- `positive` と `unconditional` は、同じ target に差分語を足しただけである。
- `target` に目的概念の語がない。
- category に合う構図配分になっている。
- 髪色・服装・背景・視点が同一 prompt 内で非対称になっていない。
- 低品質語・否定語で反対側を作っていない。
- 身体・年齢・魅力系では `adult woman, fully clothed, nonsexual` が全フィールドに入っている。
- avoid_terms が使われていない。
