# Age slider variation examples

Age slider は、顔・肌・体の成熟感だけを動かし、人物の同一性、構図、服装、背景、品質をできるだけ固定する。安全側の基本設計は `older adult` と `younger adult` の対比にし、未成年を示す語は使わない。

## 推奨する差分語

| 方向 | 使いやすい語 | 注意 |
| --- | --- | --- |
| older 側 | `older adult face`, `fine facial lines`, `subtle nasolabial folds`, `mature facial structure` | `elderly` は効果が強いので、破綻する場合は後段で使う |
| younger adult 側 | `younger adult face`, `smooth skin`, `soft facial features`, `early adult appearance` | `child`, `teen`, `young girl`, `young boy` は使わない |
| neutral 側 | `adult woman`, `adult man`, `adult person`, `mature body proportions` | neutral に `young` / `old` を混ぜない |

最初は older 側 3-4 語、younger adult 側 3-4 語で十分。語を増やすより、全 prompt で同じ対比を繰り返す方が安定しやすい。

## バリエーション軸の例

| 軸 | 例 | 入れる理由 | 入れすぎた時の問題 |
| --- | --- | --- | --- |
| 被写体 | `adult woman`, `adult man`, `adult person` | 性別表現への過学習を避ける | 顔立ち差が大きすぎると年齢差分がぼける |
| 構図 | `portrait`, `close-up face`, `upper body`, `bust portrait` | 顔寄りと上半身で効きを確認する | full body が多いと顔の年齢変化が弱くなる |
| 視点 | `front view`, `three-quarter view` | 正面専用 slider 化を避ける | 横顔は評価が難しいので初期 set では少なめ |
| 髪色 | `black hair`, `brown hair`, `blonde hair`, `silver hair`, `red hair` | 髪色と年齢の結び付きを減らす | silver hair は older 側に吸着しやすいので少量 |
| 髪型 | `short hair`, `medium hair`, `long hair` | 髪型固定への過学習を避ける | 髪型変更を positive/unconditional 差分に混ぜない |
| 装飾 | `no jewelry`, `simple earrings`, `simple glasses` | 顔周辺の軽い条件差に耐性を持たせる | glasses は年齢属性に見えやすいので少量 |
| 背景 | `plain background`, `simple gray background`, `simple studio background` | 背景への吸着を抑える | 複雑な背景は年齢評価を邪魔する |
| 解像度 | `1024x1024`, `896x1152` | 顔中心と縦長構図の両方を見る | 1MP を大きく超える設定は避ける |

## 小規模 set の構成例

3-4 prompts で効きを見る場合は、年齢差分語を固定し、構図だけを軽く変える。

| index | subject | shot | view | hair | background |
| --- | --- | --- | --- | --- | --- |
| 0 | adult woman | portrait | front view | black medium hair | simple gray background |
| 1 | adult woman | upper body | three-quarter view | brown long hair | plain background |
| 2 | adult man | portrait | front view | black short hair | simple gray background |
| 3 | adult man | upper body | three-quarter view | brown medium hair | plain background |

この段階で見るのは「年齢方向が動くか」だけ。髪色や服装の多様性はまだ増やさない。

## 標準 set の構成例

8-12 prompts では、顔中心と上半身を半分ずつにする。

| 配分 | 例 |
| --- | --- |
| subject | adult woman 4-6、adult man 4-6 |
| shot | portrait 4、close-up face 2、upper body 4、bust portrait 2 |
| view | front view と three-quarter view を半々 |
| hair color | black, brown, blonde, red を中心に、silver は少量 |
| background | simple gray / plain / simple studio をローテーション |

構図ごとに subject と髪色が偏らないようにする。例えば `adult man` が portrait だけ、`adult woman` が upper body だけ、という分け方は避ける。

## 本番寄り set の構成例

24 prompts 前後では、同じ差分語を使いながら条件を均等に広げる。

| block | prompts | 内容 |
| --- | --- | --- |
| A | 0-5 | adult woman, portrait / close-up, front / three-quarter, 主要髪色 |
| B | 6-11 | adult woman, upper body / bust portrait, front / three-quarter, 主要髪色 |
| C | 12-17 | adult man, portrait / close-up, front / three-quarter, 主要髪色 |
| D | 18-23 | adult man, upper body / bust portrait, front / three-quarter, 主要髪色 |

評価用 indices は学習用と同じものに加え、未学習の髪色や服装を 2-4 個だけ別途用意すると、汎化と過学習を切り分けやすい。

## 症状別の Age slider 修正

| 症状 | 推定原因 | 修正例 |
| --- | --- | --- |
| 老け方向で髪が白くなるだけ | `silver hair` や `gray hair` が多すぎる | 髪色を neutral 条件に固定し、older 側から髪色語を抜く |
| 若返り方向で未成年っぽくなる | younger 側の語が強すぎる、または unsafe age terms が混ざる | `younger adult face`, `early adult appearance` に置き換える |
| 体格まで大きく変わる | older/younger 側に体型語が混ざっている | `mature body proportions` を全フィールドに入れ、体型語を差分から抜く |
| 服装が変わる | older 側だけ落ち着いた服、younger 側だけカジュアル服になっている | 服装は各 prompt 内で同一にする |
| 顔のしわだけ強すぎる | `visible wrinkles`, `deep wrinkles`, `elderly` が強すぎる | `fine facial lines`, `subtle nasolabial folds` へ弱める |
| close-up では効くが upper body で効かない | 顔領域が小さい構図が多い、または差分語が顔だけ | upper body の数を増やし、`mature facial structure` など顔全体の語を使う |

## テンプレート

```yaml
- target: "masterpiece, best quality, score_7, safe, 1girl, adult woman, solo, portrait, front view, black hair, medium hair, no jewelry, simple gray background"
  positive: "masterpiece, best quality, score_7, safe, 1girl, adult woman, solo, portrait, front view, black hair, medium hair, no jewelry, simple gray background, older adult face, subtle nasolabial folds, fine facial lines, mature facial structure"
  unconditional: "masterpiece, best quality, score_7, safe, 1girl, adult woman, solo, portrait, front view, black hair, medium hair, no jewelry, simple gray background, younger adult face, smooth skin, soft facial features, early adult appearance"
  neutral: "masterpiece, best quality, score_7, safe, 1girl, adult woman, solo, portrait, front view, black hair, medium hair, no jewelry, simple gray background"
  guidance_scale: 2.0
  action: enhance
  width: 1024
  height: 1024
  batch_size: 1
```

このテンプレートから変更する時は、まず `target` / `positive` / `unconditional` / `neutral` の共通部分を同時に変更し、最後に age 差分語だけを確認する。
