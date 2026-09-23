# 胸部サイズ・衣服フィットの単方向学習を検討する

更新日: 2026-09-22

**2026-09-23追記:** 胸部増大v2のLora8と衣服ゆったりv3のLora9について、提供画像17枚と学習レポートを評価し、胸部増大v4を追加した。[最新評価・v4の変更・次の検証](lora8-lora9-breast-v4-evaluation.md)を参照。以下の「未評価」は2026-09-22時点の記録であり、v4自体の学習・生成は未実施。

**作成済みの学習例:** 本書のC案に基づき、[胸部増加v2](../prompts-anima-breast_size_slider_v2.yaml)、[胸部減少v3](../prompts-anima-breast_size_slider_v3.yaml)、[衣服密着v2](../prompts-anima-clothing_fit_slider_v2.yaml)、[衣服ゆったりv3](../prompts-anima-clothing_fit_slider_v3.yaml)を追加した。各8項目で、元の目的4句・共通条件・YAML数値を維持。v2/v3は別方向の候補であり、元YAMLの置き換えやD案の句削減ではない。使用設定は[親README](../README.md)を参照。学習・生成評価はまだ行っていない。

対象は[breast_size](../prompts-anima-breast_size_slider.yaml)と[clothing_fit](../prompts-anima-clothing_fit_slider.yaml)。両方とも方向別の単方向学習を試す価値がある。特に、胸部サイズと衣服の密着度が同時に動く可能性を分離して調べたい。ただし、今回確認したのは各8項目の文章と設定であり、この2種類の学習結果・生成画像の評価ではない。以下の問題は仮説、文章は試験案であり、現行YAMLの正式な置き換えではない。

手順は[単方向学習・プロンプト改善の調査方法](slider-direction-investigation.md)と[Prompt variation guide](README.md)に従う。ageでは方向分離と文章調整を経て改善したが、複数条件を同時に変えており、単方向化だけの効果は確定していない。

## 1. 現行YAMLの構造

両ファイルとも全8項目で`target = neutral`。`positive`と`unconditional`は同じ項目の`target`全文へ、各ファイル共通の4句を追記した形である。4ロール間で服装・背景・視点を変える非対称はない。一方、`target`には胸部サイズ／服のゆとりの明示的な中間基準がない。

全項目の設定は`guidance_scale: 2.0`、`action: enhance`、`width: 896`、`height: 1152`、`batch_size: 1`。YAMLの`action`から両方向学習の実施歴はわからない。`direction_loss`は学習ノードと実行レポートで確認する。

### breast_size：差分句の全文

```text
positive:
larger covered bust silhouette, fuller bust volume under clothing, rounded covered chest shape, natural bust volume under clothing

unconditional:
smaller covered bust silhouette, flatter chest volume under clothing, minimal covered chest shape, low bust volume under clothing
```

共通条件は`masterpiece, best quality, score_9, score_8, safe, 1girl, adult woman, fully clothed, nonsexual, solo`。全項目が`upper body, waist-up portrait`、暗い茶髪、`opaque modest top`、暗色の衣服、`neutral colors, covered torso visible`である。背景・姿勢・表情は全て次の同文。

```text
clean bright neutral studio, pale wall, no dark background, same background, background unchanged, pose unchanged, looking at viewer, neutral expression
```

| index | 用途案 | 視点 | 髪型・配置 | 衣服の原文 |
| --- | --- | --- | --- | --- |
| 0 | train | front view | medium hair / hair away from face | simple dark knit top |
| 1 | train | three-quarter view | long hair / hair away from face | simple dark cardigan over top |
| 2 | train | front view | short hair / hair tucked behind ears | simple dark sweater |
| 3 | train | three-quarter view | medium hair / hair away from face | simple dark jacket over shirt |
| 4 | train | front view | long hair / hair away from face | simple dark cardigan and blouse |
| 5 | train | three-quarter view | short hair / side-parted hair | simple dark ribbed top |
| 6 | eval | front view | long hair / hair away from face | simple dark cardigan |
| 7 | eval | three-quarter view | medium hair / hair tucked behind ears | simple dark vest over shirt |

### clothing_fit：差分句の全文

```text
positive:
closely fitted clothing, tailored fit, smooth fabric contour, neat fitted silhouette

unconditional:
loose clothing fit, relaxed fit, extra fabric folds, boxy loose silhouette
```

共通条件は`anime illustration, adult woman, solo, fully clothed, nonsexual`、暗い茶髪、暗色の衣服。各項目の衣服句に続く共通部分は次のとおり。

```text
same garment type, neutral colors, body shape unchanged, clean bright neutral studio, pale wall, no dark background, same background, background unchanged, pose unchanged, outfit clearly visible, neutral expression
```

| index | 用途案 | 画角 / 視点 | 髪型 | 衣服の原文 |
| --- | --- | --- | --- | --- |
| 0 | train | upper body / front view | medium hair | simple dark blouse |
| 1 | train | waist up / three-quarter view | long hair | simple dark cardigan over top |
| 2 | train | thigh-up / front view | short hair | simple dark jacket and skirt |
| 3 | train | upper body / three-quarter view | medium hair | simple dark sweater |
| 4 | train | waist up / front view | long hair | simple dark long-sleeve top and skirt |
| 5 | train | thigh-up / three-quarter view | short hair | simple dark casual dress |
| 6 | eval | full body / front view | long hair | simple dark coat over dress |
| 7 | eval | waist up / three-quarter view | medium hair | simple dark vest over shirt |

`train=0–5 / eval=6–7`は今回固定する分割であり、YAML自体に用途指定はない。breast_sizeのevalは同画角・近い衣服群なので広い一般化検証にはならない。clothing_fitのindex 6は学習にない全身画角とコートを同時に含み、失敗時に原因を一意に絞れない。index 7と後述の追加生成文も併用する。

## 2. 調べたい問題と保持条件

| 対象 | 文章から立てられる仮説 | 画像で確かめること |
| --- | --- | --- |
| breast_size | 基準サイズ無指定の`target`に、豊かな胸部対ほぼ平坦な胸部の差を足している | 中程度からの連続変化になるか、低強度から急変するか |
| breast_size | 4句の反復に加え、`rounded`と`natural`は容量以外の形状・自然さも含む | 容量ではなく丸い陰影・光沢・描画の変更で効果を表現していないか |
| breast_size | knit / sweater / ribbedの生地や重ね着は、着衣越しの容量の読み取り方を変える | 胸部が変わるのか、服が密着・薄手化するだけなのか |
| clothing_fit | ゆとりの基準が無指定で、密着と箱型の対極を学習する | 標準フィットから目的側へ穏やかに調整できるか |
| clothing_fit | `tailored`は仕立て、`smooth`は表面・しわ、`boxy`は裁断形状まで動かし得る | サイズのゆとりの変更が、別デザイン・別生地への変化になっていないか |
| clothing_fit | `extra fabric folds`はゆとりだけでなく描き込み量にも対応し得る | 服全体・髪・顔へ細線や陰影が増えないか |

これらは現時点で確認された不具合ではない。`same garment type`、`body shape unchanged`、`background unchanged`等は言語上の誘導であり、画像を固定する機能ではない。

胸部サイズで許容するのは成人の着衣越しの胸部容量・輪郭と、それに必要な局所的な生地の張り・しわの変化。胸以外の肩幅・胴幅・腰幅・頭身・年齢・顔・姿勢・服の種類や丈は保持対象とする。「服のピクセルが完全一致」を合格条件にすると目的変化まで禁止してしまう。

衣服フィットで許容するのは服と身体の間のゆとり、布の張り、自然なしわ・垂れ方の変化。身体そのものの胸部容量・肩幅・胴幅・腰幅は保持対象とする。服が密着して元の体形が見えやすくなる現象と、身体の変形を区別する。服越しに判別不能なら断定せず「判定保留」とする。袖口や裾の幅の変化は目的に沿い得るが、袖丈・裾丈の大幅変化、露出、別の襟型・装飾・生地への変更は別軸で採点する。

## 3. 方向を分けたC案（未検証）

どちらも`target = unconditional = neutral = 共通文 + 基準句`、`positive = 共通文 + 目的句`とする。`action=enhance`、`direction_loss=enhance_only`で方向ごとに独立LoRAを学習する。反対方向を使わない場合はそのLoRAを作る必要はない。単方向LoRAの負強度を逆方向用の代用品にはしない。

C段階では共通文を現行各項目の`target`全文のまま使い、目的句も現行4句を保持する。最初から差分を短くする変更はDへ分ける。基準統一は`target`軌道と差分の両方を変更するため、B→Cを参照文だけの効果とは呼ばない。

### 胸部：中程度を基準に、増加／減少を独立化

index 0の共通文例（他のindexは元の髪・衣服・視点をそのまま使う）:

```text
masterpiece, best quality, score_9, score_8, safe, 1girl, adult woman, fully clothed, nonsexual, solo, upper body, waist-up portrait, front view, dark brown hair, medium hair, hair away from face, opaque modest top, simple dark knit top, neutral colors, covered torso visible, clean bright neutral studio, pale wall, no dark background, same background, background unchanged, pose unchanged, looking at viewer, neutral expression
```

| ロール／方向 | 共通文へ追加する試験句 |
| --- | --- |
| 基準（両方向共通） | `moderate bust volume under clothing` |
| 増加のpositive | `larger covered bust silhouette, fuller bust volume under clothing, rounded covered chest shape, natural bust volume under clothing` |
| 減少のpositive | `smaller covered bust silhouette, flatter chest volume under clothing, minimal covered chest shape, low bust volume under clothing` |

`moderate`が安定した中間状態を描く保証はない。学習前にLoRAなしの基準／目的文で確認する。D以降の簡潔な到達案は、同じ基準から`larger bust volume under clothing`／`smaller bust volume under clothing`への差だが、4句から一気に置換すると個々の語の影響がわからなくなるため、まず1句ずつ検討する。

### 衣服：標準フィットを基準に、密着／ゆったりを独立化

index 0の共通文例:

```text
anime illustration, adult woman, solo, fully clothed, nonsexual, upper body, front view, dark brown hair, medium hair, simple dark blouse, same garment type, neutral colors, body shape unchanged, clean bright neutral studio, pale wall, no dark background, same background, background unchanged, pose unchanged, outfit clearly visible, neutral expression
```

| ロール／方向 | 共通文へ追加する試験句 |
| --- | --- |
| 基準（両方向共通） | `regular clothing fit` |
| 密着のpositive | `closely fitted clothing, tailored fit, smooth fabric contour, neat fitted silhouette` |
| ゆったりのpositive | `loose clothing fit, relaxed fit, extra fabric folds, boxy loose silhouette` |

こちらも`regular clothing fit`が衣服ごとに同じ量のゆとりを意味するとは限らない。LoRAなし診断で確認する。最終的な簡潔案は`regular clothing fit`→`closely fitted clothing`／`loose clothing fit`。全ロールの成人・着衣・非性的文脈は維持し、露出や素材の透過を目的に含めない。

## 4. A～Dの実験を分離する

| 条件 | 文章 | 方式 | 意図的に変えるもの |
| --- | --- | --- | --- |
| A | 現行YAMLの固定コピー | bidirectional | 現行対極文の両方向対照 |
| B | Aと同じ全項目・全数値 | enhance_only | A→Bは方式のみ |
| C | 上記の基準統一、目的4句は維持 | enhance_only | B→Cは基準と対比の再設計 |
| D | Cから症状に対応する1句だけ変更 | enhance_only | C→Dは単一の文章仮説 |

まず既存positive方向（胸部増加、衣服密着）でA/B/Cを比較する。減少・ゆったりの需要があれば別Cを追加する。逆方向まで方式差を厳密に比較する場合、既存positive/unconditionalを入れ替えた同文ペアでA逆/B逆を作り、元Aの負枝も補助比較として残す。元Bは逆方向の対照にはならない。

Dの候補は症状を確認してから選ぶ。胸部増加なら`natural bust volume under clothing`の削除、丸み偏重があれば別実験で`rounded covered chest shape`の削除。減少で平坦化が急なら`minimal covered chest shape`の削除。衣服密着で表面が平滑化するなら`smooth fabric contour`の削除、仕立てが変わるなら別実験で`tailored fit`の削除。ゆったりで描き込みが増えるなら`extra fabric folds`、形が箱型に偏るなら別実験で`boxy loose silhouette`を削除する。複数の削除を1実験へまとめない。

生地語への吸着が疑われる場合は、別のDとして例えばbreast_size index 5の`simple dark ribbed top`を`simple dark top`へ全ロール同時に変更する。効果句の削除とは別runにする。顔・背景の共通句の大幅整理、多様性追加、学習率・係数変更も別実験へ分ける。

## 5. 正式Anima2.9B設定を使う際の固定条件

[正式ワークフローと設定一覧](../../workflows/README.md)は`enhance_only`、900 steps、共通lr `1.5e-5`、self-attention `8e-6`、cross-attention `2.5e-5`、MLP `1.5e-6`、rank/alpha `16/16`を採用している。比較ではこれを出発点として、Aだけ`direction_loss`を`bidirectional`へ変更する。ノードの学習対象や各overrideも合わせて固定し、共通lrだけの一致で同条件とはしない。

配布版は提供値を保持するため、seedの`control_after_generate`は`randomize`、出力prefixは`loras/anima_age_slider`である。比較用コピーでは`fixed`にし、概念・方向・run IDごとに出力名を分ける。`vary_seed=true`（学習内のstepごとのseed変化）とは別の制御なので混同しない。

```text
effective_eta = eta × YAML.guidance_scale × teacher_guidance_scale
             = 1.5 × 2.0 × 1.0 = 3.0
```

age正規版のYAML guidanceは1.0だったため実効係数1.5。この2種類の2.0を無断で1.0へ変えると、方式比較に教師差分の係数変更まで混ざる。A～Dでは2.0を保持し、必要なら別の係数探索を行う。ノルム参照も固定・記録する。

正式ワークフローは学習ノードに1024×1024を明示するため、YAMLの896×1152よりノード指定が優先される。実行レポートで解決後の解像度を確認する。YAML記載の縦長条件で実学習したと記録しない。

モデル名・SHA-256、YAML・LoRAのSHA-256、学習／評価seed、train/eval indices、ノルム参照、scheduler、時刻設定、推論設定を各runに保存する。同stepsでもAは二枝、Bは一枝なので計算時間は一致しない。初期LoRAのdown重みはグローバルRNGで初期化され、学習seedだけでは初期state一致を保証しない。初期stateを照合できなければ留保を記録し、採用候補は複数学習で反復する。詳細は[調査手順の実験条件](slider-direction-investigation.md#5-実験条件と成果物を固定する)を参照する。

## 6. 未学習生成文と採点

両スライダーを個別に適用し、最初は同時使用しない。次の2本は8項目の学習文とは異なる生成positive候補。胸部サイズ・フィットの目的句は入れず、LoRA適用差を観察する。基準状態が不安定な場合はそれ自体を記録し、基準句を追加した別セットとして比較する。

```text
P1: anime illustration, adult woman, solo, fully clothed, nonsexual, waist-up portrait, front view, short black hair, opaque navy long-sleeve crew-neck cotton shirt, arms relaxed at sides, neutral expression, light gray studio background

P2: anime illustration, adult woman, solo, fully clothed, nonsexual, thigh-up, three-quarter view, auburn hair tied back, opaque olive long-sleeve blouse with a high neckline, dark straight trousers, arms relaxed at sides, neutral expression, pale blue studio background
```

通常生成のnegativeは全条件で同一にする。各文×seed `1, 10000`で強度`0, +0.5, +1, +1.5, +2`を比較し、急変部分だけ細分化する。ファイル名ではなくPNG入力の実LoRA・実強度を監査し、強度0画像のRGB一致も確認する。Aの負強度は別表とし、単方向案の比較は各方向LoRAの正強度で行う。

| 評価軸 | breast_size | clothing_fit |
| --- | --- | --- |
| 目的効果 | 正面の胸部輪郭・三四分視の前方突出が、肩幅等に対して増減するか | 胴・腕まわりの服のゆとり、張り・しわが目的方向へ変わるか |
| 交絡 | 密着や濃い陰影だけで容量が増したように見えていないか | 胸部や胴自体の変形だけで服が細く／太く見えていないか |
| 衣服保持 | 種類・襟・丈・色・素材を維持し、必要な局所伸縮を許す | 種類・襟・丈・色・素材を維持し、目的に沿う幅・しわの変化を許す |
| 共通保持 | 成人性、顔、髪、胸以外の体格、表情、姿勢、背景、画角、描画 | 成人性、顔、髪、体格、表情、姿勢、背景、画角、描画 |

効果を0=なし、1=弱い、2=実用、3=過剰、各保持軸を0=維持～3=破綻で記録する。実用の暫定目安は「両視点で目的輪郭／ゆとりの変化を判読でき、身体・衣服の別概念の変更に頼っていない」こと。同じ強度と同じ効果量の両方で比較し、効果が弱くなっただけの案を改善に数えない。服越しの身体量は厳密計測できないため、同効果量は目視近似と明示する。

暫定採用条件は2文×2seed全てで効果2を得られる強度があり、目的外の体格・衣服・背景変化が各1以下であること。強度の単調性と使用可能範囲も記録する。良かった案だけ全身・別素材・重ね着・別背景、追加seedへ広げる。学習MSEは教師への適合であり、異なる教師やAの二枝平均とBの一枝の値だけで画質順位を決めない。

## 7. 最小の着手順

1. 元YAML・正式ワークフローを固定コピーし、モデルと係数・解決後サイズを記録する。両概念の基準／目的文をLoRAなしで通常生成し、意味が描き分けられるか確認する。
2. まず胸部増加のA/B/Cをtrain 0–5、eval 6–7で比較する。元々同一画角なので、画角差より基準・方向設計へ着目しやすい。短縮探索なら全条件を同stepsにそろえ、採用前に正式900 stepsで揃えて確認する。
3. 次に衣服密着のA/B/Cを同じ手順で比較する。全身evalの悪化だけで不採用にせず、衣服・画角の影響を追加画像で分ける。
4. 各Cに実際の症状が出た場合だけ、該当するDを1件選ぶ。減少／ゆったり方向は必要に応じて独立追加する。
5. 同効果量の保持と一般化が改善した案を再学習seedでも確認してから正規化する。改善しない場合は現行を維持し、方式分離だけでは改善しなかった結果も残す。

本検討では新規学習・生成・正規YAML変更を行っていない。未確定なのは両概念の実際の副作用、提案基準句の安定性、単方向化の利点であり、上記の対照画像が次の判断材料となる。
