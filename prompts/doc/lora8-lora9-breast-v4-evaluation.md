# Lora8 / Lora9の評価と胸部増大v4

**2026-09-24追記:** [Lora10の評価と採用記録](lora10-breast-evaluation.md)により、胸部増大v4を正式採用し、初版・v2・縮小用v3はarchiveへ移動した。以下の候補・保持という記述は2026-09-23時点の履歴。

評価日: 2026-09-23。対象はユーザー提供のLoRA 2本、学習レポート2本、生成画像17枚。**v4は新しく学習する文章候補であり、副作用の改善を実証したLoRAではない。** 既存v2（胸部増大）、v3（胸部減少）、衣服用YAMLは保持する。

## 1. 資料と実行条件の監査

| 資料 | 学習レポートが指すYAML | 提供画像の実強度 |
| --- | --- | --- |
| `生成Lora/Lora8` | `prompts-anima-breast_size_slider_v2.yaml` | 0～4、0.5刻み、9枚 |
| `生成Lora/Lora9` | `prompts-anima-clothing_fit_slider_v3.yaml` | 0、0.5、1、1.5、2、2.5、3、4、8枚 |

Lora9は胸部スライダーの別版ではなく、衣服をゆったりさせる別概念。両者の優劣から胸部プロンプトの原因を直接特定することはできない。**Lora9の`3.4.png`はPNGの実行入力では4.0**。ファイル名を3.4や3.5として評価しない。

PNGの`prompt`を調べ、LoRA強度のリンク先`easy float`まで解決した。各系列内では強度以外の実行グラフが同一。両系列の強度0の画像はRGBのSHA-256が完全一致する。

推論は`oneObsession_anima29BV1.safetensors`、`qwen_3_06b_base.safetensors`、`qwen_image_vae.safetensors`、seed 1、840×1280、30 steps、CFG 4、`er_sde / beta`、denoise 1。LoRA名は提供ファイル名と一致し、model/clip両強度に同じ値が渡っている。保存されたLoRAは拡散モデル用で、clip強度の指定からテキストエンコーダーも学習されたとは判断しない。

生成positive（全画像共通）:

```text
anime illustration, solo, adult woman, full body, head-to-toe visible, feet visible, centered composition, standing, front view, arms relaxed at sides, looking at viewer, neutral expression, shoulder-length brown hair, plain gray long-sleeved shirt, straight navy trousers, flat shoes, plain light gray background, soft even lighting,
```

生成negative（全画像共通）:

```text
worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia
```

両学習レポートで共通する条件は、Anima 2.9B / 40 blocks、900 steps、rank/alpha 16/16、1024×1024、AdamW、共通lr `1.5e-5`、weight decay 0.01、self-attention `8e-6`、cross-attention `2.5e-5`、MLP `1.5e-6`。`enhance_only`、`teacher_norm_reference=target`、eta 1.5、YAML guidance 2、teacher guidance 1で実効係数3。学習軌道は20 steps / simple、時刻抽出shift 3、loss weighting none、train 0～5 / eval 6～7。学習stepの記録seedは両方583650396415791～583650396416690。ただし初期LoRA乱数の一致までは示さない。

| レポート | 初期eval MSE | 最終eval MSE |
| --- | --- | --- |
| Lora8 | 0.0009605091 | 0.0004512075 |
| Lora9 | 0.0010426778 | 0.0005057325 |

双方とも学習に使っていない評価項目の教師予測へのMSEは低下。目的の違う教師間で、この値を画質や副作用の順位に使わない。LoRAはそれぞれ1,200 tensors / 32,768,400 elementsを読み取り、NaN/Infなし。破損を示す非有限値は見つからないが、数値の有限性は意味的な品質の保証ではない。

完全なSHA-256、画像別実強度、推論グラフ、学習設定はローカルの[metadata_audit.json](../../生成Lora/lora8_lora9_evaluation_20260923/metadata_audit.json)へ保存した。`生成Lora/`は未追跡資料で、cloneには含まれない。学習レポートには学習元checkpoint名・YAML本文・当時のYAML hashがなく、PNGにもLoRA hashはない。したがって、現行YAMLが当時と同一だったことや、同名LoRAのバイナリ一致は未証明。以下の文章分析は、レポートのパスに対応する現行YAMLに基づく。

## 2. 画像から確認できたこと

### Lora8: 胸部増大

| 実強度 | 目視所見 |
| --- | --- |
| 0～0.5 | 胸の輪郭変化は小さく、主に胸下の陰影が変化。体格保持は比較的良いが、目的効果も弱い |
| 1～1.5 | 胸部の増大が明確。同時に腰・上腿の外形が横へ広がり、袖のしわや髪の毛先も変わる |
| 2～2.5 | 胸部の強い増大と下半身の幅増加が続く。顔の目の開き・輪郭にも変化 |
| 3～4 | 腰・上腿の広がりが目立ち、腕と袖の輪郭も変化。胸部形状が大きく丸まり、顔の印象も変わる |

ユーザーの「胸以外の腰・太ももも太くなる」という観察と一致する。ただし服越しの外形であり、実際の身体体積を測定したものではない。胸の張りだけでは説明できない下半身の輪郭変化は確認できる。0.5は保持寄りだが、1で得られる効果を弱めているだけなので、これだけで問題解消とは扱わない。この1文・1seedでは、胸に十分効き、下半身が確実に保持される範囲は確認できていない。

### Lora9: 衣服のゆとり

| 実強度 | 目視所見 |
| --- | --- |
| 0.5 | 上着と袖にゆとりが出る。顔・髪・姿勢の変化は比較的小さい |
| 1～1.5 | 上下の服が明確にゆったりする。前髪が額にかかり、袖口・ズボンのウエスト仕様も変わる |
| 2 | 長い裾、落ちた肩線、前髪・目つきの変化が目立つ |
| 2.5 | 手をポケットへ入れる姿勢に変化 |
| 3～4 | 腕を左右へ広げ、袖と裾が大幅に拡張。靴の意匠も変わる |

服の幅や自然なしわの増加は目的どおり。一方、前髪・表情・腕の姿勢・靴の変更は目的外であり、「髪型だけが変わる」とは限定できない。ゆったりした服の下の体形は判別困難なので、体形が完全に保持されたとは断定しない。今回の画像では0.5～1.5を次の実用評価範囲とし、高強度の過大な変化と分けて扱う。衣服v3を今回変更する根拠は胸部v4とは分ける。

## 3. 原因として考えられること

1. **目的句に容量以外の意味が混ざる。** 胸部v2のpositiveは`larger covered bust silhouette, fuller bust volume under clothing, rounded covered chest shape, natural bust volume under clothing`。基準は`moderate bust volume under clothing`の1句。容量に加えて、輪郭・丸さ・自然さの違いまで教師差分へ入る可能性がある。どの語が腰や太ももを変えたかは、今回の画像だけでは特定できない。
2. **上半身の学習・評価だけでは下半身の保持を直接確認できない。** 胸部v2の全8項目は`upper body, waist-up portrait`。全身推論への転用で相関した体格変化が出る仮説は妥当だが、全身化だけで治る証拠はない。
3. **同一の共通文は、身体を固定する制約ではない。** この実装は`positive_prediction - unconditional_prediction`を用いた教師と全体MSEで学習する。`unconditional`は通常生成のnegative欄ではなく比較基準。`neutral`も腰・髪を固定する損失ではない。胸だけのマスクや部位別保持損失はない。`same hairstyle`等の追加だけでは不変性を保証しない。
4. **高強度で目的外の方向も強く現れ得る。** 今回は実効教師係数3.0、推論強度最大4。ただし係数だけを下げて目的効果も同時に弱める方法は、属性分離の改善と区別する必要がある。

関連する一次資料として、[Concept Sliders公式実装](https://github.com/rohitgandikota/sliders#textual-concept-sliders)は、年齢と性別などの結び付きを抑えるために属性条件を使う例を示している。保持したい属性を両側で条件付けするという考え方は次案の参考になるが、同実装の`--attributes`がこのノードにあるわけではなく、Animaでの効果を保証する資料でもない。

## 4. 採用するv4と見送る案

今回の[v4](../prompts-anima-breast_size_slider_v4.yaml)では**positiveの目的句だけ**を次へ置き換える。

```text
target = unconditional = neutral:
  （v2と同じ共通文）, moderate bust volume under clothing

positive:
  （v2と同じ共通文）, larger bust volume under clothing
```

両側の語彙・語順を揃え、差を`moderate`対`larger`へ絞る。8項目すべての基準3ロール、髪型・服装・画角・背景・品質句・数値はv2と同一。胸の減少方向は既存v3を引き続き使う。v4は増大用の後継候補。

| 案 | 判断と理由 |
| --- | --- |
| 目的4句を容量1句へ簡潔化 | **v4に採用**。比較可能性を保ち、目的差分の余分な意味を削る最初の実験 |
| 全身／太ももまでの画角と下半身体格条件を導入 | 次案。腰・脚を観察できる文脈を増やすが、画角・衣服・軌道まで変わるため今回と分離 |
| guidance 2→1、学習率・rankを変更 | 別実験。効果低下と副作用低下が同時に起きる可能性があり、文章修正と混ぜない |

これは従来検討書の「1句ずつ削る」厳密な切り分けより大きい、**目的句全体の簡潔化を試す一括候補**。改善しても`rounded`単独などの因果効果とは呼ばない。必要なら4句→3句→2句→1句の中間条件を別runで比較する。

全身化する次案では、共通条件として具体的な腰・ヒップ・上腿の体格と髪型を両側に等しく置き、胸部の大小と体格条件を交差させる。常に細身だけを指定すると「胸を増やすと脚が細くなる」という逆の副作用も評価が必要。`body shape unchanged`の一括指定は目的の胸部変化まで含むため、保持する部位を分ける。

衣服側を将来変更するなら、基準`regular clothing fit`に対して`loose clothing fit`だけへ簡潔化する案と、髪・袖丈・裾丈・姿勢を具体的に両側へ指定する案を別々に試す。現時点ではユーザー評価も良く、まず既存v3の低～中強度を維持する。

## 5. v4の学習と採用判定

- `prompt_yaml=prompts-anima-breast_size_slider_v4.yaml`。`custom_prompt_yaml_path`が残っているとそちらが優先されるので空欄かv4のパスにする。
- `direction_loss=enhance_only`、train `0,1,2,3,4,5`、eval `6,7`、`teacher_norm_reference=target`、eta 1.5、teacher guidance 1、YAML guidance 2を保持。YAMLコメント自体はノード設定を変更しない。
- まずLora8と同じ900 steps、rank/alpha 16/16、学習率等を使い、**ノードで1024×1024を明示**する。YAMLはv2との比較用に896×1152を保持しているため、ノード0/0では学習サイズが変わる。
- 出力例は`loras/anima_breast_increase_v4`。既存LoRAの上書きや負強度による胸部縮小の代用はしない。
- 学習元checkpointとSHA-256を記録し、可能ならv2も同環境で再学習する。過去の学習元が不明のまま、Lora8との差をプロンプトだけの因果効果とは断定しない。

学習前に、LoRAなしで基準／目的文を同じseedから生成し、胸部の差と目的外の体格変化を見る。目的文自体が下半身を広げるなら、学習だけで解決するとは期待しない。

学習後は上記の全身文をそのまま使い、同じ推論設定でseed `1, 10000`、強度`0, 0.5, 1, 1.5, 2`を比較する。1付近が急なら0.75、1.25を追加。強度3～4は耐性確認として別扱いにする。さらに未学習の三四分視全身文も確認する:

```text
anime illustration, adult woman, solo, fully clothed, nonsexual, full body, head-to-toe visible, feet visible, standing, three-quarter view, arms relaxed at sides, short black hair with a side part, opaque olive long-sleeve crew-neck shirt, straight charcoal trousers, flat shoes, neutral expression, pale blue studio background, soft even lighting
```

評価は胸部容量、腰・ヒップ・上腿の外形、肩・腕、髪・顔、衣服仕様、姿勢、背景を分ける。正面だけでなく三四分視の胸部突出も見て、陰影や服の密着だけの変化と区別する。**同じ強度と、胸が同程度に増えた強度の両方で比較**する。効果が弱まっただけなら改善判定にしない。

暫定採用条件は、2文×2seedで実用的な胸部増大が得られ、同程度の効果のv2より腰・上腿の変化が小さく、髪・顔・服・姿勢に新たな明確な悪化がないこと。目視による暫定判定であり、身体の実寸測定ではない。満たさなければv4を改善済みとして正規化せず、上記の全身・体格条件を分離した次案へ進む。

検証結果: `python -m pytest tests/test_prompt_util.py -q`は29件成功。v4全8項目のローダー読み込み・標準バリデーション、v2から変更されたフィールドがpositiveのみであること、基準3ロールの一致、既存v2/v3の維持、文書内ローカルリンクと差分の空白検査を確認した。実装コードは変更していない。新規学習・画像生成は実施していない。
