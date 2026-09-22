# 他スライダーの単方向学習・プロンプト改善を調査する方法

更新日: 2026-09-22

目的は、age sliderで採用した「目的方向ごとに学習を分け、基準文と目的文の差分を調整する」方法が、他のスライダーにも有効かを調べること。全YAMLを一括変換せず、候補を1種類ずつ比較する。本書はローカル実装とageの評価履歴に基づく実験手順であり、他スライダーでの改善実績を示すものではない。

## 1. 出発点となるageの経過

| 段階 | 変更・観察 |
| --- | --- |
| v4 | 幼齢と老齢を対比する文章で`bidirectional`を実行。幼齢方向は動くが、老齢方向では自然な加齢より痩身・硬い描画・背景変化が目立った |
| v5 / v6 | 成人を共通基準に、老齢・幼齢を別YAMLと`enhance_only`へ分離。実効係数も6.25から1.5へ変更し、提供画像では改善 |
| v7 / v8 | 老齢側のしわ・表情、幼齢側の光沢・衣服を調整。老齢効果の弱まり、幼齢側の白縁や描画簡略化が残った |
| v9 / v10 | v9は老齢側に`facial wrinkles`を追加。v10は幼齢比率を維持し、全ロールから`flat color fills, simple cel shading`を削除 |
| 正規採用 | v9を[aging](../prompts-anima-aging_slider_fullbody.yaml)、v10を[deaging](../prompts-anima-deaging_slider_fullbody.yaml)として採用。v1～v8は[archive](../archive/README.md)で履歴を保持 |

最新の正しい生成画像では、v9は+1で明確な老齢化が得られたが、過密なしわ・下がった口角・衣服変更が残る。v10は+1～+1.5が保持重視の候補、+2では強い幼齢比率と白縁の軽減が見られたが、+2.5で白縁が再発した。正規採用は全副作用の解消を意味しない。

幼齢化の大きい頭・短い手足・低頭身は今回の目的であり、副作用として消す対象ではない。老齢側は直立を望み、猫背は目的に含めなかった。このように「必要な変化」を実験前に定義する。

**単方向化だけの因果効果は未証明。** v4→v5/v6は文章・方式・実効係数が同時に変わった。実学習解像度は全て1024×1024であり、初期提案にあった解像度変更と実績を混同しない。後段には推論モデルの変更があり、学習元モデル名・ハッシュも未記録だった。最新v9/v10の確認範囲は共通全身文・seed 1で、広い一般化は未確認。

## 2. 実装上の意味をそろえる

根拠は[training.py](../../anima_slider_node/training.py)の`flow_loss_for_record`、`teacher_from_parts`と、[slider_loss.py](../../anima_slider_node/slider_loss.py)の`flow_slider_teacher`。以下の`prediction`は同じlatent・時刻でのモデル予測であり、通常生成画像ではない。

```text
direction = prediction(positive) - prediction(unconditional)
effective_eta = eta × YAML.guidance_scale × teacher_guidance_scale
teacher = prediction(target) ± effective_eta × direction
最後に teacher_norm_reference の指定に従い全体ノルムを合わせる
```

- `bidirectional`はLoRA +1と−1の二枝を学習し、両損失を等重みで平均する。
- `enhance_only`はLoRA +1の一枝を学習する。本手順では`action: enhance`と組み合わせ、目的を`positive`に置く。
- `direction_loss`はノード設定であり、YAML名や`action`だけでは実際の学習方式がわからない。学習レポートで確かめる。
- `erase_only`という`direction_loss`は実装されていない。逆方向を調べる場合も別の目的文・別LoRAで`enhance_only`を実行する。単方向LoRAの負強度を代用品にしない。
- YAMLの`positive`は差分の目的側、`unconditional`は差し引く参照文。どちらも、推論ワークフローの生成positive / negative入力とは別の役割を持つ。
- `neutral`は選択時のノルム参照に使われる。背景や顔の保持を直接保証する損失ではない。`same background`などの語も画像固定機能ではない。
- +2以上は学習時の+1を越える適用であり、強度を上げれば品質も上がるとは限らない。

## 3. 最初の候補を選ぶ

以下は現行YAMLの文章から立てた仮説。既存の全スライダーが両方向学習済み、または失敗済みとはみなさない。手元に実学習レポートと症状画像がある候補を優先する。

| 優先度 | 候補 | 内容上の着眼点・比較する方向 |
| --- | --- | --- |
| 高 | [mature_face](../prompts-anima-mature_face_slider.yaml) | 成人の成熟顔対`young girl`・丸い頬。年齢と輪郭が同時に動く可能性。成人基準→成熟顔で年齢・輪郭の目的を先に区別 |
| 高 | [emotion_valence](../prompts-anima-emotion_valence_slider.yaml) | 明るい表情対憂鬱な表情。目・眉・口も変化する。中立→喜び、中立→悲しみを独立比較 |
| 高 | [chibi_style](../prompts-anima-chibi_style_slider.yaml) | 頭身だけでなく顔の簡略化も対比に含む。通常頭身→低頭身で、簡略化を目的に含めるか先に決める |
| 中 | [smile_intensity](../prompts-anima-smile_intensity_slider.yaml) | 笑顔対閉じた口。口の開閉・歯の露出・目の変化を、笑顔の強さと分けて採点 |
| 中 | [line_weight](../prompts-anima-line_weight_slider.yaml) | 太い線対細い線。標準線幅→太線／細線を分け、色・陰影・細部の保持を確認 |
| 対照候補 | [rain_intensity](../prompts-anima-rain_intensity_slider.yaml) | 強雨対弱雨。雨量と暗さ・濡れ表現を区別。方向分離が不要な概念も比較対象として残す |

候補ごとに「目的」「動いてよいもの」「保持したいもの」「現在の困り方」を各1文で書く。逆側の目的がない場合、逆方向LoRAの作成は不要。

## 4. A～Dで変更要因を分ける

| 条件 | YAML | direction_loss | 確かめること |
| --- | --- | --- | --- |
| A | 既存YAMLの固定コピー | bidirectional | 両方向の基準 |
| B | Aと本文・設定値が完全同一 | enhance_only | 意図的な変更因子を学習方式だけにした比較（初期化等の一致は別途確認） |
| C | 意味のある基準文へ方向別に設計 | enhance_only | B→Cで基準・対比を再設計した差 |
| D | Cから1つの語句・1つの仮説だけを変更 | enhance_only | C→Dで文章調整の差 |

A/Bは`action: enhance`の既存セットで開始する。別actionのセットは先に意図を確認し、action変更も行うなら別の対照実験として扱う。AとBを別のstepsや係数で走らせない。二枝対一枝のため計算量は異なり、同stepsは同計算時間ではない。学習時間も記録する。

既存の実行が`enhance_only`なら、その実績はB側の参考資料となる。Aは新規比較として実行し、既存実績を両方向の証拠に読み替えない。実行条件や重みの対応が欠ける過去資料は参考に留め、A/Bを同条件で作り直す。

Cは各項目で`target = unconditional = neutral`を同文にし、`positive`だけ目的属性を置き換える。基準は空文や単なる属性削除ではなく、成人、普通の頭身、中立の表情、標準線幅など、意味のある出発点にする。人物・画角・背景・服装などの共通句はそろえる。

```text
喜び用: target / unconditional / neutral = 共通条件 + calm neutral facial expression
        positive = 共通条件 + bright happy expression
悲しみ用: target / unconditional / neutral = 同じ中立基準
          positive = 共通条件 + melancholic facial expression
両方とも action=enhance、direction_loss=enhance_only、別々に学習
```

Cは通常、target軌道も差分も変更するため、「unconditionalだけの効果」とは呼ばない。基準統一の各要素まで分離したい場合は追加条件を設ける。Dでは年齢句、描画句、衣服句などを一度に変えず、1句の追加・削除から始める。共通の描画句を変える実験では全ロールへ同じ編集を適用し、その範囲を記録する。

## 5. 実験条件と成果物を固定する

各条件は新しいrun IDと出力先を持たせ、YAML、学習ワークフロー、学習JSON、LoRA、生成ワークフロー、PNGを上書きせず保存する。設定UIの見た目ではなく、実行レポート・PNGの`prompt`入力を監査する。

| 区分 | 固定・記録するもの |
| --- | --- |
| 同一性 | repoのcommitと未コミット差分、YAML本文とSHA-256、LoRAファイル名とSHA-256、run IDの対応 |
| モデル | 学習元・推論元それぞれのチェックポイント名とSHA-256、CLIP・VAE、ComfyUI／ノード版。推論モデル名から学習元を推定しない |
| 学習 | train/eval indices、学習seed・eval_seed、steps、optimizer、rank/alpha、学習対象モジュール、batch_size |
| 初期化 | LoRA初期化時のRNG条件（CPU／使用GPU）、初期stateの照合結果・ハッシュ。制御・確認できなければ未確認と記録 |
| 学習率 | 共通lrとself-attention／cross-attention／MLP等の実効lr。個別overrideがあれば共通lrだけの変更では比較条件が変わらない場合がある |
| 教師 | direction_loss、action、eta、各項目のguidance_scale、teacher_guidance_scale、積であるeffective_eta、teacher_norm_reference |
| 解像度等 | ノード指定とYAML値、解決後width/height、学習内steps・scheduler、timestep_sampling、flow shift、loss weighting |
| 評価 | initial/final eval実施有無、評価時刻・sigma、評価対象文。省略時は未評価として記録 |
| 推論 | 生成positive/negative全文、seed、width/height、steps、CFG、sampler、scheduler、denoise、LoRAの実適用強度（model/clip） |

同じseed設定に加え、レポートに記録された学習seed列・時刻サンプリング列も比較する。条件変更でこれらがずれた場合は報告し、文章だけの厳密な比較としない。全条件を同じ初期モデルから開始し、別条件の学習途中重みを引き継がない。

**学習seed一致だけでは初期LoRA重みはそろわない。** [lora_network.py](../../anima_slider_node/lora_network.py)の`LoRALinear`は`torch.nn.init.normal_`へ専用generatorを渡さず、デバイスのグローバルRNGでdown重みを初期化する。学習側の`request.seed`はlatentや時刻サンプリングに使われるが、この初期化とは結び付いていない。up重みはゼロでもdown重みの違いは学習に影響し得るため、同じcheckpoint・学習seed・eval_seedだけで厳密な対照としない。厳密比較では初期化RNGと初期stateの一致を別途確認する。現行ノードで制御・確認できない場合はその制約を記録し、各条件で複数学習を反復して傾向を判断する。単一のA/B差を方式の単独因果の確証とは呼ばない。

ageの実績条件は`eta=1.5`、YAML `guidance_scale=1.0`、`teacher_guidance_scale=1.0`、`teacher_norm_reference=target`だったが、他スライダーの最適値ではない。例えば既存YAMLのguidanceが2.0なら同じetaでも実効係数は3.0になる。A～D内では係数を固定し、係数探索が必要なら別実験に分ける。

## 6. 小さい比較から始める

1. **文章確認:** 既存ローダー／検証手順でYAMLの必須キー、サイズ、indicesを検証する。A/Bは同じ3～4学習項目から始めてもよいが、その場合は同じ部分集合に固定する。未学習の評価文を別に2本以上確保する。
2. **LoRAなし診断:** 同じseedで基準文と目的文をそれぞれ通常生成のpositiveへ入力し、目的概念をモデルが描けるか確認する。生成negativeは共通に保つ。この画像は学習内の教師予測の再現ではない。
3. **A/B/C比較:** 同じ学習設定で実行し、学習前後の評価を保存する。安価な短時間探索を行うなら全条件で同stepsとし、採用前には同じ本比較stepsで再学習する。
4. **推論比較:** 学習に使わない生成文2本以上×seed 2つ（例: 1、10000）で、同じ強度グリッドを生成する。初回候補は0、+0.5、+1、+1.5、+2。急変区間は0.1～0.25刻みを追加し、明確な破綻後は無理に上げない。
5. **D比較:** 最も説明可能な症状を1つ選び、1要素だけ変更する。CとDを同じ評価グリッドで比較する。
6. **一般化確認:** 良かった案だけ、未使用の衣装・背景・視点・画角へ広げる。全身用途なら全身を必須にする。別推論モデルは別条件として評価し、最初の比較へ混ぜない。採用候補は学習seedも変えて再確認する。

双方向Aの逆枝−1等は別表で記録する。C/Dの反対方向は、その方向用の独立LoRAの正強度と比較する。共通の中立基準を外れた文章では効果が変わる可能性があるため、一般化段階に入れて確かめる。

PNGはファイル名の強度と実行値を照合する。ageでは旧版LoRA画像の取り違え、および`3.5.png`と`4.png`の実強度の逆転が起きた。重みのハッシュを生成時に記録し、同名上書きも避ける。強度0画像のRGBハッシュが一致するかも確認し、不一致なら入力差を調べてから比較する。

## 7. 判定は効果と副作用を分ける

**同じ強度での比較と、同じ効果量での比較の両方を行う。** 例えばA+1とC+1に加えて、同程度の笑顔・雨量・頭身になるA+1とC+1.5も比較する。厳密な同効果量が選べなければ「近似」と記録する。目的効果が消えたため副作用が減った案を改善としない。

| 軸 | 記録方法 |
| --- | --- |
| 目的効果 | 0=なし、1=弱い、2=実用、3=過剰。概念ごとに事前に画像上の目安を定義 |
| 保持 | 表情、頭身、衣服、髪、背景、色、線・陰影、構図から目的外の軸を選び、各0=維持、1=軽微、2=明確に変化、3=破綻 |
| 調整性 | 強度増加に対する効果の順序、急変、戻り、使える強度範囲 |
| 一般化 | 未学習文・seed・別条件ごとの成功数／総数。良い画像だけを抜粋しない |

比較前に目的効果2を得る条件と、許容できない副作用を決める。例えば「評価文2本×2seed全てで目的効果2が得られ、衣服と背景の変化が1以下」を暫定合格条件にする。必要な頭身変化などは保持スコアから除外し、目的効果側で採点する。

- **採用候補:** 同程度の目的効果で副作用が減り、複数の未学習文・seedで再現する。一般化確認後に正規化を検討する。
- **失敗:** 目的効果を得る前に破綻する、保持を改善すると目的効果が失われる、単方向化による利点が比較条件下で見られない。成功しない概念も記録する。
- **保留:** 効果量の対応がない、特定seedだけ改善、条件や重みの対応が不足、学習が未収束。原因を断定せず不足条件を1つ補う。

MSEの低下は教師予測への適合の指標で、画像品質の点数ではない。Aの二枝平均とBの一枝、さらに教師を変えたC/Dの最終MSEを並べて品質順位を決めない。可能ならAの目的側枝を別に見つつ、最終判断は目的効果・保持・一般化の画像評価で行う。

## 8. 実験記録テンプレート

```markdown
# <concept> / <run ID> / <date>
- 仮説・調べる方向:
- 目的効果 / 動いてよい軸 / 保持軸:
- 合格条件 / 失敗条件:
- A/B/C/Dの対応、比較元run、今回だけ変えた要素:
- repo commit・差分 / YAML path・SHA-256:
- 学習元 / 推論元 / CLIP / VAE（名前とSHA-256）:
- direction_loss / action / eta / guidance / teacher_guidance / effective_eta / norm_reference:
- steps / rank・alpha / optimizer / 実効lrとoverride / 学習対象:
- LoRA初期化RNG条件 / 初期state照合・ハッシュ / 制御不能・未確認事項 / 学習反復数:
- 解決後解像度 / batch / 学習内scheduler・時刻設定 / loss weighting:
- train indices / eval indices / 学習seed・列 / eval_seed / 学習時間:
- 学習前後evalの有無・値・時刻 / レポートpath:
- LoRA path・SHA-256 / 生成ワークフローpath:
- 生成positive・negative / seed一覧 / 全推論設定 / 強度一覧:
- PNG入力監査 / ファイル名と実強度 / 強度0 RGB一致:

| 文・seed | 条件・実強度 | 目的効果 | 保持軸別の変化 | 画像path・所見 |
| --- | --- | --- | --- | --- |
| | | | | |

- 同強度比較の結果:
- 同効果量比較の対応と結果（近似なら明記）:
- 未学習文・seedの成功数 / 別条件・再学習seedの結果:
- 判定: 採用候補 / 失敗 / 保留
- 言えること / 言えないこと / 次に変える1要素:
```

文章設計の共通ルールは[Prompt variation guide](README.md)を併用する。実行前には現行ノードの設定・レポート項目を確認し、記録されない情報は別の実験台帳へ明示的に残す。
