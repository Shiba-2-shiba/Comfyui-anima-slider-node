# Prompt variation guide

このディレクトリの prompt YAML は、Anima/Cosmos RFlow 向け Slider LoRA の学習方向を定義する。各 prompt は `target` を基準に、`positive` と `unconditional` の差分だけが目的の slider 概念になるように作る。

## 基本方針

- `target` と `neutral` は、学習したい属性を含まない基準 prompt にする。
- `positive` は slider を強める側、`unconditional` は弱める側または反対側にする。
- `positive` と `unconditional` の差分は、できるだけ単一概念に絞る。
- 画風、品質タグ、人物数、構図、背景、服装の大枠は同じ prompt 内でそろえる。
- 変化させるバリエーションは、学習概念と混ざらない nuisance 条件に限定する。

## バリエーション軸

新しい prompt set は、次の軸を必要に応じて組み合わせる。すべてを増やすより、概念を乱さない軸を均等に入れる。

| 軸 | 目的 | 目安 |
| --- | --- | --- |
| 被写体属性 | 特定の髪色、性別表現、体型、顔立ちへの過学習を避ける | 4-8 種類 |
| 構図 | portrait / upper body / waist up / full body への偏りを避ける | 2-4 種類 |
| 視点 | front view / three-quarter view などへの固定を避ける | 2 種類程度 |
| 髪型・髪色 | slider 概念以外の外見条件を分散する | 4-8 種類 |
| 服装 | 属性が服装に吸着するのを避ける | 3-6 種類 |
| 背景 | 背景に概念が乗るのを避ける | simple / plain / studio 程度 |
| 解像度 | 実運用の aspect ratio に合わせる | 1MP 前後、検証済み範囲 |

## 採用スコア

prompt を追加する前に、各項目を 0-2 点で確認する。合計 12 点以上を採用目安にする。

| 項目 | 0 点 | 1 点 | 2 点 |
| --- | --- | --- | --- |
| 概念純度 | 複数概念が混ざる | 少し混ざる | 目的概念だけが差分 |
| 対称性 | positive/unconditional の条件が非対称 | 一部だけ非対称 | 差分以外はほぼ同一 |
| 安全性 | unsafe / 未成年 / 性的文脈がある | 表現が曖昧 | adult / safe / nonsexual が明確 |
| 多様性 | 既存 prompt とほぼ重複 | 1 軸だけ違う | 2 軸以上で有効に違う |
| 制御性 | 出力が別概念へ流れやすい | 少し不安定 | 画像上で確認しやすい属性 |
| 評価しやすさ | 結果判定が主観的すぎる | 比較に迷う | A/B 比較で変化が読める |
| 実行コスト | 高解像度または巨大 batch | やや重い | 既存設定と同程度 |

## 配分目安

- 最小検証: 3-4 prompts。概念が動くかを短時間で見る。
- 標準検証: 8-12 prompts。構図、髪色、服装を分散する。
- 本番寄り検証: 24 prompts 前後。被写体属性と構図を均等にし、評価用 indices も同数用意する。

同じバリエーション軸を増やしすぎない。例えば髪色だけ 12 種類にしても、構図が portrait だけなら portrait 専用 slider になりやすい。

## NG パターン

- `positive` だけ服装、背景、画風、解像度相当の情報が増えている。
- `unconditional` に目的概念と無関係な低品質タグや否定表現を入れている。
- 年齢系 slider で `child`, `teen`, `minor`, `schoolgirl`, `schoolboy`, `young girl`, `young boy`, `student`, `school uniform` を無自覚に使っている。
- 体型・胸部など身体特徴の slider で `adult`, `fully clothed`, `nonsexual` が不足している。
- 1 prompt 内で構図や背景まで反対方向に変えている。
- 既存 prompt の語順やタグ量から大きく外れて、差分方向が読みにくい。

## 学習結果からの修正推定

学習後の改善が弱い場合は、画像の症状から prompt の問題を推定して修正する。まず LoRA rank や steps を疑う前に、`positive` と `unconditional` の差分が本当に目的属性だけかを見る。

| 学習結果の症状 | 推定原因 | prompt 側の修正 |
| --- | --- | --- |
| slider を強くしても変化が薄い | 差分語が弱い、抽象的、または `target` 側にも目的属性が残っている | `positive` / `unconditional` の対比語を 2-4 個に絞って明確化し、`target` / `neutral` から目的属性を抜く |
| 変化は出るが別人化しやすい | 髪型、服装、顔立ちなども差分に混ざっている | 同一 prompt 内では髪、服、構図、背景を完全にそろえ、目的属性だけを入れ替える |
| 構図やポーズまで変わる | 構図語が positive/unconditional で非対称 | `portrait`, `upper body`, `front view` などを全フィールドで同じ位置に入れる |
| 背景や画風に効果が乗る | 背景・品質タグ・画風タグの分布が偏っている | 背景は `plain background` / `simple studio background` 程度に固定し、品質タグを全フィールドで統一する |
| 一部の prompt では効くが他では効かない | バリエーション軸が少なく、特定条件に過学習している | 効かない条件を 2-4 prompt 追加し、髪色・視点・構図を均等に増やす |
| 効果が強すぎて破綻する | 差分語が過剰、または極端な語が多い | 極端な語を減らし、観察しやすい中間表現に置き換える。`guidance_scale` も既存 set に合わせる |
| 逆方向が汚くなる | `unconditional` が低品質・否定・別カテゴリの prompt になっている | `unconditional` は低品質化ではなく、同じ品質の反対属性として書く |
| 目的属性以外の年齢・性別・体型が動く | 属性語が相互に絡んでいる | 目的外属性を `adult`, `fully clothed`, `nonsexual`, `same body proportions` などで固定する |
| 評価画像では良いが実用 prompt で崩れる | 学習 prompt が単純すぎる | 実用でよく使う構図・髪型・服装を少量ずつ追加し、複雑な背景や装飾は最後に足す |

修正は一度に複数軸へ広げない。最初は 3-4 prompts の小さい set で、差分語の書き換えだけを試す。次に効いた方向を 8-12 prompts へ拡張する。

## 追加時チェックリスト

- YAML は list of mappings で、各要素に `target`, `positive`, `unconditional`, `neutral`, `guidance_scale`, `action`, `width`, `height`, `batch_size` を入れる。
- `neutral` は原則 `target` と同じにする。
- `action` は通常 `enhance` にする。
- `guidance_scale` は既存 set と比較可能な値から始める。
- `width * height` は `validate_prompts` の上限を超えない。
- 新しい身体・年齢関連 set は、まず小さな prompt 数で cache dry run と短時間学習を行う。

## レビュー観点

新しい prompt set のレビューでは、生成文として自然かよりも、差分ベクトルとしてきれいかを優先する。良い set は「どの prompt でも、変えているのは slider で学習したい属性だけ」と説明できる。
