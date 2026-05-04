# Anima Slider prompt YAML set

This directory contains Anima Slider training prompt YAML files.

## 8プロンプト構成のYAML

The 44 files below use:
- 8 prompts
- neutral == target
- action: enhance
- batch_size: 1
- positive/unconditional differ by the intended slider axis

## ファイル一覧

### 年齢・顔立ち・体型

- `prompts-anima-age_slider.yaml` — `age_slider`。期待される効果: 成人キャラクターの外見年齢を、若々しい成人顔から年齢感のある成熟した顔立ちへ寄せる。
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

## その他のYAML

These files are present in the same directory but are not part of the 8-prompt set. They are listed here for completeness:

- `prompts-anima-age_slider_fullbody.yaml` — 3 prompts。期待される効果: 全身または上半身の構図で、幼い体型・顔立ちから高齢女性の顔立ち、手、姿勢、体格へ大きく年齢方向を変化させる。
- `prompts-anima-age_slider_v2.yaml` — 24 prompts。期待される効果: 髪色、髪型、構図のバリエーションを多く持たせながら、成人女性の顔を若い成人印象から成熟した年齢感のある顔立ちへ寄せる。
