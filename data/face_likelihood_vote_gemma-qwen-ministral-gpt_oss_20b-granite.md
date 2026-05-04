# Face_likelihood — 5-way vote (gemma-qwen-ministral-gpt_oss_20b-granite)

**Encoders:** gemma, qwen, ministral, gpt_oss_20b, granite
**Ground-truth floor:** ≥3 v3 emissions
**Faces compared:** 573 (overlap)
**Faces with ground truth:** 166

## Per-encoder accuracy on GT subset

| encoder | correct | total | accuracy |
|---|---:|---:|---:|
| gemma | 86 | 166 | 51.8% |
| qwen | 50 | 166 | 30.1% |
| ministral | 53 | 166 | 31.9% |
| gpt_oss_20b | 74 | 166 | 44.6% |
| granite | 58 | 166 | 34.9% |

## Voting accuracy on GT subset

| scheme | correct | denom | accuracy | notes |
|---|---:|---:|---:|---|
| strict majority (≥2) | 88 | 159 | 55.3% | abstains on 7 all-distinct |
| confidence-weighted | 84 | 166 | 50.6% | argmax on Σ softmax |

## Vote strength distribution on GT subset

| strength | n | share |
|---|---:|---:|
| 2-1-1-1 | 40 | 24.1% |
| 2-2-1 | 36 | 21.7% |
| 3-1-1 | 33 | 19.9% |
| 4-1 | 27 | 16.3% |
| 3-2 | 16 | 9.6% |
| 5 | 7 | 4.2% |
| 1-1-1-1-1 | 7 | 4.2% |

## Pairwise agreement matrix (whole overlap)

| pair | agree | total | rate |
|---|---:|---:|---:|
| gemma ↔ qwen | 128 | 573 | 22.3% |
| gemma ↔ ministral | 165 | 573 | 28.8% |
| gemma ↔ gpt_oss_20b | 213 | 573 | 37.2% |
| gemma ↔ granite | 150 | 573 | 26.2% |
| qwen ↔ ministral | 115 | 573 | 20.1% |
| qwen ↔ gpt_oss_20b | 153 | 573 | 26.7% |
| qwen ↔ granite | 116 | 573 | 20.2% |
| ministral ↔ gpt_oss_20b | 147 | 573 | 25.7% |
| ministral ↔ granite | 216 | 573 | 37.7% |
| gpt_oss_20b ↔ granite | 161 | 573 | 28.1% |

## Pairwise agreement matrix (GT subset)

| pair | agree | total | rate |
|---|---:|---:|---:|
| gemma ↔ qwen | 45 | 166 | 27.1% |
| gemma ↔ ministral | 52 | 166 | 31.3% |
| gemma ↔ gpt_oss_20b | 78 | 166 | 47.0% |
| gemma ↔ granite | 50 | 166 | 30.1% |
| qwen ↔ ministral | 36 | 166 | 21.7% |
| qwen ↔ gpt_oss_20b | 48 | 166 | 28.9% |
| qwen ↔ granite | 34 | 166 | 20.5% |
| ministral ↔ gpt_oss_20b | 42 | 166 | 25.3% |
| ministral ↔ granite | 67 | 166 | 40.4% |
| gpt_oss_20b ↔ granite | 55 | 166 | 33.1% |

## Tie-breaker analysis: gemma ↔ qwen disagreements (n=121) — does ministral break them?

- ministral sides with **gemma**: 36/121 (29.8%)
- ministral sides with **qwen**: 20/121 (16.5%)
- ministral **dissents from both**: 65/121 (53.7%)

On the 56 cases where ministral sided with one of them (2-1 majority), majority-vote was correct on **31/56 = 55.4%**.

### 65 cases all 3 disagree (1-1-1)

| face | gemma | qwen | ministral | empirical | emits |
|---|---|---|---|---|---:|
| `(ﾉ◕ヮ◕)` | HP | LP | HN-D | HP | 407 |
| `(◕‿◕✿)` | LP | HP | NB | NB | 151 |
| `(๑˃‿˂)` | HP | LP | NB | HP | 113 |
| `(≧‿≦)` | HP | LN | HN-D | HP | 105 |
| `(￣▽￣)` | LP | HP | NB | NB | 72 |
| `(・̀‿・́)` | HN-S | HP | NB | NB | 69 |
| `(︿︿)` | NB | HN-D | LN | HN-D | 49 |
| `(´ω`)` | LN | LP | NB | LN | 48 |
| `(>_<)` | HN-S | LP | HN-D | HN-S | 44 |
| `(｡╯︵╰｡)` | LN | HN-D | NB | LN | 39 |
| `(・̀ω・́)` | HN-S | HN-D | NB | NB | 34 |
| `(＾‿＾)` | NB | HP | HN-D | NB | 34 |
| `(ﾟдﾟ)` | HN-S | HN-D | LN | HN-D | 30 |
| `(` | LN | HN-D | NB | LP | 30 |
| `(o_o)` | HN-S | HP | HN-D | NB | 24 |
| `(´艸`)` | LP | HP | NB | LN | 24 |
| `(・_・;)` | HN-S | HN-D | LN | NB | 22 |
| `(｡・̀‿-)` | NB | LP | LN | NB | 21 |
| `(⌒▽⌒)` | NB | HP | LN | NB | 21 |
| `(=^・ω・^=)` | HN-D | HP | LP | LP | 19 |
| `(⊙﹏⊙)` | HN-S | HN-D | LN | HN-S | 19 |
| `(¬‿¬)` | NB | HN-D | LN | NB | 15 |
| `(⊙_⊙;)` | HN-S | LN | HN-D | HN-S | 13 |
| `(>_<;)` | HN-S | HN-D | LN | HN-S | 13 |
| `(￣▽￣;)` | HN-S | HP | LN | HN-S | 10 |
| `(｡˃‿˂)` | HP | LP | NB | HP | 10 |
| `(˙꒳˙)` | NB | LP | LN | NB | 8 |
| `(ᵔㅂᵔ)` | HP | LP | HN-S | HN-S | 7 |
| `(>﹏<｡)` | HN-S | LP | LN | HN-S | 7 |
| `(°◇°)` | HN-S | HP | NB | NB | 7 |
| `(=^-^=)` | LN | HN-S | LP | LP | 6 |
| `(￣∇￣;)` | LN | HP | NB | LP | 6 |
| `(o^_^o)` | NB | HP | LP | NB | 6 |
| `(´-`)` | LN | HN-D | NB | LN | 6 |
| `(;;)` | HN-S | LP | NB | LN | 6 |
| `(☞ﾟヮﾟ)` | HP | HN-D | NB | HN-S | 5 |
| `(ﾟｰﾟ)` | HP | HN-D | LN | LN | 5 |
| `(≧ω≦)` | HP | LP | NB | LP | 5 |
| `(˃⌑˂)` | HP | HN-S | NB | LN | 5 |
| `(⊙◞⊙)` | HN-S | HP | NB | HN-S | 5 |
| `(⊙_☉;)` | HN-S | HN-D | LN | HN-S | 5 |
| `(´∀`)` | LN | HP | NB | LP | 5 |
| `(´∇`)` | LP | HP | NB | HP | 5 |
| `(ノ◕ヮ◕)` | HP | LP | NB | HP | 4 |
| `(ᵒ̌ᵃᵒ̌)` | HP | HN-D | HN-S | HP | 4 |
| `(°〇°)` | HN-S | HP | NB | HP | 4 |
| `(ﾟ∇ﾟ)` | HP | LP | LN | HP | 4 |
| `(ᵒ̌ᵐᵒ̌)` | HP | HN-D | HN-S | HN-S | 4 |
| `(ᵔㅅᵔ)` | HP | HN-D | NB | HN-D | 4 |
| `(◠‿◠)` | NB | LP | LN | HP | 3 |
| `(๑・́₃・̀๑)` | HN-S | HN-D | NB | LN | 3 |
| `(￣_￣)` | HN-D | HN-S | LN | HN-D | 3 |
| `(ﾟヘﾟ)` | HP | HN-D | LN | HN-S | 3 |
| `(ﾟ‿ﾟ)` | NB | HN-S | LN | LP | 3 |
| `(¯︿¯)` | NB | HN-D | LN | LN | 3 |
| `(✿◕‿◕✿)` | HP | LP | NB | LP | 3 |
| `(╯✧▽✧)` | HP | LP | NB | HP | 3 |
| `(ｏ・ω・)` | NB | LP | HN-D | HP | 3 |
| `(°д°)` | HN-S | HP | NB | HN-S | 3 |
| `(´-ω-`)` | LN | LP | NB | LN | 3 |
| `(︵︵)` | LN | HN-D | HN-S | HN-S | 3 |
| `(°ロ°)` | HN-S | HP | NB | HN-S | 3 |
| `(・_・ヾ` | HN-S | HN-D | NB | NB | 3 |
| `(⊙_◎)` | HN-S | LP | NB | HN-S | 3 |
| `(๑><๑)` | HN-S | LP | HN-D | LP | 3 |

## Per-empirical-quadrant accuracy

| empirical | n | gemma | qwen | ministral | gpt_oss_20b | granite | majority | weighted |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HP | 28 | 17/28 | 7/28 | 3/28 | 14/28 | 1/28 | 13/26 | 13/28 |
| LP | 30 | 11/30 | 15/30 | 7/30 | 10/30 | 14/30 | 14/29 | 13/30 |
| HN-D | 19 | 7/19 | 12/19 | 6/19 | 9/19 | 4/19 | 8/18 | 5/19 |
| HN-S | 28 | 18/28 | 4/28 | 7/28 | 15/28 | 1/28 | 17/27 | 12/28 |
| LN | 32 | 14/32 | 9/32 | 13/32 | 13/32 | 23/32 | 19/31 | 22/32 |
| NB | 29 | 19/29 | 3/29 | 17/29 | 13/29 | 15/29 | 17/28 | 19/29 |

## Faces where every encoder missed empirical (n=23)

These can't be recovered by any vote scheme; they bound the ceiling on cross-encoder agreement with v3 sampling.

| face | gemma_pred | qwen_pred | ministral_pred | gpt_oss_20b_pred | granite_pred | empirical | emits |
|---|---|---|---|---|---|---|---|
| `(´・ω・`)` | LN | LP | LN | LN | LN | NB | 68 |
| `(◕‿◕)` | NB | HN-D | NB | NB | NB | HP | 40 |
| `(^-^)` | NB | NB | NB | NB | NB | LP | 37 |
| `(` | LN | HN-D | NB | HN-S | LN | LP | 30 |
| `(o_o)` | HN-S | HP | HN-D | HN-S | LN | NB | 24 |
| `(・_・;)` | HN-S | HN-D | LN | LN | LN | NB | 22 |
| `(⁀ᗜ⁀)` | HP | HP | HN-D | HN-S | NB | LN | 19 |
| `(๑・̀ㅁ・́๑)` | HN-S | HN-S | LN | HP | LN | LP | 17 |
| `(˘▽˘)` | LP | LP | NB | LP | LP | HP | 10 |
| `(￣ω￣;)` | LN | LN | LN | LN | LN | HN-S | 10 |
| `(`⌒´)` | LN | LN | NB | NB | NB | HP | 7 |
| `(￣∇￣;)` | LN | HP | NB | HN-D | LN | LP | 6 |
| `(˃⌑˂)` | HP | HN-S | NB | LP | HP | LN | 5 |
| `(`⌒´メ)` | LN | LN | NB | HN-S | NB | HP | 5 |
| `(️️)` | HN-S | HP | HN-S | HN-S | LN | HN-D | 5 |
| `(ˆ‿ˆԅ)` | LP | LP | LN | LP | LN | HP | 3 |
| `(>‿◠)` | HP | HP | LP | HN-D | LN | HN-S | 3 |
| `(◠‿◠)` | NB | LP | LN | NB | NB | HP | 3 |
| `(　´∀`　)` | LP | HP | HP | LN | NB | HN-S | 3 |
| `(〃ﾟ3ﾟ〃)` | HP | HP | LP | HN-S | NB | HN-D | 3 |
| `(ｏ・ω・)` | NB | LP | HN-D | HN-S | LP | HP | 3 |
| `(ﾟ‿ﾟ)` | NB | HN-S | LN | HP | LN | LP | 3 |
| `(￣^￣)` | NB | HN-S | NB | NB | NB | LP | 3 |
