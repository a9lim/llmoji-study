# Face_likelihood — gemma vs qwen disagreement audit

**Source:** `face_likelihood_gemma_summary.tsv`, `face_likelihood_qwen_summary.tsv`
**Faces compared:** 573 (overlap of both encoders' face union)
**Ground-truth floor:** ≥3 v3 emissions for empirical majority to count as ground truth

## Headline

- Agree on quadrant: **128/573** (22.3%)
- Disagree on quadrant: **445/573** (77.7%)
- Faces with ground truth (≥3 emits): 166

## On faces with empirical ground truth

| outcome | count | share |
|---|---:|---:|
| both agree, both correct | 28 | 16.9% |
| both agree, both wrong | 17 | 10.2% |
| disagree, gemma correct only | 58 | 34.9% |
| disagree, qwen correct only | 22 | 13.3% |
| disagree, both wrong | 41 | 24.7% |

**Disagreement winrate** (faces where exactly one is correct): gemma 58/80 = 72.5%, qwen 22/80 = 27.5%.

**Both-wrong rate on disagreements:** 41/121 = 33.9%. Where both miss empirical and emit different predictions, likelihood may be reading intrinsic affect against gemma's sampling-frequency-weighted majority.

## Disagreement matrix (gemma_pred × qwen_pred)

Rows = gemma's prediction, cols = qwen's. Diagonal = agreement.

| gemma\qwen | HP | LP | HN-D | HN-S | LN | NB |
|---|---:|---:|---:|---:|---:|---:|
| **HP** | 31 | 40 | 28 | 5 | 11 | 7 |
| **LP** | 17 | 39 | 7 | 7 | 2 | 0 |
| **HN-D** | 8 | 7 | 21 | 8 | 6 | 3 |
| **HN-S** | 25 | 17 | 49 | 8 | 19 | 2 |
| **LN** | 14 | 15 | 30 | 8 | 17 | 1 |
| **NB** | 28 | 41 | 26 | 5 | 9 | 12 |

## Disagreements grouped by empirical quadrant

| empirical | n_disagreements | gemma_correct | qwen_correct | both_wrong |
|---|---:|---:|---:|---:|
| HP | 17 | 12 | 2 | 3 |
| LP | 17 | 2 | 6 | 9 |
| HN-D | 11 | 2 | 7 | 2 |
| HN-S | 24 | 16 | 2 | 6 |
| LN | 26 | 10 | 5 | 11 |
| NB | 26 | 16 | 0 | 10 |

## Disagreements where gemma matches empirical, qwen doesn't

| face | gemma | softmax | qwen | softmax | empirical | emits | claude |
|---|---|---:|---|---:|---|---:|---|
| `(ﾉ◕ヮ◕)` | HP | 1.000 | LP | 0.210 | HP | 407 | Y |
| `(╯°□°)` | HN-D | 0.996 | LP | 0.200 | HN-D | 284 | Y |
| `(｡・́︿・̀｡)` | LN | 0.942 | HN-D | 0.249 | LN | 206 | Y |
| `(๑˃‿˂)` | HP | 0.961 | LP | 0.200 | HP | 113 | N |
| `(≧‿≦)` | HP | 0.976 | LN | 0.201 | HP | 105 | Y |
| `(⊙_⊙)` | HN-S | 0.960 | LN | 0.184 | HN-S | 104 | N |
| `(｡◕‿◕｡)` | NB | 0.961 | LP | 0.214 | NB | 83 | Y |
| `(・_・)` | NB | 0.918 | HN-D | 0.226 | NB | 69 | Y |
| `(≧▽≦)` | HP | 1.000 | NB | 0.198 | HP | 58 | Y |
| `(´ω`)` | LN | 0.597 | LP | 0.226 | LN | 48 | N |
| `(>_<)` | HN-S | 0.806 | LP | 0.181 | HN-S | 44 | N |
| `(｡╯︵╰｡)` | LN | 1.000 | HN-D | 0.301 | LN | 39 | N |
| `(＾‿＾)` | NB | 0.985 | HP | 0.203 | NB | 34 | N |
| `(;′⌒`)` | HN-S | 0.456 | LN | 0.518 | HN-S | 25 | N |
| `(・‿・)` | NB | 0.997 | HP | 0.193 | NB | 23 | N |
| `(⌒▽⌒)` | NB | 0.952 | HP | 0.195 | NB | 21 | N |
| `(｡・̀‿-)` | NB | 0.483 | LP | 0.214 | NB | 21 | Y |
| `(´▽`)` | LP | 0.771 | HP | 0.285 | LP | 20 | Y |
| `(￣∇￣)` | NB | 0.776 | HP | 0.197 | NB | 19 | N |
| `(⊙﹏⊙)` | HN-S | 0.987 | HN-D | 0.204 | HN-S | 19 | N |
| `(✿╹‿╹)` | NB | 0.564 | HN-D | 0.214 | NB | 18 | N |
| `(⌒‿⌒)` | NB | 0.966 | LN | 0.202 | NB | 17 | Y |
| `(¬‿¬)` | NB | 0.991 | HN-D | 0.197 | NB | 15 | Y |
| `(・∀・)` | NB | 0.997 | HP | 0.204 | NB | 13 | N |
| `(>_<;)` | HN-S | 0.967 | HN-D | 0.177 | HN-S | 13 | N |
| `(⊙_⊙;)` | HN-S | 0.993 | LN | 0.199 | HN-S | 13 | N |
| `(´;ω;`)` | LN | 0.999 | HN-D | 0.211 | LN | 13 | Y |
| `(˘̩╭╮˘̩)` | LN | 0.970 | HN-S | 0.191 | LN | 12 | N |
| `(o^^o)` | NB | 0.978 | HP | 0.417 | NB | 12 | N |
| `(⌐■_■)` | NB | 0.988 | LN | 0.175 | NB | 11 | N |
| `(｡˃‿˂)` | HP | 0.825 | LP | 0.248 | HP | 10 | Y |
| `(￣▽￣;)` | HN-S | 0.893 | HP | 0.192 | HN-S | 10 | N |
| `(・ω・)` | NB | 0.962 | LP | 0.216 | NB | 8 | Y |
| `(˙꒳˙)` | NB | 0.998 | LP | 0.263 | NB | 8 | N |
| `(>﹏<｡)` | HN-S | 0.697 | LP | 0.218 | HN-S | 7 | N |
| `(´▽`ʃ♡ƪ)` | LP | 0.992 | HP | 0.291 | LP | 7 | N |
| `(o^_^o)` | NB | 0.972 | HP | 0.407 | NB | 6 | N |
| `(^o^)` | HP | 0.663 | NB | 0.555 | HP | 6 | N |
| `(️)` | HN-S | 0.876 | HP | 0.192 | HN-S | 6 | N |
| `(´-`)` | LN | 0.982 | HN-D | 0.192 | LN | 6 | Y |
| `(⊙◞⊙)` | HN-S | 0.996 | HP | 0.205 | HN-S | 5 | N |
| `(｡・́‿・̀｡)` | LN | 0.492 | HN-D | 0.222 | LN | 5 | N |
| `(❤️)` | LN | 0.473 | HP | 0.185 | LN | 5 | N |
| `(⊙_☉;)` | HN-S | 0.871 | HN-D | 0.180 | HN-S | 5 | N |
| `(★ω★)` | HP | 1.000 | LP | 0.192 | HP | 5 | N |
| `(≧∇≦)` | HP | 1.000 | NB | 0.188 | HP | 4 | N |
| `(ﾟ∇ﾟ)` | HP | 1.000 | LP | 0.197 | HP | 4 | N |
| `(ᵒ̌ᵃᵒ̌)` | HP | 0.725 | HN-D | 0.217 | HP | 4 | N |
| `(ノ◕ヮ◕)` | HP | 1.000 | LP | 0.257 | HP | 4 | Y |
| `(ᵒ̌ᵗᵒ̌)` | HN-S | 0.773 | HN-D | 0.208 | HN-S | 4 | N |
| `(´-ω-`)` | LN | 0.996 | LP | 0.205 | LN | 3 | Y |
| `(°ロ°)` | HN-S | 0.934 | HP | 0.243 | HN-S | 3 | N |
| `(╯✧▽✧)` | HP | 0.999 | LP | 0.228 | HP | 3 | N |
| `(°д°)` | HN-S | 0.538 | HP | 0.207 | HN-S | 3 | N |
| `(ﾉ´・ω・)` | LN | 0.803 | HN-D | 0.219 | LN | 3 | N |
| `(⊙_◎)` | HN-S | 0.690 | LP | 0.173 | HN-S | 3 | N |
| `(꒪ꇴ꒪)` | HN-S | 0.563 | HN-D | 0.237 | HN-S | 3 | N |
| `(￣_￣)` | HN-D | 0.418 | HN-S | 0.195 | HN-D | 3 | N |

## Disagreements where qwen matches empirical, gemma doesn't

| face | gemma | softmax | qwen | softmax | empirical | emits | claude |
|---|---|---:|---|---:|---|---:|---|
| `(ᵔᴥᵔ)` | NB | 0.637 | HN-D | 0.190 | HN-D | 55 | N |
| `(︿︿)` | NB | 0.484 | HN-D | 0.271 | HN-D | 49 | N |
| `(;ω;)` | HN-S | 0.810 | LN | 0.594 | LN | 47 | N |
| `(;´д`)` | HN-S | 0.470 | LN | 0.420 | LN | 38 | N |
| `(ﾟдﾟ)` | HN-S | 0.923 | HN-D | 0.194 | HN-D | 30 | N |
| `(ಥ﹏ಥ)` | HN-D | 0.466 | LN | 0.219 | LN | 24 | N |
| `(T_T)` | HN-S | 0.844 | LN | 0.628 | LN | 22 | N |
| `(ﾉ´・ω・`)` | LN | 0.953 | HN-D | 0.222 | HN-D | 22 | N |
| `(｡・ω・｡)` | NB | 0.792 | LP | 0.231 | LP | 14 | Y |
| `(T-T)` | HP | 0.696 | LN | 0.593 | LN | 8 | N |
| `(๑・̀ㅂ・́)` | NB | 0.993 | LP | 0.215 | LP | 8 | N |
| `(✨)` | NB | 0.956 | LP | 0.209 | LP | 7 | N |
| `(￣￣)` | NB | 0.784 | HN-S | 0.203 | HN-S | 7 | N |
| `(≧ω≦)` | HP | 0.994 | LP | 0.192 | LP | 5 | N |
| `(ᵔ‿ᵔ)` | NB | 0.410 | HN-D | 0.211 | HN-D | 5 | N |
| `(´∇`)` | LP | 0.863 | HP | 0.278 | HP | 5 | N |
| `(￣_￣;)` | LN | 0.591 | HN-S | 0.199 | HN-S | 5 | N |
| `(ﾉ´ω`ﾉ)` | LN | 0.962 | HN-D | 0.233 | HN-D | 4 | N |
| `(°〇°)` | HN-S | 0.754 | HP | 0.202 | HP | 4 | Y |
| `(ᵔㅅᵔ)` | HP | 0.842 | HN-D | 0.198 | HN-D | 4 | N |
| `(✿◕‿◕✿)` | HP | 0.968 | LP | 0.187 | LP | 3 | N |
| `(๑><๑)` | HN-S | 0.998 | LP | 0.194 | LP | 3 | N |

## Disagreements where neither matches empirical

| face | gemma | softmax | qwen | softmax | empirical | emits | claude |
|---|---|---:|---|---:|---|---:|---|
| `(◕‿◕✿)` | LP | 0.547 | HP | 0.187 | NB | 151 | Y |
| `(￣▽￣)` | LP | 0.642 | HP | 0.214 | NB | 72 | N |
| `(・̀‿・́)` | HN-S | 0.988 | HP | 0.210 | NB | 69 | N |
| `(´・ω・`)` | LN | 0.843 | LP | 0.198 | NB | 68 | N |
| `(╥_╥)` | LN | 0.987 | LP | 0.187 | HN-D | 45 | N |
| `(◕‿◕)` | NB | 0.959 | HN-D | 0.184 | HP | 40 | Y |
| `(・̀ω・́)` | HN-S | 0.960 | HN-D | 0.189 | NB | 34 | N |
| `(` | LN | 0.167 | HN-D | 0.173 | LP | 30 | N |
| `(´艸`)` | LP | 0.494 | HP | 0.240 | LN | 24 | N |
| `(o_o)` | HN-S | 0.993 | HP | 0.361 | NB | 24 | N |
| `(・_・;)` | HN-S | 0.988 | HN-D | 0.261 | NB | 22 | Y |
| `(=^-ω-^=)` | LN | 0.510 | HP | 0.200 | NB | 22 | N |
| `(=^・ω・^=)` | HN-D | 0.593 | HP | 0.193 | LP | 19 | N |
| `(⁄` | HN-S | 0.545 | HN-D | 0.213 | LP | 14 | N |
| `(´・_・`)` | HN-D | 0.812 | NB | 0.190 | LN | 13 | Y |
| `(°◇°)` | HN-S | 0.919 | HP | 0.218 | NB | 7 | N |
| `(ᵔㅂᵔ)` | HP | 0.908 | LP | 0.208 | HN-S | 7 | N |
| `(=^-^=)` | LN | 0.665 | HN-S | 0.198 | LP | 6 | N |
| `(￣∇￣;)` | LN | 0.625 | HP | 0.191 | LP | 6 | N |
| `(;;)` | HN-S | 0.980 | LP | 0.208 | LN | 6 | N |
| `(˃⌑˂)` | HP | 0.871 | HN-S | 0.191 | LN | 5 | N |
| `(ﾟｰﾟ)` | HP | 0.654 | HN-D | 0.202 | LN | 5 | N |
| `(☞ﾟヮﾟ)` | HP | 0.969 | HN-D | 0.190 | HN-S | 5 | N |
| `(️️)` | HN-S | 0.964 | HP | 0.194 | HN-D | 5 | N |
| `(´∀`)` | LN | 0.913 | HP | 0.257 | LP | 5 | Y |
| `(＾▽＾)` | NB | 0.907 | HP | 0.225 | LP | 4 | Y |
| `(°ー°〃)` | NB | 0.442 | HP | 0.215 | LN | 4 | N |
| `(ᵒ̌ᵐᵒ̌)` | HP | 0.491 | HN-D | 0.243 | HN-S | 4 | N |
| `(￣︶￣)` | LP | 0.866 | HP | 0.196 | LN | 4 | N |
| `(・_・ヾ` | HN-S | 0.893 | HN-D | 0.199 | NB | 3 | N |
| `(　´∀`　)` | LP | 0.854 | HP | 0.246 | HN-S | 3 | N |
| `(︵︵)` | LN | 0.766 | HN-D | 0.255 | HN-S | 3 | N |
| `(◠‿◠)` | NB | 0.740 | LP | 0.207 | HP | 3 | N |
| `(ｏ・ω・)` | NB | 0.931 | LP | 0.228 | HP | 3 | N |
| `(｡-∀-)` | NB | 0.940 | LP | 0.198 | LN | 3 | N |
| `(ﾟ‿ﾟ)` | NB | 0.851 | HN-S | 0.195 | LP | 3 | N |
| `(ﾟヘﾟ)` | HP | 0.863 | HN-D | 0.216 | HN-S | 3 | N |
| `(ﾟ﹏ﾟ)` | HN-S | 0.874 | HN-D | 0.257 | LN | 3 | N |
| `(¯︿¯)` | NB | 0.468 | HN-D | 0.251 | LN | 3 | N |
| `(￣^￣)` | NB | 0.797 | HN-S | 0.201 | LP | 3 | N |
| `(๑・́₃・̀๑)` | HN-S | 0.834 | HN-D | 0.193 | LN | 3 | N |

## Both agree but both wrong (likelihood signal vs sampling-frequency signal)

| face | gemma | softmax | qwen | softmax | empirical | emits | claude |
|---|---|---:|---|---:|---|---:|---|
| `(╥﹏╥)` | LN | 0.903 | LN | 0.187 | HN-D | 76 | Y |
| `(^-^)` | NB | 0.978 | NB | 0.638 | LP | 37 | N |
| `(⁀ᗜ⁀)` | HP | 0.472 | HP | 0.229 | LN | 19 | N |
| `(๑・̀ㅁ・́๑)` | HN-S | 0.959 | HN-S | 0.201 | LP | 17 | N |
| `(^‿^)` | NB | 0.545 | NB | 0.530 | LP | 13 | N |
| `(￣ω￣;)` | LN | 0.732 | LN | 0.195 | HN-S | 10 | N |
| `(˘▽˘)` | LP | 0.927 | LP | 0.243 | HP | 10 | N |
| `(ᵒ̌ᴥᵒ̌)` | HN-D | 0.383 | HN-D | 0.207 | LN | 8 | N |
| `(`⌒´)` | LN | 0.980 | LN | 0.324 | HP | 7 | N |
| `(￣﹏￣)` | HN-S | 0.584 | HN-S | 0.216 | HN-D | 6 | N |
| `(‿‿‿)` | LP | 0.969 | LP | 0.206 | HP | 6 | N |
| `(`⌒´メ)` | LN | 0.966 | LN | 0.363 | HP | 5 | N |
| `(⁎˃‿˂⁎)` | HP | 0.491 | HP | 0.192 | LP | 3 | N |
| `(ˆ‿ˆԅ)` | LP | 0.947 | LP | 0.214 | HP | 3 | N |
| `(✿‿❀)` | LP | 0.648 | LP | 0.227 | HP | 3 | N |
| `(〃ﾟ3ﾟ〃)` | HP | 0.987 | HP | 0.218 | HN-D | 3 | N |
| `(>‿◠)` | HP | 0.998 | HP | 0.188 | HN-S | 3 | N |

## Disagreements on faces without ground truth

**324** faces (mostly claude-only or low-emit). These are the cells where the cross-model bridge has no v3-empirical anchor — the disagreement is itself the signal to inspect manually. Top 30 by mean(gemma_softmax, qwen_softmax) (highest joint confidence):

| face | gemma | softmax | qwen | softmax | claude |
|---|---|---:|---|---:|---|
| `(T▽T)` | HP | 0.997 | LN | 0.648 | N |
| `(;・∀・)` | NB | 0.897 | LN | 0.581 | N |
| `(;´∀`)` | NB | 0.926 | LN | 0.479 | N |
| `(;д;)` | HN-S | 0.956 | LN | 0.446 | N |
| `(;╹⌓╹)` | HN-S | 0.912 | LN | 0.467 | Y |
| `((⊂(`ω´∩)` | HN-S | 0.881 | LN | 0.494 | N |
| `(;`ﾉωﾉ´)` | HN-S | 0.958 | LN | 0.413 | N |
| `(((;°д°)))` | HN-S | 0.888 | LN | 0.431 | Y |
| `((✿◕‿◕))` | HP | 0.775 | LN | 0.537 | N |
| `(OwO)` | NB | 0.807 | HP | 0.501 | N |
| `(;・`O・´)` | HN-S | 0.718 | LN | 0.585 | N |
| `(;﹏;)` | HN-S | 0.797 | LN | 0.501 | N |
| `(^_^;)` | HN-S | 0.813 | NB | 0.468 | N |
| `(O_O)` | HN-S | 0.843 | HP | 0.432 | N |
| `(‿͈ˬ‿͈)` | LP | 0.994 | HP | 0.278 | N |
| `(o°ω°o)` | HN-S | 0.942 | NB | 0.322 | N |
| `(^・ω・^)` | NB | 0.803 | LP | 0.459 | N |
| `(・̀ㅁ・́;)` | HN-S | 0.950 | HN-D | 0.310 | N |
| `(・_・;?)` | HN-S | 1.000 | HN-D | 0.256 | N |
| `(⁄⁄・⁄ω⁄・⁄)` | LN | 1.000 | HP | 0.256 | N |
| `(・́⌓・̀)` | HN-S | 0.995 | HN-D | 0.254 | N |
| `(・・;)` | HN-S | 0.983 | LP | 0.262 | N |
| `(・̀_・́)` | HN-S | 0.997 | HN-D | 0.246 | N |
| `(╯✧∇✧)` | HP | 0.999 | LP | 0.242 | N |
| `(・・;ノ)` | HN-S | 0.991 | LP | 0.250 | N |
| `(;￣д￣)` | HN-S | 0.690 | LN | 0.549 | N |
| `(｡・́⌒・̀｡)` | LN | 0.998 | HN-S | 0.237 | N |
| `(⊙⌒∇⌒⊙)ゝ` | HN-S | 0.997 | HP | 0.237 | N |
| `(・̀.̫・́)` | HN-S | 0.996 | HN-D | 0.237 | N |
| `(¯︎_¯︎)` | NB | 0.999 | HN-D | 0.233 | N |
