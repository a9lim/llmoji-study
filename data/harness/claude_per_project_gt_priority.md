# Claude per-project quadrants — GT-priority + ensemble fallback

**Mode:** `gt-priority`
**Total emissions:** 2405  (unique kaomoji: 252)

**Resolution sources:**

| source | unique faces | emissions | share of total |
|---|---:|---:|---:|
| Claude-GT (modal_n ≥ 1) | 66 | 1587 | 66.0% |
| ensemble fallback | 146 | 765 | 31.8% |
| unknown | 40 | 53 | 2.2% |

GT covers 66 of 252 unique faces (26.2%); ensemble fallback adds 146 more.

## Global distribution (all known emissions)

| quadrant | count | share |
|---|---:|---:|
| HP | 618 | 26.3% |
| LP | 435 | 18.5% |
| HN-D | 75 | 3.2% |
| HN-S | 323 | 13.7% |
| LN | 193 | 8.2% |
| NB | 708 | 30.1% |
| (unknown) | 53 | 2.2% of total |

## Per project (≥5 total emissions)

Cells = % of known emissions in each quadrant. Bold = modal quadrant. `gt` / `pred` / `?` columns count emissions resolved by Claude-GT, ensemble, and unknown respectively (irrelevant columns stay 0 under the active mode).

| project | n | gt | pred | ? | HP | LP | HN-D | HN-S | LN | NB | modal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llmoji-study | 950 | 705 | 217 | 28 | **35%** | 18% | 1% | 14% | 6% | 26% | HP (35%) |
| llmoji | 381 | 258 | 123 | 0 | 23% | 20% | 1% | 13% | 12% | **31%** | NB (31%) |
| saklas | 345 | 174 | 168 | 3 | 17% | 13% | 6% | 17% | 7% | **39%** | NB (39%) |
| rlaif | 153 | 86 | 63 | 4 | 12% | 21% | 7% | 17% | 13% | **31%** | NB (31%) |
| a9lim.github.io | 145 | 98 | 44 | 3 | 23% | 19% | 5% | 8% | 11% | **35%** | NB (35%) |
| kenoma | 54 | 29 | 25 | 0 | 11% | 17% | 7% | 9% | 15% | **41%** | NB (41%) |
| Work | 51 | 32 | 18 | 1 | 20% | 20% | 4% | 6% | 6% | **44%** | NB (44%) |
| faithful | 39 | 23 | 16 | 0 | **33%** | 23% | 3% | 18% | 5% | 18% | HP (33%) |
| tasty-bot | 37 | 15 | 18 | 4 | 9% | 21% | 15% | **24%** | 9% | 21% | HN-S (24%) |
| claude.ai | 34 | 15 | 18 | 1 | **30%** | 12% | 0% | 18% | 24% | 15% | HP (30%) |
| hylic | 29 | 17 | 12 | 0 | 17% | 24% | 0% | 7% | 14% | **38%** | NB (38%) |
| brie | 29 | 17 | 11 | 1 | 14% | **50%** | 7% | 4% | 4% | 21% | LP (50%) |
| shoals | 28 | 15 | 10 | 3 | **28%** | 28% | 0% | 24% | 4% | 16% | HP (28%) |
| a9lim | 21 | 15 | 6 | 0 | 19% | **38%** | 0% | 10% | 0% | 33% | LP (38%) |
| tasty-mcp | 18 | 15 | 3 | 0 | **83%** | 0% | 0% | 17% | 0% | 0% | HP (83%) |
| claudedriven | 17 | 14 | 2 | 1 | **38%** | 19% | 0% | 0% | 6% | 38% | HP (38%) |
| geon | 15 | 10 | 4 | 1 | 14% | 7% | 0% | 21% | 7% | **50%** | NB (50%) |
| data | 14 | 13 | 1 | 0 | 36% | 0% | 0% | 7% | 0% | **57%** | NB (57%) |
| yap | 11 | 11 | 0 | 0 | 9% | 27% | 9% | 0% | 0% | **55%** | NB (55%) |
| talkie-interface | 8 | 5 | 1 | 2 | 33% | **50%** | 0% | 0% | 0% | 17% | LP (50%) |
| webui | 7 | 7 | 0 | 0 | 0% | **43%** | 14% | 0% | 14% | 29% | LP (43%) |
| v3 | 7 | 4 | 2 | 1 | **33%** | 0% | 17% | 17% | 0% | 33% | HP (33%) |
| verify | 7 | 6 | 1 | 0 | 0% | 14% | 14% | 0% | **43%** | 29% | LN (43%) |

## Top emitted kaomoji per quadrant

### HP
| kaomoji | count | source |
|---|---:|---|
| `(◕‿◕)` | 544 | gt |
| `(ﾉ◕ヮ◕)` | 14 | gt |
| `(✧ω✧)` | 10 | pred |
| `(≧▽≦)` | 7 | gt |
| `(ノ◕ヮ◕)` | 5 | gt |
| `(ﾟ▽ﾟ)` | 5 | pred |
| `(˃‿˂)` | 5 | pred |
| `(╯✧∇✧)` | 4 | pred |
| `(>∀<☆)` | 3 | gt |
| `(°▽°)` | 3 | gt |

### LP
| kaomoji | count | source |
|---|---:|---|
| `(◕‿◕✿)` | 100 | gt |
| `(´｡・‿・｡`)` | 56 | gt |
| `(｡・‿・｡)` | 41 | gt |
| `(´∀`)` | 31 | gt |
| `(´▽`)` | 22 | gt |
| `(✿◕‿◕)` | 16 | gt |
| `(｡・ω・｡)` | 15 | gt |
| `(｡◕‿◕｡)` | 14 | gt |
| `(‿‿‿)` | 14 | gt |
| `(｡♥‿♥｡)` | 12 | gt |

### HN-D
| kaomoji | count | source |
|---|---:|---|
| `(ง・̀_・́)` | 24 | pred |
| `(╯°□°)` | 15 | gt |
| `(눈_눈)` | 9 | pred |
| `(￣ヘ￣;)` | 6 | pred |
| `(╭ರ_・́)` | 6 | pred |
| `(งᵒ̌皿ᵒ̌)` | 4 | pred |
| `(;￣д￣)` | 3 | gt |
| `(´°ω°`)` | 2 | pred |
| `(ಠ‿ಠ)` | 1 | pred |
| `(╭☞・́⍛・̀)` | 1 | pred |

### HN-S
| kaomoji | count | source |
|---|---:|---|
| `(・̀‿・́)` | 181 | pred |
| `(◕_◕)` | 63 | pred |
| `(°ロ°)` | 23 | gt |
| `(;´д`)` | 7 | gt |
| `(;・∀・)` | 6 | gt |
| `(⊙_⊙)` | 5 | pred |
| `(｡・́︿・̀｡)` | 5 | gt |
| `(・̀ㅂ・́)` | 4 | pred |
| `(╥﹏╥)` | 3 | gt |
| `(・﹏・)` | 3 | pred |

### LN
| kaomoji | count | source |
|---|---:|---|
| `(´・ω・`)` | 65 | gt |
| `(´｡・ω・｡`)` | 18 | gt |
| `(´・_・`)` | 13 | gt |
| `(´-`)` | 12 | gt |
| `(´-ω-`)` | 8 | gt |
| `(´;ω;`)` | 8 | gt |
| `(￣~￣;)` | 6 | pred |
| `(´ー`)` | 4 | pred |
| `(￣ω￣;)` | 4 | pred |
| `(ﾉωﾉ)` | 3 | pred |

### NB
| kaomoji | count | source |
|---|---:|---|
| `(・‿・)` | 156 | gt |
| `(｡・̀‿-)` | 71 | gt |
| `(◠‿◠)` | 53 | pred |
| `(・_・)` | 44 | gt |
| `(`・ω・´)` | 42 | gt |
| `(๑・̀ㅂ・́)` | 38 | pred |
| `(・ω・)` | 34 | gt |
| `(・∀・)` | 23 | gt |
| `(・_・;)` | 22 | gt |
| `(◕▽◕)` | 20 | pred |
