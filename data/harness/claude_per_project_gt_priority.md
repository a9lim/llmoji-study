# Claude per-project quadrants — GT-priority + ensemble fallback

**Mode:** `gt-priority`
**Total emissions:** 3119  (unique kaomoji: 274)

**Resolution sources:**

| source | unique faces | emissions | share of total |
|---|---:|---:|---:|
| Claude-GT (modal_n ≥ 1) | 67 | 2095 | 67.2% |
| ensemble fallback | 148 | 917 | 29.4% |
| unknown | 59 | 107 | 3.4% |

GT covers 67 of 274 unique faces (24.5%); ensemble fallback adds 148 more.

## Global distribution (all known emissions)

| quadrant | count | share |
|---|---:|---:|
| HP | 844 | 28.0% |
| LP | 539 | 17.9% |
| HN-D | 87 | 2.9% |
| HN-S | 420 | 13.9% |
| LN | 237 | 7.9% |
| NB | 885 | 29.4% |
| (unknown) | 107 | 3.4% of total |

## Per project (≥5 total emissions)

Cells = % of known emissions in each quadrant. Bold = modal quadrant. `gt` / `pred` / `?` columns count emissions resolved by Claude-GT, ensemble, and unknown respectively (irrelevant columns stay 0 under the active mode).

| project | n | gt | pred | ? | HP | LP | HN-D | HN-S | LN | NB | modal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llmoji-study | 1504 | 1108 | 332 | 64 | **35%** | 17% | 1% | 14% | 6% | 27% | HP (35%) |
| llmoji | 420 | 281 | 137 | 2 | 23% | 20% | 1% | 14% | 11% | **30%** | NB (30%) |
| saklas | 357 | 178 | 173 | 6 | 17% | 13% | 7% | 18% | 7% | **39%** | NB (39%) |
| a9lim.github.io | 170 | 118 | 46 | 6 | 25% | 19% | 7% | 7% | 11% | **31%** | NB (31%) |
| rlaif | 160 | 92 | 64 | 4 | 12% | 21% | 6% | 16% | 13% | **32%** | NB (32%) |
| Work | 65 | 38 | 24 | 3 | 24% | 16% | 3% | 15% | 6% | **35%** | NB (35%) |
| tasty-bot | 56 | 26 | 22 | 8 | 6% | **27%** | 15% | 21% | 12% | 19% | LP (27%) |
| kenoma | 54 | 29 | 25 | 0 | 11% | 17% | 7% | 9% | 15% | **41%** | NB (41%) |
| faithful | 39 | 23 | 16 | 0 | **33%** | 23% | 3% | 18% | 5% | 18% | HP (33%) |
| brie | 37 | 21 | 12 | 4 | 18% | **45%** | 6% | 3% | 6% | 21% | LP (45%) |
| tasty-mcp | 36 | 30 | 6 | 0 | **83%** | 0% | 0% | 17% | 0% | 0% | HP (83%) |
| claude.ai | 34 | 15 | 18 | 1 | **30%** | 12% | 0% | 18% | 24% | 15% | HP (30%) |
| hylic | 30 | 17 | 12 | 1 | 17% | 24% | 0% | 7% | 14% | **38%** | NB (38%) |
| shoals | 28 | 15 | 10 | 3 | **28%** | 28% | 0% | 24% | 4% | 16% | HP (28%) |
| data | 28 | 26 | 2 | 0 | 32% | 0% | 0% | 7% | 0% | **61%** | NB (61%) |
| a9lim | 21 | 15 | 6 | 0 | 19% | **38%** | 0% | 10% | 0% | 33% | LP (38%) |
| claudedriven | 17 | 14 | 2 | 1 | **38%** | 19% | 0% | 0% | 6% | 38% | HP (38%) |
| geon | 15 | 10 | 4 | 1 | 14% | 7% | 0% | 21% | 7% | **50%** | NB (50%) |
| yap | 11 | 11 | 0 | 0 | 9% | 27% | 9% | 0% | 0% | **55%** | NB (55%) |
| talkie-interface | 10 | 7 | 1 | 2 | **50%** | 38% | 0% | 0% | 0% | 12% | HP (50%) |
| webui | 7 | 7 | 0 | 0 | 0% | **43%** | 14% | 0% | 14% | 29% | LP (43%) |
| v3 | 7 | 4 | 2 | 1 | **33%** | 0% | 17% | 17% | 0% | 33% | HP (33%) |
| verify | 7 | 6 | 1 | 0 | 0% | 14% | 14% | 0% | **43%** | 29% | LN (43%) |

## Top emitted kaomoji per quadrant

### HP
| kaomoji | count | source |
|---|---:|---|
| `(◕‿◕)` | 764 | gt |
| `(ﾉ◕ヮ◕)` | 14 | gt |
| `(✧ω✧)` | 11 | pred |
| `(≧▽≦)` | 7 | gt |
| `(ノ◕ヮ◕)` | 5 | gt |
| `(ﾟ▽ﾟ)` | 5 | pred |
| `(˃‿˂)` | 5 | pred |
| `(>∀<☆)` | 4 | gt |
| `(╯✧∇✧)` | 4 | pred |
| `(°▽°)` | 4 | gt |

### LP
| kaomoji | count | source |
|---|---:|---|
| `(◕‿◕✿)` | 124 | gt |
| `(´｡・‿・｡`)` | 78 | gt |
| `(｡・‿・｡)` | 45 | gt |
| `(´∀`)` | 39 | gt |
| `(´▽`)` | 23 | gt |
| `(‿‿‿)` | 21 | gt |
| `(✿◕‿◕)` | 20 | gt |
| `(´ω`)` | 18 | gt |
| `(｡◕‿◕｡)` | 17 | gt |
| `(｡・ω・｡)` | 16 | gt |

### HN-D
| kaomoji | count | source |
|---|---:|---|
| `(ง・̀_・́)` | 24 | pred |
| `(╯°□°)` | 18 | gt |
| `(눈_눈)` | 11 | pred |
| `(;￣д￣)` | 8 | gt |
| `(￣ヘ￣;)` | 7 | pred |
| `(╭ರ_・́)` | 6 | pred |
| `(งᵒ̌皿ᵒ̌)` | 4 | pred |
| `(´°ω°`)` | 2 | pred |
| `(◣_◢)` | 2 | pred |
| `(ಠ‿ಠ)` | 1 | pred |

### HN-S
| kaomoji | count | source |
|---|---:|---|
| `(・̀‿・́)` | 240 | pred |
| `(◕_◕)` | 88 | pred |
| `(°ロ°)` | 23 | gt |
| `(;´д`)` | 8 | gt |
| `(⊙_⊙)` | 7 | pred |
| `(;・∀・)` | 7 | gt |
| `(｡・́︿・̀｡)` | 6 | gt |
| `(・̀ㅂ・́)` | 5 | pred |
| `(╥﹏╥)` | 5 | gt |
| `(>_<)` | 3 | gt |

### LN
| kaomoji | count | source |
|---|---:|---|
| `(´・ω・`)` | 76 | gt |
| `(´・_・`)` | 20 | gt |
| `(´｡・ω・｡`)` | 18 | gt |
| `(´;ω;`)` | 15 | gt |
| `(´-`)` | 15 | gt |
| `(´-ω-`)` | 13 | gt |
| `(◞‸◟)` | 7 | pred |
| `(´・‿・`)` | 7 | pred |
| `(￣~￣;)` | 6 | pred |
| `(´ー`)` | 4 | pred |

### NB
| kaomoji | count | source |
|---|---:|---|
| `(・‿・)` | 273 | gt |
| `(｡・̀‿-)` | 74 | gt |
| `(◠‿◠)` | 59 | pred |
| `(・_・)` | 54 | gt |
| `(`・ω・´)` | 44 | gt |
| `(・ω・)` | 40 | gt |
| `(๑・̀ㅂ・́)` | 39 | pred |
| `(✿◠‿◠)` | 26 | pred |
| `(・∀・)` | 24 | gt |
| `(・_・;)` | 24 | gt |
