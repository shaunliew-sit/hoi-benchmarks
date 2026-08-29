# Multi-model HOI comparison — stratified by proposal difficulty (s_ref)

HICO **grounding** + SWIG **referring (action)**, split EASY (`s_ref≥0.5`) / HARD (`s_ref<0.5`).
No re-eval — `stratify_all_tasks.py --models-config models_compare.json`. The two SAHA rows
(Qwen3-VL-8B+SFT, +SFT+GRPO) are the in-house trained models; all others are baselines.

> ⚠️ `claude_result/` and `gpt_result/` are byte-identical (same MD5) → Claude/GPT rows match
> until the misplaced copy is fixed. Claude/GPT headline 'AR 4.59' = AR@0.5; our AR = mean IoU 0.5–0.95.

## HICO-GROUNDING — AR / @.5 / @.75 / @s / @m / @l (%)

### ALL
| model | AR | @.5 | @.75 | @s | @m | @l |
|---|---|---|---|---|---|---|
| Qwen3-VL-4B | 9.28 | 24.40 | 5.44 | 0.03 | 0.54 | 13.66 |
| Qwen3-VL-8B | 30.38 | 50.40 | 30.62 | 3.76 | 19.21 | 37.88 |
| Qwen3-VL-8B+SFT | 30.04 | 52.94 | 29.50 | 3.39 | 18.66 | 37.62 |
| Qwen3-VL-8B+SFT+GRPO | 31.03 | 54.94 | 30.36 | 3.58 | 19.31 | 38.83 |
| Qwen3-VL-32B | 34.18 | 55.42 | 34.76 | 4.50 | 22.33 | 42.31 |
| InternVL3-8B | 26.95 | 49.18 | 25.78 | 0.88 | 7.81 | 31.62 |
| InternVL3-38B | 30.83 | 52.19 | 30.97 | 1.13 | 9.44 | 36.07 |
| Groma-7B | 3.55 | 5.55 | 3.76 | 0.00 | 0.93 | 5.00 |
| Groma-7B-HOI-FT | 8.46 | 12.99 | 8.79 | 0.10 | 1.10 | 12.31 |
| Claude-4.5-Opus | 0.97 | 4.61 | 0.10 | 0.00 | 0.06 | 1.43 |
| GPT-5.2 | 0.97 | 4.61 | 0.10 | 0.00 | 0.06 | 1.43 |

### EASY
| model | AR | @.5 | @.75 | @s | @m | @l |
|---|---|---|---|---|---|---|
| Qwen3-VL-4B | 11.91 | 31.19 | 7.02 | 0.08 | 0.69 | 15.75 |
| Qwen3-VL-8B | 37.93 | 61.46 | 38.75 | 8.76 | 25.31 | 43.06 |
| Qwen3-VL-8B+SFT | 37.86 | 65.75 | 37.53 | 9.78 | 25.98 | 42.72 |
| Qwen3-VL-8B+SFT+GRPO | 38.98 | 67.78 | 38.57 | 10.24 | 26.86 | 43.94 |
| Qwen3-VL-32B | 42.61 | 67.59 | 43.87 | 10.13 | 29.24 | 48.11 |
| InternVL3-8B | 33.52 | 60.10 | 32.48 | 7.12 | 22.15 | 38.15 |
| InternVL3-38B | 38.52 | 63.99 | 39.14 | 8.72 | 26.35 | 43.55 |
| Groma-7B | 4.49 | 6.96 | 4.78 | 0.00 | 1.16 | 5.72 |
| Groma-7B-HOI-FT | 10.82 | 16.52 | 11.29 | 0.15 | 1.46 | 14.18 |
| Claude-4.5-Opus | 1.21 | 5.77 | 0.13 | 0.00 | 0.07 | 1.62 |
| GPT-5.2 | 1.21 | 5.77 | 0.13 | 0.00 | 0.07 | 1.62 |

### HARD
| model | AR | @.5 | @.75 | @s | @m | @l |
|---|---|---|---|---|---|---|
| Qwen3-VL-4B | 0.51 | 1.69 | 0.18 | 0.01 | 0.19 | 1.06 |
| Qwen3-VL-8B | 5.16 | 13.42 | 3.45 | 1.23 | 4.50 | 8.01 |
| Qwen3-VL-8B+SFT | 3.90 | 10.13 | 2.65 | 0.67 | 2.58 | 6.92 |
| Qwen3-VL-8B+SFT+GRPO | 4.44 | 12.04 | 2.90 | 0.76 | 2.72 | 8.05 |
| Qwen3-VL-32B | 6.03 | 14.72 | 4.28 | 1.57 | 5.69 | 8.91 |
| InternVL3-8B | 4.98 | 12.68 | 3.39 | 1.00 | 4.29 | 7.90 |
| InternVL3-38B | 5.12 | 12.76 | 3.65 | 1.20 | 5.32 | 7.22 |
| Groma-7B | 0.42 | 0.85 | 0.35 | 0.00 | 0.33 | 0.73 |
| Groma-7B-HOI-FT | 0.57 | 1.18 | 0.42 | 0.08 | 0.13 | 1.22 |
| Claude-4.5-Opus | 0.16 | 0.75 | 0.00 | 0.00 | 0.04 | 0.35 |
| GPT-5.2 | 0.16 | 0.75 | 0.00 | 0.00 | 0.04 | 0.35 |

## SWIG-REFERRING — METEOR / BS-F1 (%)

### ALL
| model | METEOR | BS-F1 |
|---|---|---|
| Qwen3-VL-4B | 18.18 | 75.48 |
| Qwen3-VL-8B | 19.08 | 77.31 |
| Qwen3-VL-8B+SFT | 16.35 | 75.05 |
| Qwen3-VL-8B+SFT+GRPO | 19.36 | 77.90 |
| Qwen3-VL-32B | 19.98 | 78.11 |
| InternVL3-8B | 19.27 | 77.66 |
| InternVL3-38B | 20.54 | 78.59 |
| Groma-7B | 5.76 | 59.44 |
| Groma-7B-HOI-FT | 33.20 | 86.08 |
| Claude-4.5-Opus | 14.10 | 73.51 |
| GPT-5.2 | 14.10 | 73.51 |

### EASY
| model | METEOR | BS-F1 |
|---|---|---|
| Qwen3-VL-4B | 18.16 | 75.44 |
| Qwen3-VL-8B | 19.06 | 77.26 |
| Qwen3-VL-8B+SFT | 16.65 | 75.26 |
| Qwen3-VL-8B+SFT+GRPO | 19.74 | 78.19 |
| Qwen3-VL-32B | 19.99 | 78.06 |
| InternVL3-8B | 19.30 | 77.65 |
| InternVL3-38B | 20.62 | 78.61 |
| Groma-7B | 5.93 | 59.57 |
| Groma-7B-HOI-FT | 33.42 | 86.23 |
| Claude-4.5-Opus | 14.46 | 73.76 |
| GPT-5.2 | 14.46 | 73.76 |

### HARD
| model | METEOR | BS-F1 |
|---|---|---|
| Qwen3-VL-4B | 18.29 | 75.73 |
| Qwen3-VL-8B | 19.24 | 77.79 |
| Qwen3-VL-8B+SFT | 13.85 | 73.32 |
| Qwen3-VL-8B+SFT+GRPO | 16.17 | 75.47 |
| Qwen3-VL-32B | 19.93 | 78.49 |
| InternVL3-8B | 19.00 | 77.81 |
| InternVL3-38B | 19.87 | 78.39 |
| Groma-7B | 4.34 | 58.33 |
| Groma-7B-HOI-FT | 31.36 | 84.86 |
| Claude-4.5-Opus | 11.10 | 71.48 |
| GPT-5.2 | 11.10 | 71.48 |

