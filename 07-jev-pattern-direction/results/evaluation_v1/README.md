# Direct pattern-guided five-session forecast pilot

Retrospective exploratory evaluation; prompting, not fine-tuning.

| Arm | Accuracy | Macro-F1 | Brier ↓ | Log loss ↓ |
| --- | ---: | ---: | ---: | ---: |
| zero_shot | 37.33% | 0.339 | 0.862890 | 1.547088 |
| candles | 38.00% | 0.344 | 0.830781 | 1.408924 |
| formations | 36.67% | 0.314 | 0.875197 | 1.531091 |
| combined | 39.00% | 0.332 | 0.859144 | 1.480652 |
| training_prior | 36.67% | 0.179 | 0.665765 | 1.096851 |
| uniform | 36.67% | 0.179 | 0.666667 | 1.098612 |
| combined_blank | 27.00% | 0.142 | 1.105721 | 1.798869 |
| combined_control_subset | 39.00% | 0.318 | 0.837441 | 1.422893 |

Primary comparison: combined versus zero-shot, paired Brier. See metrics.json for intervals, single-guidance comparisons, image control and repeatability.

Exploratory, historical adaptive research. Four-week intervals count retained dates, carrying all tickers together; gaps span additional calendar time. No refit uncertainty/multiplicity correction or independent confirmation.

Continuation criteria: {"primary_combined_vs_zero_shot_ci_positive": false, "combined_brier_below_prior": false, "combined_point_brier_below_both_single_guidance_arms": false, "image_control_ci_positive": true}

No execution costs, turnover or trading P&L were evaluated.
