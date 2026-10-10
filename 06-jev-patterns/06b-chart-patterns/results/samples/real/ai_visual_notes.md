# AI visual inspection of the frozen real-data grids

Inspected all seven grids (38 distinct selected examples) before model inference. This is an AI-assisted qualitative inspection, not a blinded expert annotation study. User review is pending. No inspected examples were replaced and no labels or thresholds were changed based on appearance.

- **Head and Shoulders:** TXN 2006-11-03 has the most legible three-peak structure and terminal decline. IWM's left shoulder is a broad plateau. XLF, BA and PFE have uneven or locally fragmented shoulders. CSCO has a visible central head but a broad left side and gap-driven exit.
- **Inverse Head and Shoulders:** both examples have a central trough and upward exit; PG's shoulders are uneven and DIS has several competing local turns. Their main pivots need careful review.
- **Double Top:** NKE has two broad elevated regions, but the intervening trough is shallow at grid scale and the final gap dominates. Treat the geometry as subtle.
- **Double Bottom:** JNJ has a W-like base followed by a strong rise, with several minor turns competing for the first trough. Review which pivots define the label.
- **Triangle:** all four examples are visually difficult. The sloped and irregular sequences can look like broad trends or repeated swings without overlaid detector boundaries. Detector agreement alone does not make these clear triangles.
- **Bull Flag:** the selected examples generally show an upward move and a smaller pause before continuation. Some consolidations look rounded, sideways or brief relative to the full window, particularly the gap-heavy NVDA, ORCL and PLTR examples. “Uptrend” alone is insufficient evidence of a flag.
- **None:** examples include declines, gapped trends, choppy ranges and reversals. QQQ and some other rows contain local peak/trough structures; absence means neither frozen detector accepted a completed target event in the screen, not absence of every plausible hand-drawn formation.

Asterisks are algorithmic warnings generated before this inspection. No numerical visual-quality rate is inferred from them. Use the full-resolution source PNGs and event coordinates in real_selection_details.json when adjudicating unclear geometry. A reviewed successor dataset should preserve these initial labels and document any changes rather than overwriting this freeze.
