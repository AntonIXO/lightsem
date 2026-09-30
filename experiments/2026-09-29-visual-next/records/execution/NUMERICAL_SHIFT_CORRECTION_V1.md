# Numerical correction to circular-shift evaluation

During CPU rescoring of accepted baseline C, the original shift helper failed
before writing its first family result. For source interval [98.9,140), candidate
133.2 and offset6.8, binary arithmetic produced remainder41.09999999999998,
strictly below duration41.099999999999994. Adding98.9 rounded the final timestamp
to140, outside the half-open support. Six other combinations in this same source
reproduced the issue. This is a numerical boundary defect, not evidence about
model quality. Some first-family non-null scores had been computed in memory;
no family outcome was serialized or used to choose this correction.

The new local visual-next shift helper keeps the original offsets, modulo,
confidence arrays, candidate counts and source intervals. It clips only numeric
overshoot to the closest representable timestamp within [start,end), using
nextafter(end,start). Corrections larger than two floating-point spacings raise
an error. Every correction is counted in the per-family report. No interval is
expanded, no family replaced, no rate/tolerance/threshold changed and no shift
redrawn. Non-boundary outputs are verified identical to the historical helper.
The inherited schema description now accurately says full64s rather than the
historical helper's center8 wording; supplied intervals were already full64s.

Five focused tests pass, including the real-timestamp reproduction, exact
non-boundary equivalence, empty support and event-recovery/null-support tests.
Keep original historical code and results unchanged. V1/V2 baseline rescore
attempts and archived original scorer are retained. Freeze V3 before continuing
CPU rescoring; use the same corrected instrument for every future H3 arm and
control. The existing27/30 shift-support failure remains failed. No model was
retrained or selected in response to this correction.
