# Parent check of reported lean-video caption loss

The independent audit reported intermittent missing/partial captions, including frame2404. Parent inspection of its saved `pair_002.png` and `stack_002.png` shows BOTH complete labels at that frame.

A CPU check reconstructed the exact OpenCV caption glyph masks and tested all45 decoded native sample pairs. Every expected glyph pixel above128 has all decoded channels above80: minimum recall **1.0 on both sides**. No caption-loss evidence was found in this sample set. This is a thresholded sampled-pixel check, not OCR or proof for all8319frames.

Therefore the reported caption limitation is **not corroborated** and must not be repeated as an established video defect. The original report/observer remains intact. Evidence: `stage_lean_visual_audit_01/parent_caption_check.json`; code `outputs/check_lean_caption_claim.py`. Numeric lean failure and bounded shaft observations are unchanged.
