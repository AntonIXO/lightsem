# Verified frozen contact-sheet construction

Source: pose_stage_01/frozen/render_pose_stage_study.py, `review_sheets` (renderer SHA d2a1a79047ac13ac714600e4f56cf9566f5b0f94ba020d7d5f573ffd7fb61830).

Each decoded1920x360 RGB composite is resized via `image.resize((960, 180))`; no resample argument is supplied (Pillow default, not a declared nearest-neighbour operation). Two columns and six rows, 12 samples/page, 22px text header per row, total sheet1920x1212. Unfilled cells retain gray padding. The attachment tool further displayed these sheets at1200x758 JPEG quality82; that is a separate display transform. Nativefast stacked640x1080 inspections avoid these spatial resizes, though attachment encoding may still be lossy.
