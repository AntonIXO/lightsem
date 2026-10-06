# V3: preserve structural semantics through the fixture readout

The V2 study passed its narrow crossing/sensitivity gates but a new, explicit
relationship check exposed an independent failure. During bone-preserving hand
adduction, source wrist distance decreased from 1.441 m to 0.086 m. V2 method 08's
internal bridge contracted, while desired fixture targets separated from
1.101 m to 4.090 m. Actual finite rays separated from 0.962 m to 1.850 m.
This is a readout inversion, not a representation-collapse claim. All V2
artifacts and computed engineering verdicts remain preserved and scoped.

V3 introduces an optional absolute structural field, a fixed reference, and
explicit shared anatomical groups. Each such group receives ONE fixed spatial
rotation/translation plus a uniform fit scale. Individual fixture offsets cannot
break its relative geometry. Extra outputs sample the same group without
changing existing fixture identities or source timing. Garland fields sample
the same structural coordinates so contact centers also converge.

Method 03 now exports absolute expanded FK points. The complete 22-point
structure is preserved up to a similarity transform before mechanical fitting.
Method 08 replaces overshooting neighbor averages with instantaneous graph
consensus, `(I + L(w)) Q = P`, with `w = c / (2 max(1-c, 1e-5))`. Its wrists,
head and torso share an absolute spatial mapping. Other thirteen methods retain
their previous default output behavior; their rerun is not independent evidence.

The corrected controlled clap reduces emitted finite-ray separation from
1.208 m to 0.000302 m and restores it on release. Frozen poses, permutation,
fixture expansion, body bone lengths, and shared shape geometry pass regression
checks. These tests were observed during implementation and are admission
evidence, not a blinded scientific confirmation.

Shared bounding can vary uniform scale with the source pose; it does not preserve
absolute stage size. Large genuine ray changes remain for mechanical fitting.
The largest inspected method-03 Shpongle motor-pan step was 81.1 degrees, but
its actual ray step was 12.4 degrees near a tilt-coordinate pole. Both motor
coordinate error and true 3D ray error are therefore reported. Enumerating valid
Euler branches did not provide an exact lower-step alternative; the existing
compiler remains unchanged.

Final V3 artifacts use
`/mnt/lightsem-data/experiments/lightsem/body_transfer_20261004_v3`.
The previous numerical study and six schematic videos remain at the V2 root.
The primary crossing gate stays at a 0.10 absolute reduction and the source
sensitivity gate is unchanged. No failures are concealed by lowering thresholds,
altering bodies, changing P2, or adding autonomous motion.
