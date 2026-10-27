# Cross-size development-v2 residual-coverage audit

Status: **development diagnostic only**.  No untouched task was opened or run.

## 1. Why the old finite spiral was not a guarantee

For an Archimedean spiral

\[
r(\theta)=b\theta,\qquad b=\frac{h}{2\pi},
\]

adjacent turns are separated by pitch \(h\).  A sufficient ideal-plane
condition for a capture tube of radius \(r_{\rm cap}\) to cover every point in
a visual disk of radius \(B\) is

\[
h\le 2r_{\rm cap},\qquad R_{\max}\ge B+\frac{h}{2}.
\]

The second inequality supplies the missing outer half-turn.  The v2 contracts
instead used \(R_{\max}=B-r_{\rm cap}\).  Thus the word “full” in the stage
label described the declared finite fallback path, not complete disk coverage.

| size | \(B\) | \(r_{cap}\) | \(h\) | v2 \(R_{max}\) | sufficient \(R_{max}\) | v2 path/time | sufficient path/time |
|---|---:|---:|---:|---:|---:|---:|---:|
| M10 | 6.000 mm | 0.300 mm | 0.480 mm | 5.700 mm | 6.240 mm | 212.86 mm / 94.60 s | 255.06 mm / 113.36 s |
| M12 | 8.000 mm | 0.370 mm | 0.592 mm | 7.630 mm | 8.296 mm | 309.20 mm / 137.42 s | 365.50 mm / 162.44 s |

These are commanded-path times at 2.25 mm/s, before the conical probe, reset,
capture contingency and terminal reserve.  M12 therefore cannot execute even
the current finite path inside 130 s.  M10 cannot execute the sufficient-disk
path plus its 19.71 s terminal reserve and fixed phases inside 130 s either.
Tracking error, pre-hard-stop escape, a missing sensor handoff and terminal
rejection remain additional failure modes even when the ideal capture tube
intersects the true center.

## 2. What happened on the named v2 failures

The values below are offline diagnostics; true-center distances are never fed
back to the controller.

| trial | direction | shortcut | ideal nearest spiral distance | arc time to nearest point | executed fallback stage | diagnosis |
|---|---:|---:|---:|---:|---:|---|
| M10-006 | correct (3.84 deg control error) | 1.305 mm | 0.189 mm < 0.300 mm | 53.0 s | 42.44 s | safety escape ended the path before its ideal intercept |
| M10-010 | wrong (178.92 deg) | 0 | 0.508 mm > 0.300 mm | 87.8 s | 88.47 s | wrong direction and an uncovered outer annulus |
| M12-003 | correct (4.26 deg) | 0 | 0.197 mm < 0.370 mm | 119.2 s | 112.17 s | finite fallback stopped before the late intercept |
| M12-004 | correct (1.19 deg) | 0 | 0.472 mm > 0.370 mm | 131.7 s | 105.30 s | outer annulus is geometrically uncovered and intercept is beyond budget |
| M12-011 | correct (11.53 deg) | planned 2.500 mm, not executed | 0.552 mm > 0.370 mm | 121.4 s | 102.98 s | global-reserve deadlock rejected a useful shortcut, then ran an uncovered path |

The old scheduler required

\[
T_{shortcut}+T_{global}+T_{terminal}\le 130\;{\rm s}
\]

before allowing the shortcut.  On M12, \(T_{global}>130\) s by itself, so this
condition could never pass.  This was a scheduling contradiction, not evidence
that the shortcut was physically unsafe.

## 3. Frozen residual policy and its exact claim

The new development policy compares two **alternative** schedules:

\[
T_{res}=T_{shortcut}+T_{corridor}+T_{terminal},\qquad
T_{global}=T_{reset}+T_{spiral}+T_{terminal}.
\]

It selects the posterior corridor only when all of the following online gates
hold:

\[
q_{dir}\ge0.10,\qquad p_-\le0.35,\qquad T_{res}\le T_{remain}.
\]

No true center is read.  Let \(\hat e\) be the posterior direction and
\(\hat e_\perp\) its planar normal.  In coordinates

\[
s=(x-x_0)^\top\hat e,\qquad u=(x-x_0)^\top\hat e_\perp,
\]

the serpentine reference covers

\[
s\in[L_*,B],\qquad |u|\le W,\qquad W=B\sin 12^\circ.
\]

The frozen lane spacing is

\[
h_{lane}=2(r_{cap}-\epsilon_d),\qquad \epsilon_d=0.15\ {\rm mm}.
\]

Here \(\epsilon_d\) is a **development engineering allowance**, not an
empirically validated tracking bound or confidence interval.  If the actual
tip-reference error obeys \(\|e_{xy}(t)\|\le\epsilon_d\) throughout the relevant
lane and the center satisfies

\[
r\cos\delta\in[L_*,B],\qquad |r\sin\delta|\le W,
\]

then the triangle inequality gives

\[
d(x_{tip},x_c)\le \frac{h_{lane}}{2}+\epsilon_d=r_{cap}.
\]

This is the only geometric coverage statement.  It is conditional on the
posterior sector containing the center, \(L_*\) not passing the center, the
tracking bound holding, and the complete relevant lane being executed.  It
does not guarantee that the direction is correct, the online contact cue fires,
the chamfer capture remains stable, or the terminal verifier accepts.

The final frozen controller traverses the corridor at 1.5 mm/s.  A
measured-tip outer servo uses only robot state to reduce reference error, with
gain 1.5, 1.0 mm correction saturation, 2.0 mm/s correction slew limit, and a
1.2 mm / 0.15 s tracking-abort condition.  It neither reads nor estimates the
true center.  Raw hard-wrench thresholds and the 130 s active-time budget are
unchanged.

The conservative scheduler uses \(L_*=0\), so any accepted shortcut only
reduces the corridor length.  Its frozen bounds are:

| size | lane spacing | lanes | \(L_*=0\) polyline | speed | fingerprint + zero shortcut + transition/corridor + terminal |
|---|---:|---:|---:|---:|---:|
| M10 | 0.300 mm | 11 | 86.970 mm | 1.5 mm/s | 82.926 s |
| M12 | 0.440 mm | 9 | 96.633 mm | 1.5 mm/s | 95.092 s |

If the conditional branch misses, any remaining global outward motion is
explicitly named **best effort**; it is never reported as completed coverage.
The sensor-only capture handoff and terminal verifier remain authoritative.

## 4. Development evidence ladder

Every result below reuses named development tasks.  No untouched task was read
or run, and none of these rows is confirmatory evidence.

| policy rung | named result | observation |
|---|---|---|
| Initial 7-lane corridor, low-force search | <code>cross_size_s4_residual_corridor_m12_development</code> | M12-003/004/011 all exhausted the corridor and then the 130 s common budget; 0/3 bottom, 0 hard stops. |
| Nominal-controller restore, 0.44 mm lanes, no outer servo, 2.25 mm/s | <code>cross_size_s4_residual_corridor_m12_development_v2</code> | M12-003 and 004 reached CAD bottom in 16.558 s and 13.990 s. M12-011 had a corridor soft escape and later exhausted search at 122.228 s; its reference-error P99 was 0.627 mm. |
| Add measured-tip outer servo at 2.25 mm/s | <code>cross_size_s4_residual_corridor_m12_011_development_v3</code> | M12-011 P99 improved to 0.285 mm, but the corridor soft-escaped after 2.021 s. Raw physical torque reached 41.363 mNm, above the unchanged 40 mNm soft threshold and below the 60 mNm hard threshold. |
| Frozen servo, reduce corridor speed to 1.5 mm/s and correction slew to 2.0 mm/s | <code>cross_size_s4_residual_corridor_m12_003_004_development_v4b</code> and <code>cross_size_s4_residual_corridor_m12_011_development_v4</code> | M12-003/004/011 reached CAD bottom in 17.597/15.632/22.872 s; 3/3 used sensor capture, 0 used global fallback, and 0 hit hard stops. |

The final M12 reference-error P99 values were 0.201, 0.180 and 0.321 mm,
respectively.  All exceed the 0.150 mm design allowance.  Therefore the 3/3
development result supports the physical usefulness of the slower tracked
corridor plus chamfer/sensor handoff, but it **does not validate** the
conditional capture-tube guarantee.  The capture radius is an analytic
sufficient-radius surrogate, not an exact boundary of the chamfer's physical
basin.

The M12-011 v4 contract predates the later commissioning-snapshot addition,
whereas the M12-003/004 v4b contract records source hashes and its commissioning
snapshot.  The combined 3/3 result is consequently regression evidence for the
frozen development combination, not a clean single-factor causal ablation.

A separate named M10 development check also keeps search success distinct from
full-loop success:

- M10-006 obtained a corridor sensor handoff without a hard stop, but the
  terminal dwell verifier rejected all attempts and the run ended
  <code>funnel_safe_abort</code>; fixing search did not fix terminal insertion.
- M10-010 had near-zero direction quality, so the corridor gate correctly
  rejected the posterior branch.  The finite global fallback then ended
  <code>search_radius_exhausted</code>; the reliability gate prevents an
  unjustified shortcut but cannot make the fallback complete.

The associated records are under
<code>outputs/current/diagnostics/cross_size_s4_residual_corridor_m10_development_v2</code>.

## 5. Code and evidence boundary

- `HybridResidualCoverageConfig` and
  `posterior_directional_corridor_path` live in
  `src/render_m8_search_comparison.py`; historical callers remain unchanged
  because the configuration defaults to disabled.
- `src/run_cross_size_s4_full_loop_evaluation.py` explicitly freezes the S4
  development/next-evaluation configuration and records the alternative
  schedule rule in the task contract.
- `tests/test_hybrid_residual_coverage.py` checks the sector capture-tube
  geometry, frozen development parameters and both size schedule bounds.
- Thresholds are development choices.  They must be frozen before any new
  untouched evaluation; they do not inherit confirmation from v2 or from the
  targeted v4 development reruns.
