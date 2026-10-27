# Cross-size S4 development: slide notes

All estimator variants use Proposed+ hybrid. Pure outward is the nonlearned
baseline and straight-to-center is a privileged execution diagnostic. Every
method shares size-commissioned admittance/R3, unchanged raw hard thresholds,
a 130 s budget, and zero extension. Offline true-center/contact-tag scoring is
not available to the online controller.

## M10

- M8 zero-shot + P+ hybrid: 7/12 (58.3%), hard=2, >90deg=1.
- S4 calibration-only + P+ hybrid: 9/12 (75.0%), hard=1, >90deg=1.
- S4 state/direction-head FT + P+ hybrid: 9/12 (75.0%), hard=1, >90deg=1.
- pure outward: 7/12 (58.3%), hard=3, >90deg=0.
- privileged straight-to-center: 7/12 (58.3%), hard=4, >90deg=0.
- Interpretation: `development_diagnostic_only_no_confirmatory_claim`.

## M12

- M8 zero-shot + P+ hybrid: 9/12 (75.0%), hard=1, >90deg=0.
- S4 calibration-only + P+ hybrid: 9/12 (75.0%), hard=0, >90deg=0.
- S4 state/direction-head FT + P+ hybrid: 9/12 (75.0%), hard=0, >90deg=0.
- pure outward: 8/12 (66.7%), hard=3, >90deg=0.
- privileged straight-to-center: 11/12 (91.7%), hard=0, >90deg=0.
- Interpretation: `development_diagnostic_only_no_confirmatory_claim`.

This phase permits tuning and is diagnostic, never confirmatory.
