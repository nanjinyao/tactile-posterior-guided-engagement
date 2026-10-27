# Cross-size S4b untouched: slide notes

All estimator variants use Proposed+ hybrid. Pure outward is the nonlearned
baseline and straight-to-center is a privileged execution diagnostic. Every
method shares size-commissioned admittance/R3, unchanged raw hard thresholds,
a 130 s budget, and zero extension. Offline true-center/contact-tag scoring is
not available to the online controller.

## M10

- M8 zero-shot + P+ hybrid: 12/12 (100.0%), hard=0, executed>90deg=0, latent>90deg=0.
- S4b calibration-only + P+ hybrid: 12/12 (100.0%), hard=0, executed>90deg=0, latent>90deg=0.
- S4b base-anchored residual FT + P+ hybrid: 10/12 (83.3%), hard=1, executed>90deg=0, latent>90deg=0.
- pure outward: 11/12 (91.7%), hard=0, executed>90deg=0, latent>90deg=0.
- privileged straight-to-center: 12/12 (100.0%), hard=0, executed>90deg=0, latent>90deg=0.
- Interpretation: `S4_fine_tuning_not_sufficient_for_frozen_gate`.

## M12

- M8 zero-shot + P+ hybrid: 10/12 (83.3%), hard=0, executed>90deg=0, latent>90deg=0.
- S4b calibration-only + P+ hybrid: 11/12 (91.7%), hard=0, executed>90deg=0, latent>90deg=0.
- S4b base-anchored residual FT + P+ hybrid: 10/12 (83.3%), hard=0, executed>90deg=0, latent>90deg=0.
- pure outward: 8/12 (66.7%), hard=0, executed>90deg=0, latent>90deg=0.
- privileged straight-to-center: 12/12 (100.0%), hard=0, executed>90deg=0, latent>90deg=0.
- Interpretation: `fine_tuned_passed_but_necessity_not_identified`.

The untouched set is forbidden for tuning and becomes permanently test-only after readout.
