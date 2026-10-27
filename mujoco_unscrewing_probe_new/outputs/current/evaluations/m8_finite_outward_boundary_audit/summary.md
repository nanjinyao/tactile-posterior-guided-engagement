# M8 Finite Outward Boundary Audit

## Result

- Adjacent-turn pitch bound: **satisfied** (0.4000 <= 0.5000 mm/rev).
- Finite path endpoint radius: 4.645304 mm = R - r_eff.
- Endpoint-aligned boundary samples beyond r_eff: **65535/65536** (99.998474%).
- Boundary nearest-distance min / median / worst: 0.250000 / 0.445059 / 0.642053 mm, versus r_eff=0.250000 mm.
- Full outer boundary within r_eff: **False**.
- Complete visual-disk coverage certificate: **False**.

## Interpretation

The pitch bound controls spacing between adjacent turns, but the finite spiral has no padded outer closing turn. Its r_eff tube therefore does not cover the outer boundary of the declared visual-support disk.

## Claim boundary

The current strict Outward method remains an empirical search comparator. The adjacent-turn pitch bound is satisfied, but this finite-path audit is not a complete-domain coverage certificate and does not certify capture, insertion, finite-time success, or hardware behavior.

## Change boundary

This is a read-only derived audit: policy changed=False, trial files changed=False, simulation rerun=False.
