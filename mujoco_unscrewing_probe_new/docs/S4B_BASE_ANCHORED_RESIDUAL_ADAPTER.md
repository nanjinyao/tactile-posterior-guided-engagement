# S4b base-anchored residual adapter

## Scope and claim boundary

This component is a development train + head-training-held-out-validation
safety adapter around two already
frozen estimators: the M8 base ensemble and the size-specific S4b fine-tuned
(FT) ensemble. It uses both estimates from the **same four-channel conical
fingerprint**. The selector reads only the S4b `train` and `validation` roles.
It reads zero S4b calibration examples, zero full-loop pose labels and zero
untouched rollouts.

The development full-loop counterexamples motivated the architecture only:
fine-tuning must not be allowed to create a longer shortcut merely by becoming
more confident. They did not select any numerical parameter. The S4b
validation split is held out from FT head-weight fitting, but it is used to
select this adapter and is therefore **not** a new test set. This stage proves
only train/validation posterior and shortcut-admissibility diagnostics. It
does not prove a full-loop improvement.

## Base-anchored posterior and vector shrinkage

Let $p_B$ and $p_F$ be the base and FT class posteriors from one fingerprint.
For both contact state and direction, the adapted posterior is

$$
p_A = p_B + \alpha(p_F-p_B)=(1-\alpha)p_B+\alpha p_F,
\qquad 0<\alpha<1.
$$

Thus neither parent can be silently removed. For circular direction class
angles $\theta_k$, define $z(p)=\sum_k p_k e^{i\theta_k}$. Linearity gives

$$
z_A=z_B+\alpha(z_F-z_B),
\qquad
\hat\theta_A=\arg z_A.
$$

This is an affine residual adapter in posterior space and exactly the same
shrinkage in the first circular moment used for control.

## Conservative shortcut envelope

The adapted direction is not allowed to imply more shortcut confidence than
either frozen parent. The runtime reports

$$
\rho_{\rm exec}=\min\{|z_B|,|z_F|,|z_A|\},
$$

$$
\Delta_{\rm exec}=\max\{\Delta_B,\Delta_F,
d_{S^1}(\hat\theta_B,\hat\theta_F)\},
$$

and uses the union of the two frozen APS class sets. Its scalar maximum-class
confidence is also the minimum of base, FT and adapted confidence. Therefore
the adapter can rotate the estimate toward FT, but it cannot make the shortcut
longer by reporting a confidence or ensemble span that is more optimistic than
both parents.

The deployable abstention gate is

$$
g=\mathbf 1\left[|z_F|\ge \rho_{\min}\right]
\mathbf 1\left[d_{S^1}(\hat\theta_B,\hat\theta_F)\le90^\circ\right].
$$

The $90^\circ$ limit is fixed geometrically: estimates in opposite open
half-planes do not justify directed transport. If $g=0$, the runtime sets
$\rho_{\rm exec}=0$ and $\Delta_{\rm exec}=180^\circ$. In the current planner,
that makes shortcut quality zero and preserves the finite-budget search. It is
an abstention, not a claim that the direction was corrected.

## Development train + head-training-held-out-validation selection rule

Candidates are a fixed grid with $0.50\le\alpha\le0.98$ and
$0\le\rho_{\min}\le0.80$; neither $\alpha=0$ nor $\alpha=1$ is legal. For every
M10/M12 `train` and `validation` stratum, a candidate must satisfy:

1. accepted control-direction errors above $90^\circ$: zero;
2. raw $>90^\circ$ count: no worse than the frozen FT estimator;
3. state accuracy: no worse than the frozen FT estimator;
4. overall gate coverage: at least 0.80;
5. each observed contact-state coverage: at least 0.70.

The first feasible pair in ascending $\alpha$, then ascending
$\rho_{\min}$, is selected. This is the maximum safe shrinkage toward the base,
followed by the least abstention compatible with the frozen constraints. The
shared M10/M12 result is

$$
\boxed{\alpha=0.92,\quad 1-\alpha=0.08,\quad
\rho_{\min}=0.60,\quad \delta_{\max}=90^\circ.}
$$

## Head-training-held-out validation diagnostics

| Size | Role | Raw state acc. | Raw control med./p90 | Raw $>90^\circ$ | Gate coverage | Accepted p90 | Accepted $>90^\circ$ |
|---|---|---:|---:|---:|---:|---:|---:|
| M10 | train | 0.889 | 4.61/13.79 deg | 0 | 0.931 | 11.27 deg | 0 |
| M10 | validation | 1.000 | 4.00/15.44 deg | 1 | 0.972 | 12.01 deg | 0 |
| M12 | train | 0.931 | 4.91/14.06 deg | 1 | 0.819 | 9.61 deg | 0 |
| M12 | validation | 0.861 | 3.59/19.58 deg | 0 | 0.861 | 15.37 deg | 0 |

The validation rows above were unseen by FT gradient updates, but were used for
adapter selection. They must not be presented as untouched confirmatory
evidence.

The single M10 validation catastrophe is
`s4bva_m10_double_overhang_r5_p225_s0`. Its true direction is $202.5^\circ$,
whereas base and FT means are approximately $351.6^\circ$ and $358.1^\circ$.
Because they agree on the wrong region, convex shrinkage cannot honestly flip
the estimate. FT concentration is only 0.5963, below the selected 0.60 floor,
so the adapter reports zero shortcut quality. The raw outlier remains one; the
**accepted directed-shortcut** outlier count becomes zero.

## Auditable artifacts

- implementation: `src/cross_size_s4b_base_anchored_adapter.py`;
- frozen runtime contract: `outputs/current/models/cross_size_s4b_base_anchored_adapter/adapter_contract.json`;
- complete candidate/metric report: `outputs/current/models/cross_size_s4b_base_anchored_adapter/selection_report.json`;
- concise selection handoff: `outputs/current/models/cross_size_s4b_base_anchored_adapter/SELECTION.md`;
- concise table: `outputs/current/models/cross_size_s4b_base_anchored_adapter/summary.md`;
- tests: `tests/test_cross_size_s4b_base_anchored_adapter.py`.

This component itself does not launch full-loop evaluation. The adapter was
subsequently frozen into two separately hashed runner contracts:

- development-v5: M10 fine-tuned 12/12 with zero hard stops; M12 10/12 with
  zero hard stops; both registered gates pass;
- untouched-v1: M10 fine-tuned 10/12 with one primary hard stop and therefore
  fails; M12 fine-tuned 10/12 with zero hard stops and passes.

Untouched M10 zero-shot/calibration-only are both 12/12, and untouched M12
calibration-only is 11/12. The adapter therefore does not establish
fine-tuning necessity or monotonic task improvement. Untouched-v1 is retired
as test-only evidence and is forbidden for later adapter/controller tuning.

Artifacts:

- `outputs/current/diagnostics/cross_size_s4b_full_loop_development_v5/`;
- `outputs/current/evaluations/cross_size_s4b_full_loop_untouched_v1/`.
