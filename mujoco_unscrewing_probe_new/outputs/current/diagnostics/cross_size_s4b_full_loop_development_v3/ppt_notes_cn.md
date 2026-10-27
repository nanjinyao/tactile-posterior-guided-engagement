# Cross-size S4b development：PPT 注释

所有 estimator 变体统一使用 Proposed+ hybrid；纯 outward 是非学习基线，
straight-to-center 是读取真中心的 privileged 执行诊断。所有方法共享尺寸
commissioning 后的 admittance/R3 漏斗、未放宽的 raw hard threshold、
130 s 与零预算扩展。严格成功中的离线 true-center/contact-tag 审计不可用于在线控制。

## M10

- M8 zero-shot + P+ hybrid：12/12 (100.0%), hard=0, >90deg=1。
- S4b calibration-only + P+ hybrid：11/12 (91.7%), hard=0, >90deg=1。
- S4b state/direction-head FT + P+ hybrid：9/12 (75.0%), hard=2, >90deg=1。
- pure outward：9/12 (75.0%), hard=1, >90deg=0。
- privileged straight-to-center：12/12 (100.0%), hard=0, >90deg=0。
- 结论：`development_diagnostic_only_no_confirmatory_claim`。

## M12

- M8 zero-shot + P+ hybrid：10/12 (83.3%), hard=2, >90deg=0。
- S4b calibration-only + P+ hybrid：10/12 (83.3%), hard=0, >90deg=0。
- S4b state/direction-head FT + P+ hybrid：11/12 (91.7%), hard=1, >90deg=0。
- pure outward：5/12 (41.7%), hard=4, >90deg=0。
- privileged straight-to-center：9/12 (75.0%), hard=1, >90deg=0。
- 结论：`development_diagnostic_only_no_confirmatory_claim`。

该阶段允许调参，只能作为诊断，不得写成确认性结论。
