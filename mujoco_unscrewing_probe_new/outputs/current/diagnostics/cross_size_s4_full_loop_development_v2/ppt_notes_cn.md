# Cross-size S4 development：PPT 注释

所有 estimator 变体统一使用 Proposed+ hybrid；纯 outward 是非学习基线，
straight-to-center 是读取真中心的 privileged 执行诊断。所有方法共享尺寸
commissioning 后的 admittance/R3 漏斗、未放宽的 raw hard threshold、
130 s 与零预算扩展。严格成功中的离线 true-center/contact-tag 审计不可用于在线控制。

## M10

- M8 zero-shot + P+ hybrid：7/12 (58.3%), hard=2, >90deg=1。
- S4 calibration-only + P+ hybrid：9/12 (75.0%), hard=1, >90deg=1。
- S4 state/direction-head FT + P+ hybrid：9/12 (75.0%), hard=1, >90deg=1。
- pure outward：7/12 (58.3%), hard=3, >90deg=0。
- privileged straight-to-center：7/12 (58.3%), hard=4, >90deg=0。
- 结论：`development_diagnostic_only_no_confirmatory_claim`。

## M12

- M8 zero-shot + P+ hybrid：9/12 (75.0%), hard=1, >90deg=0。
- S4 calibration-only + P+ hybrid：9/12 (75.0%), hard=0, >90deg=0。
- S4 state/direction-head FT + P+ hybrid：9/12 (75.0%), hard=0, >90deg=0。
- pure outward：8/12 (66.7%), hard=3, >90deg=0。
- privileged straight-to-center：11/12 (91.7%), hard=0, >90deg=0。
- 结论：`development_diagnostic_only_no_confirmatory_claim`。

该阶段允许调参，只能作为诊断，不得写成确认性结论。
