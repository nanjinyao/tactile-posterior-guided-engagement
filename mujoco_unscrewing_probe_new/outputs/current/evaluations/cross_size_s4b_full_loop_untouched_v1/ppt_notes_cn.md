# Cross-size S4b untouched：PPT 注释

所有 estimator 变体统一使用 Proposed+ hybrid；纯 outward 是非学习基线，
straight-to-center 是读取真中心的 privileged 执行诊断。所有方法共享尺寸
commissioning 后的 admittance/R3 漏斗、未放宽的 raw hard threshold、
130 s 与零预算扩展。严格成功中的离线 true-center/contact-tag 审计不可用于在线控制。

## M10

- M8 zero-shot + P+ hybrid：12/12 (100.0%), hard=0, executed>90deg=0, latent>90deg=0。
- S4b calibration-only + P+ hybrid：12/12 (100.0%), hard=0, executed>90deg=0, latent>90deg=0。
- S4b base-anchored residual FT + P+ hybrid：10/12 (83.3%), hard=1, executed>90deg=0, latent>90deg=0。
- pure outward：11/12 (91.7%), hard=0, executed>90deg=0, latent>90deg=0。
- privileged straight-to-center：12/12 (100.0%), hard=0, executed>90deg=0, latent>90deg=0。
- 结论：`S4_fine_tuning_not_sufficient_for_frozen_gate`。

## M12

- M8 zero-shot + P+ hybrid：10/12 (83.3%), hard=0, executed>90deg=0, latent>90deg=0。
- S4b calibration-only + P+ hybrid：11/12 (91.7%), hard=0, executed>90deg=0, latent>90deg=0。
- S4b base-anchored residual FT + P+ hybrid：10/12 (83.3%), hard=0, executed>90deg=0, latent>90deg=0。
- pure outward：8/12 (66.7%), hard=0, executed>90deg=0, latent>90deg=0。
- privileged straight-to-center：12/12 (100.0%), hard=0, executed>90deg=0, latent>90deg=0。
- 结论：`fine_tuned_passed_but_necessity_not_identified`。

untouched 数据禁止用于任何调参；查看结果后该集合即永久退役为测试证据。
