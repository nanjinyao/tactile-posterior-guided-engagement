# Cross-size S3 pilot：PPT 可用结论

预注册门槛：straight-to-center 执行门成功率不低于 90%；
direction-head fine-tuned 严格闭环成功率不低于 80%；
fine-tuned hard stop 不超过 0，执行方向误差大于 90° 的样本不超过 0。
所有方法共享 130 s、零扩展、R3 物理漏斗和 CAD-bottom 终止验证。
四个非 oracle 控制器使用 sensor-only capture；straight-to-center 是读取真中心的 privileged execution diagnostic。
严格成功还接受离线 true-center/contact-tag capture 审计，不能冒充在线 terminal signal。

## M10（n=12）

- M8 zero-shot：10/12 (83.3%)，hard stop=1，成功样本中位时间=22.81 s。
- size calibration only：10/12 (83.3%)，hard stop=1，成功样本中位时间=39.56 s。
- direction-head fine-tuned：10/12 (83.3%)，hard stop=1，成功样本中位时间=30.42 s。
- pure outward：9/12 (75.0%)，hard stop=1，成功样本中位时间=34.20 s。
- privileged straight-to-center：9/12 (75.0%)，hard stop=3，成功样本中位时间=4.06 s。
- fine-tuned task 指标门：未通过；straight-to-center 执行门：未通过；综合门：未通过。
- 证据解释：`oracle_execution_feasibility_gate_failed`。

## M12（n=12）

- M8 zero-shot：9/12 (75.0%)，hard stop=1，成功样本中位时间=30.49 s。
- size calibration only：8/12 (66.7%)，hard stop=3，成功样本中位时间=34.87 s。
- direction-head fine-tuned：9/12 (75.0%)，hard stop=3，成功样本中位时间=43.72 s。
- pure outward：7/12 (58.3%)，hard stop=4，成功样本中位时间=47.19 s。
- privileged straight-to-center：12/12 (100.0%)，hard stop=0，成功样本中位时间=5.28 s。
- fine-tuned task 指标门：未通过；straight-to-center 执行门：通过；综合门：未通过。
- 证据解释：`direction_head_finetuning_not_sufficient_to_pass_this_pilot_gate`。

## 必须保留的边界

这是全新 MuJoCo 尺寸适配 pilot，不是真机数据实验。只有当 fine-tuned 通过同一预注册门、
同时 zero-shot 与 cal-only 均未通过时，才能在该仿真分布内称方向头微调是必要的。
即使仿真通过，也不能推出真机数据微调的必要性；后者仍需要独立真实 F/T train/cal/test。
privileged straight-to-center 是直线路径执行诊断，不是保证更优的性能上界。
