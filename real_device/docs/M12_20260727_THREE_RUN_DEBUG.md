# M12 三次真机实验快速复盘（2026-07-27）

## 结论

三次实验没有复现仿真效果，不是单一“安全阈值太低”：

1. Proposed+ 的一次 low-force 只发生在 RAMP，且在 1.000 s、0.140 mm 内恢复；
   旧逻辑却把它当成整个方向指纹无效，导致 shortcut 被清零。
2. Proposed-local 的模型输入已经生成，但旧配置调用 `/usr/bin/python3`，在导入
   `torch` 时失败；随后实际运动使用的是透明经典线索，不是仿真模型。
3. Direct 第一次射线实际走满约 4.218 mm，问题是方向错，不是距离短；第二次重扫
   有 6 次可恢复 RAMP low-force，旧的一票否决又在第二条射线前结束流程。
4. 表面输送把很宽的接触区间误当成“力已稳定”，并允许在目标点过早返回；这会表现为
   看似没有继续下压或只走一两步。
5. Proposed+ 的旧 EIG 把离当前点超过 1.20 mm 的网格点直接视为不可达。附近节点测完
   后，即使远处还有未测节点也会假穷尽，造成停顿、折返和看似随机的探索。

## 三次日志的交叉证据

| 运行 | 锥扫 low-force | 旧方向/置信度 | 运动结果 | 根因 |
|---|---:|---:|---|---|
| `M12_proposed_plus_entropy_grid_20260727_204738_978` | RAMP 1；SCAN 0 | 约 206°；0.2665 | shortcut 归零，进入旧网格 | 恢复后的 RAMP 被全局否决 |
| `M12_proposed_local_spiral_20260727_205147_058` | 0 | 204.95°；0.2510 | 旧 shortcut 仅 0.25 mm，随后慢速残差 | 系统 Python 缺 torch；模型未参与；旧距离/速度小 |
| `M12_baseline_improved_direct_x3_20260727_205539_685` | 第一次 0；第二次 RAMP 6 | 约 205–208° | 第一条 ray 走满仍 miss；第二条 ray 前停止 | 错向 + 第二次弱事件一票否决 |

三个 touchdown 的两两 XY 距离约为 0.089、0.156、0.218 mm，Z 跨度约
0.023 mm，说明“同一放置位置”具有足够好的重复性，可用于诊断方向一致性。

2026-07-25 曾出现一个物理 downward-progress capture candidate：
`[471.263, 2.873] mm`。它没有完成插入，不能当成认证孔心；若夹具未动，仅把它作为
条件参考，则本次三个 touchdown 指向该点约为 148–153°。透明线索约 205°，相差
约 53–58°；冻结 v7 模型离线输出约 282°，相差约 129–134°。

## 模型链路与 sim-to-real 一致性

模型运行环境现固定为：

```text
real_device/model_runtime/.venv/bin/python
torch 2.4.1+cpu
numpy 1.24.4
```

对 Proposed-local 与 Direct 第一次完整锥扫进行端到端回放，归一化、CNN 和三 seed
ensemble 均成功：

| 输入 | v7 方向 | APS | seed top-1 span | simulation gate |
|---|---:|---:|---:|---:|
| Proposed-local | 282.01° | 3 | 15° | false |
| Direct | 281.67° | 6 | 180° | false |

完整锥扫的正向 Fz 均值仅约 1.771 N 和 1.923 N，而运行时按 3.5 N 目标构造 context，
即 Fz/target 约 0.51/0.55；训练 context 的相应均值约 1.00。16 个 context 维度中
有 11–13 个超过训练均值 3σ。把归一化目标从 1.0 扫到 4.0 N 并不能修好方向：
输出仍在约 274–329°，且 simulation gate 全部失败。

当前 WORLD* 信号还缺少完整 payload/CoM 重力补偿、Bota measurement-origin 到 tip
的力矩平移，以及同姿态 air-cone nuisance subtraction。这些项本身也形成随锥扫相位
变化的一阶谐波，无法仅凭一条接触锥扫与真正的接触方向特征可靠分离。

因此当前可确认的是：

- 模型特征预处理与归一化代码确实执行了；
- 旧 full-loop 确实没有让模型 posterior 选择运动；
- 运行时依赖已经修好；
- 当前真实输入与训练域不一致，而且输出方向与条件 CAD 参考明显矛盾；
- 不能通过修改一个阈值或归一化分母把该输出合理地宣称为已标定方向。

## 已落地的控制修改

- 所有表面搜索统一为 2.0 mm/s，XY 观测微步 0.30 mm。
- Pure/Proposed-local 螺旋 pitch 0.50 mm/rev、半径 4.20 mm。
- Direct 保持 4.20 mm 射线，三次方向扇形为 0/-45/+45°，reset 为
  0/+0.75/-0.75 mm；第二次恢复后的弱指纹不再被一票否决。
- 表面优先维持 4.0±1.0 N，最多做 3 次端点 Z 修正；删除宽力区间的立即返回。
- 锥扫仍以 3.5 N 为目标，deadband 0.50 N，admittance 0.20 mm/(N·s)，最大下压
  0.70 mm/s、0.70 mm，以减小下一轮真实输入与训练工况的 Fz 差异。
- 可恢复 low-force 使用连续质量：

  ```text
  quality =
      exp(-0.02*N_ramp)
    * exp(-0.12*N_scan)
    * exp(-0.10*T_low/3s)
    * exp(-0.10*R_reacquire/0.25mm)
  ```

  本次 Proposed+ 约 0.896，Direct 第二次约 0.825；不再归零。训练准入仍要求
  low-force episode 为零。
- shortcut 为 0.75–1.50 mm，使用门降至 0.01；弱但完整、已恢复的指纹仍产生有界
  位移。
- Proposed+ 先执行冻结方向的 ±12° CAD 蛇形 corridor，再执行完整 2.50 mm CAD
  圆盘上的残余 EIG。lane spacing 为 0.50 mm；旧的 1.20 mm “不可达” veto 删除。
  任意长的高层目标仍由底层拆成 0.30 mm 的新鲜 Bota 观测微步。
- 所有使用锥扫的入口在连接 KST 前，必须用哈希固定 trace 完成 venv、归一化、
  三 seed CNN 的端到端 preflight；实际锥扫后推理失败也会停止，不再静默退化。
- 实时图固定为正交投影：+X 左下、+Y 右下、+Z 向上；移除黑色 25 mm peg 轴。

独立的 20 N / 0.75 Nm hard guard、freshness、workspace、tracking、单 command owner
和 KST 异常停止保持不变。当前三次峰值约为 9.7–9.9 N、0.11–0.13 Nm，只能说明本次
没有触发 hard guard，不能证明更高 hard guard 的停止距离或结构后果已经验证。

## 下一轮最小实验顺序

1. 先跑 Pure Outward，确认 2.0 mm/s、4±1 N 与 4.2 mm 半径确实执行，不再端点早停。
2. 再跑 Proposed+，检查事件中：
   `LEARNED_RUNTIME_PREFLIGHT_PASS_NO_MOTION`、
   `CONICAL_ROUTING_QUALITY_GRADED`、
   `DIRECTIONAL_CAD_CORRIDOR_PLANNED`。
3. 再跑 Direct。按本次重复方向，第二条 `-45°` fan ray 约为 160°，最接近上述
   条件 CAD 参考；新的 recovered-low-force 规则应允许它实际执行。
4. 不把当前 282° 模型输出接成运动，先完成至少三个已知 CAD offset 的接触锥扫和
   同姿态 air-cone；用这些数据拟合 real-to-sim wrench/frame adapter，再冻结一个
   新的 M12-only bundle。先开放 Proposed+ grid-prior 权限，再评估 capped shortcut，
   最后才是完整 direct ray。
