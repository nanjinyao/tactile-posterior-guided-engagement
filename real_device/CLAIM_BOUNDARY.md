# M12 真机闭环的当前主张边界

## 现在成立到哪一层

当前代码已经形成一条结构完整、默认失效安全的闭环：

```text
安装/坐标/传感器资格化
→ touchdown
→ 主动锥形触觉观测
→ real-first 方向/状态后验（M12 development capped control）
→ 有界输送与残差搜索
→ 物理倒角捕获资格化
→ 分段插入
→ 独立到底事件
→ KST end 或显式异常终止
```

这证明的是**控制与证据链在软件层面闭合**，不是 full-loop 成功率已经得到统计证明。
M12/M10/M8 已各有一轮完整 26 点真机采样；当前 bundle 只获得 M12 的
`DEVELOPMENT_CV_CONTROL / CAPPED_SHORTCUT` 高层权限。活动配置仍将以下字段明确记为
未完成的独立计量，而不会把 operator-taught 假设冒充认证值：实体工具/孔座实例 ID、
完整 `WORLD <- fixture` 旋转和平移、实测孔心、
直立 TCP 旋转、经 pivot/touch-off 标定的完整 media-flange→tip 向量、M12 相对 yaw、
42 mm 局部 tip 高度、径向跳动、payload/CoM/惯量、sensor-origin→tip 变换、实体 M12
孔深/底面、controller load handling 和低载荷复验。

## 几何与坐标

- peg 轴和 fixture `+Z_F` 共线；按当前 TCP 约定，注册的 `+TCP-Z` 指向
  `-Z_F`。
- 选定 peg 平面的法向、法兰 A/C 切向基准面的法向与 fixture `+X_F`
  对齐。照片中从右到左是 M8、M10、M12，因此图像横向 `u = -x_F`。
- 因为 `+Z_F` 从桌面指向上方，且照片从 `+Z_F` 俯视，所以右手关系
  `+Y_F=+Z_F\times+X_F` 要求 `+Y_F` 在图中向下；数学绘图坐标为
  `(u_right,v_up)=(-x_F,-y_F)`。这只是照片投影，不改变 fixture 动作坐标。
- 竖直棱只能辅助检查工具轴方向，不能唯一确定六角 yaw；fixture 长边的竖直侧面
  跨越 `X_F-Z_F`，因此它与上述法向平行的 peg/法兰平面是**垂直**关系，而不是
  三者都平行。
- 最终控制不依赖照片或肉眼平行，而使用实测、右手、正交的三维
  `WORLD <- fixture` 旋转。配置还会独立比较安装后的直立 TCP 旋转。
- fixture frame 的 `+X_F` 由实测 M10→M12 中心向量定义，因此 M12 孔心必须使用
  as-built 距离，而不能把名义 `45 mm` 偷换成标签真值；孔心/顶面不确定度也必须小于
  采样标签预算。

CAD 中把完整正六角 peg shape 绕自身中心旋转 `-15 deg`，使选定 flat normal 与
法兰 B/C tangent-plane outward normal 都成为 flange-CAD `+45 deg`。装机只做一次
`B/C NORMAL -> +X_F`，于是 peg/socket 都是 fixture `0 deg (mod 60 deg)`；因此
不得把 `-15/+15/+45 deg` 再加到 MuJoCo、C6、训练标签、runtime 或动作里。真实 FDM
重打仍可能因材料、填充、支撑、翘曲和螺栓落座改变质量、CoM、跳动及顺应性，所以每次
重打必须生成新的实体实例 ID 并重新测量，不能沿用旧 payload 资格。粉色 CAD 的
`42 mm` 也不等于完整机器人法兰→尖端 TCP；若整链轴向误差为 `4 mm`，15° 锥扫会
产生约 `4 sin(15°)=1.04 mm` 的真实尖端假圆，足以污染接触指纹。

## 数据与模型

- 方向真值只能由合格的 as-built fixture 标定和**实测 touchdown 位置**得到：
  `实测 as-built 孔心 - 实测接触点`。CAD 孔心只是采样模板的初值；命令点、照片、
  模型预测和捕获结果都不能反过来充当标签。
- 每个尺寸预注册 26 次：16 clean、4 holdout、4 intentional quarantine 和
  2 exact-repeat sentinel。当前 12 条 holdout 已在早期可观测性分析中消费，因此只作
  report-only，不再声称 untouched；quarantine 只诊断，sentinel 只看漂移，三者均不
  参与训练、normalization 或模型选择。
- clean 训练只接受完整、新鲜、frame/phase 合同一致且无 capture/基线污染的样本。
  弱但完整的指纹可以保留强度并进入离线 QC；可恢复 low-force 在 full-loop 中仍可
  发出缩短的有界动作，但不会因此被自动升级为 clean training evidence。
- 旧打印件、未知 frame 或旧 clocking 的真机 trace 保留作历史证据，但不得旋转标签后
  混入新实体训练。实体实例、安装 session、CAD/配置哈希必须共同分组。
- 当前 DirectServo 的锥扫行在环内使用确定性 command phase 和原始 Bota 时序；
  pre/post KST 实测验证端点，但不声称逐 tick 测得完整三维尖端轨迹。48 条样本可按
  当前 development 合同作为 clean signal fingerprints 训练，因为其 phase、freshness、
  endpoint、baseline、capture 与 timing 条件均完整；但它们不是 external-tracking
  metrology 级的动态轨迹证据，这也是权限只到 `DEVELOPMENT_CV_CONTROL` 的原因。
- `real_multisize_direction_v1` 输出 48-bin 多峰方向 posterior 和三状态 posterior；
  M12 full-loop 的 TOP1 方向与 CAD-bounded 路由由它控制，圆均值仅作诊断。
  hard guard、capture、插入和 terminal success 始终由确定性物理事件链判定。
- 弱但完整的新鲜指纹仍有动作权限；quality/APS/多峰程度只连续缩短 shortcut 并放宽
  posterior/CAD 验证搜索。incomplete、stale、nonfinite、frame mismatch、硬保护或
  锥扫阶段断裂才禁止模型动作。推理失败会停止当前方法，不会静默换 classical heuristic。
- clean-only LOSO 的方向 median/p90 为 `33.62/165.23 deg`，`16/48 >90 deg`，
  因而只能称 development capped control。已消费的 4 条 M12 holdout 为
  `0` catastrophic、p90 `31.17 deg`，支持快速有界首轮，而非生产资格。

## 四种方法的公平比较

四种方法共享同一打印几何、起点分布、力/力矩限制、时间预算、capture、插入和 terminal
判据；只改变“如何从触面位置到达可捕获区域”：

| 入口 | 方法 | 锥扫 | routing / search | 模型权限 |
|---|---|---:|---|---|
| `run_m12_full_loop_proposed_plus_entropy_grid` | Proposed+ | 1 | model/CAD shortcut + TOP1-centered 80% corridor（45° 内不足则跳过）+ 95% credible EIG | M12 capped high-level control |
| `run_m12_full_loop_proposed_local_spiral` | Proposed | 1 | model/CAD shortcut + measured-endpoint local spiral | M12 capped high-level control |
| `run_m12_full_loop_baseline_pure_outward_spiral` | Pure Outward | 0 | global bounded outward spiral | 不调用 |
| `run_m12_full_loop_baseline_improved_direct_x3` | Improved Direct ×3 | ≤3 | 每次重扫/新模型方向/单条有界 ray | 每次重新推理 |

`run_m12_sampling_conical_fingerprints_26pt` 是独立的数据采集入口，不属于四方法
full-loop 比较；`run_m12_sampling_conical_fingerprints_26pt_resume(parentRunDir)`
只创建经过哈希验证的 child continuation，也不属于四方法比较。其中三个锥扫方法
要求活动 bundle 的 M12 `CAPPED_SHORTCUT` 权限；Pure Outward 明确不触发模型
preflight 或推理。

公平比较的共同锚点是通过有界 touchdown 横向漂移门后的**实测 touchdown XY**，
不是人工 air-start 命令点。该门对相对 air-start 的最大 XY 偏移在 `>0.20 mm` 记录预警、
在 `>1.00 mm` 停止，覆盖同一 KST 链路已观测的多段小耦合，但不允许无限偏离。共同的
fast-but-bounded 表面策略使用统一的自适应速度：4 N 附近且 tracking/torque 稳定时
`4 mm/s`，可恢复偏离时 `2 mm/s`，capture/insertion 交接时 `1 mm/s`。Proposed-local
pitch 为 `1.6 × capture radius`；Pure Outward 保持全局 outward 方法纯度。

这些修改仍保留有限的 `25 N / 0.75 Nm` hard guard、workspace/joint 边界、fresh/
finite feedback、命令所有者和 unknown DirectServo/KST state 收口。屏幕动画只是
低频操作可视化；科学证据是带 source timestamp 的原始 CSV、posterior、实际命令/
实测路径和 manifest。

“方向估计正确”“进入倒角可捕获区”“已插入”“任务成功”是四个不同事件。闭环不要求
每次成功，但必须在限定时间/动作预算内得到可解释的 success、abstain 或 failure，
且 failure 不能被保护性终止、回撤或模型自信度冒充。

## 仍不能声称

- 新 clocking 工具已经通过真机 full-loop、成功率或跨尺寸比较；
- 仿真模型已经获得真机控制权，或仿真通过等于 sim-to-real 通过；
- 低 wrench 下沉就是孔中心、倒角捕获、到底或可承扭；
- MuJoCo 6 mm 深度等于实测孔深；full-loop 代码现已要求实体 top-to-bottom 深度和
  bottom datum 交叉核对后，才生成 terminal depth band；
- 当前 Bota 数据已完成 measurement-origin 到 tip 的绝对 wrench 变换、完整重力补偿
  或认证测量精度；
- payload JSON 已自动成为 Sunrise controller 的 LoadData；当前 full loop 必须另外
  先用每次 redeploy 都更新的 UUID 和 `<=900 s` readback 证明当前
  station/IP/application/source 正在运行；再证明非零 LoadData 已注册且
  mass/CoM/inertia 与 payload evidence 一致，或改走同一 deployment 下、最终实体
  负载的独立 full-cone tracking 资格化。该 readback 在 KST 连接前与首个运动前
  重验，且剩余 freshness 必须覆盖整个 `600 s` hard task horizon；
- 软件静态测试、离线 replay、异常 `end` 或人工回位等于真机任务成功。
