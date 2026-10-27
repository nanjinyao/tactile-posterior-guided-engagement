# M8 接触指纹闭环：端到端科学审查

Updated: 2026-07-15

> Scope note: this is the closed-loop mechanics snapshot that predates the
> later Jin-style estimator-swap audit, cluster interval, post-hoc fingerprint
> strata and two current matrices. Those additions are owned by
> `CLAIM_BOUNDARY.md` and their dedicated `outputs/current/evaluations/`
> artifacts; they are not silently folded into this document.

## 1. 审查结论

当前方案在**限定任务内**具有清楚且克制的物理逻辑：

```text
视觉只给粗先验
-> 一次固定锥形动作主动制造可辨识的相位指纹
-> 小模型只输出方向分布与软支撑状态
-> 视觉/CAD 提供粗径向约束
-> 规划器只消耗后验中可靠的一段距离
-> finite outward spiral 保留稠密搜索机会
-> sensor-only downward-progress cue 完成搜索到插入的交接
-> peg-tip RCC / energy / CAD guard 负责捕获后的执行安全
```

这条链路的优点不是算法堆得多，而是每个模块只解决一个物理瓶颈：

- 主动探测负责**可观测性**；
- CNN 负责读取难以手工编码的**相位形状**；
- 视觉与 CAD 负责不应交给网络猜测的**尺度和几何**；
- outward spiral 负责**经验性稠密搜索**；
- 导纳、RCC 和安全盾负责**物理执行**。

因此，方案已经形成一个可信的 **M8 MuJoCo 几何接合闭环**。但它还不是完整的自动拧松系统，也不是 KUKA iiwa 真机闭环证据。当前 strict 批次条件化于 benchmark 已声明 `ambiguous_top_contact`；上游 no-contact / below-rim / ambiguous router 仍只是部署接口，尚未作为分类器评估。当前终点是 Allen bit 到达 CAD socket bottom，并通过“有符号轴向落座”或“低 wrench + 稳定深度”双通道 dwell；工具旋转、槽内耦合确认、反转拧松和螺钉提取仍在任务外。

## 2. 真正的任务和隐藏状态

### 2.1 第一版任务

已知 Allen bit/socket 大致几何，视觉使 bit 到达螺钉头上方，并已消除大的 roll/pitch/yaw 误差。接触后只处理视觉残余范围内的平面中心偏差，然后完成低力接合。

### 2.2 隐藏状态

控制真正需要但不能直接观测的是：

$$
\delta = c_{socket}-p_{tip},
$$

以及当前属于 flat support、孔口/倒角过渡、inner wall 约束还是已经插入。单个 $F_z$ 样本无法唯一确定 $\delta$ 的方向；摩擦锥还会使不同位姿产生重叠的 wrench。

### 2.3 当前可用观测

- 外置 6D F/T；
- 末端 pose、命令和 conical phase；
- 视觉中心残差先验 $p_v(\delta)$；
- 近似 CAD 几何和 socket 深度。

MuJoCo contact tag 和真实孔中心不得进入在线控制。它们只允许在 episode 结束后作离线成功评分。

## 3. 小而关键的接口

### 3.1 固定主动探测

孔口平面状态模糊时，执行一次：

$$
\operatorname{roll}=\alpha\sin\theta,\qquad
\operatorname{pitch}=\alpha\cos\theta,
\qquad \alpha=15^\circ,\quad \theta\in[0,2\pi).
$$

名义 peg-tip $XY$ 不动，导纳控制维持低下压力。15 deg 来自锥形探索论文的有效模板，不是 M8 的理论最优角；它只在孔口可自由摆动时使用。进入倒角、inner wall 或 socket 后必须回正，不能继续 15 deg 扫描。

### 3.2 学习输出

主模型是 1D circular CNN：输入按 phase 对齐的 $F_z,\tau_x,\tau_y,z$ 和无量纲上下文，输出：

- 48-bin circular direction distribution $q_\psi(\phi\mid y)$；
- APS angular set、circular concentration 和 ensemble disagreement；
- 三个 flat-support 状态的概率，且只作软径向似然。

它**不输出毫米距离、不触发 capture、不判定任务完成**。P1v5 独立仿真集的方向中位/p90 误差为 1.30/4.05 deg，7.5 deg 内比例 0.989，APS coverage 0.994；原始方向置信 ECE 仍为 0.304，因此执行接口应优先使用 APS/ensemble 结构，而不是把 max-softmax 当真实概率。

状态头在同一独立集上的 accuracy/ECE 为 0.989/0.013。独立 action-interface audit 显示，加入 soft state-CAD likelihood 后，后验覆盖从 0.994 提升到 1.000，残余真实距离中位数由 1.399 降到 0.805 mm；这只支持“软规划信息有用”，不支持状态等于真实接触模式或网络学会距离。

### 3.3 联合后验

在线中心后验是：

$$
p(c\mid y,v,G)\propto
p_v(c)\,
q_\psi\!\left(\arg(c-p_0)\mid y\right)
\sum_s p_\psi(s\mid y)L_G(\|c-p_0\|\mid s).
$$

$L_G$ 使用近似 CAD 的软径向带和非零 likelihood floor。它不能使用真实中心、MuJoCo contact points 或真实径向标签。

### 3.4 有限运输与覆盖

令 $u$ 为 circular posterior 的控制方向，对后验样本计算：

$$
s_i=(c_i-p_0)^T u.
$$

运输距离取风险调制的低分位数，并受反向概率、后验半径和真机步长上限限制：

$$
d_{move}=\operatorname{clip}\!left(Q_{\alpha(q)}(s_i)-r_{eff},0,d_{cap}(q)\right).
$$

移动后重新计算 residual posterior 的覆盖半径，再执行局部 outward Archimedean spiral。螺距始终满足：

$$
\Delta r_{turn}\le 2r_{eff}.
$$

局部搜索耗尽或发生可恢复的 pre-hard event 时，物理卸载、抬升、返回视觉锚点并执行完整 outward fallback。方向后验越不可靠，系统越连续地退化为普通覆盖搜索，而不是突然相信错误中心。

## 4. 阶段逐项审查

| 阶段 | 当前方法 | 科学判断 | 当前缺口 |
|---|---|---|---|
| 视觉粗定位 | 合成 2 mm isotropic Gaussian / 95% disk | 作为仿真先验合理 | 必须用真机重复定位得到 $\Sigma_v$；不能继续把合成分布称为 measured prior |
| guarded touchdown | 低力竖直接触和稳定 settle | 必要且物理清楚 | 真机需校准 TCP、重力补偿、表面高度和 force ramp |
| 状态路由 | 部署设计中，明显 below-rim/inserted 交给 depth+wrench rule，模糊 flat support 才 probe | 比“六类都交给 CNN”更简单可靠 | strict 24 批次从 ambiguous 状态开始；上游 router 尚未评估，阈值仍是 simulation-initial |
| conical probe | 固定 15 deg、完整 360 deg、tip-center nominally fixed | 与参考论文一致；相位可比 | 真实 iiwa 上需验证回正残差、峰值 wrench 和 adapter flex |
| 数据表示 | phase sequence + 1D circular convolution | 比强制栅格成图更贴合 $S^1$ 拓扑 | polar-image CNN 仍应作为 representation baseline，而不是主模型必要条件 |
| 学习 | 小型三 seed ensemble，circular soft targets，group split before augmentation | 模型规模与任务匹配 | 没有真实 OOD 保证；direction ECE 偏高 |
| CAD/视觉融合 | soft likelihood，不直接回归距离 | 利用物理先验且保留误差空间 | CAD 是参数化近似值，不是实际 screw 的 metrology |
| 规划 | posterior-limited transport + residual finite outward search | 兼顾捷径与残余搜索，逻辑成立 | 有限 spiral 无外边界 padding，不是完整域覆盖证明 |
| capture handoff | $z$ trend + bounded wrench + freeze XY + low-force dwell | 在线不再读 MuJoCo tag，接口正确 | 绝对深度依赖真机 CAD/表面坐标标定 |
| 插入 | upright descent + tip-RCC proxy | 符合小间隙深孔的几何要求 | 当前不是 iiwa 全臂动力学/原生控制器实测 |
| safety | hard wrench + energy/progress + CAD terminal shield + bounded recovery | 安全与学习解耦，结构合理 | 尚无 passivity proof；stall recovery 未在动态批量中证明收益 |
| terminal | CAD depth band内统一双通道 dwell：轴向落座力，或低 lateral wrench + 深度稳定 | 同时覆盖受力落座与 centered free insertion；不读真中心/tag | 阈值仍是仿真初值；真实 coupling/first-turn verifier 未实现 |

传感器因果链也被固定：每个 physics step 只在 post-step 推进一次 observation
model，控制器、belief update 和 terminal verifier 复用同一个 delayed/noisy
measured sample；未延迟的 physical wrench 只驱动 fast hard stop。终端事件后的
0.50 s inactive display tail 不计入 active budget，且不能反向改变已经冻结的
success、hard-stop 或 offline capture score；但该 tail 的 raw all-step 峰值仍保留
为独立诊断。

## 5. Strict 完整闭环的七种方法角色

| 方法 | 角色 | 搜索动作 | 它隔离的问题 |
|---|---|---|---|
| Proposed | paper-facing complete method | fingerprint posterior 的有界运输 + residual/full outward search | 一次主动触觉捷径后能否保留有效搜索 |
| Proposed+ | adaptive research branch；无 superiority claim | measured-path belief update + task-aware grid；条件触发 full-radius outward search | 在线观测是否值得改变搜索顺序 |
| Outward | geometry-only comparator | 从同一视觉锚点作普通 outward Archimedean sweep | 不使用主动触觉的经验搜索成本 |
| Entropy-grid | strong transparent active-perception diagnostic | 宽 belief 上按 capture/information 排序离散覆盖 | 信息驱动排序本身能做到什么 |
| Direct-first | mechanism diagnostic | 相同 fingerprint 后只执行第一次 direct ray | 正确方向但未知距离时的几何 margin；无物理 retry/reset claim |
| Random | weak-search lower diagnostic | 冻结 seed 的随机 waypoints | 无结构探索下界 |
| Oracle | privileged upper bound | 在线使用真中心直达 | 搜索不确定性被移除后的上界；不参与方法 claim |

Outward 是 Unscrewing 文献 surface sweep 的**可用传感器版本**，但当前没有其 bit motor 自转和 motor-torque coupling verifier，因此不能称为完整论文复现。历史 direct/retry 会重启 simulator 且没有把 reset 运动计入时间/路径，不能进入 strict paired table；当前只保留物理闭环内的 `direct_first` 作为机制诊断。

有限 Outward 的确定性边界审计进一步表明：0.4 mm/rev 虽满足相邻圈距
$\leq 2r_{eff}$，但最坏外边界 gap 为 0.6421 mm，大于 $r_{eff}=0.25$
mm；因此它是经验性搜索 comparator，不是完整视觉圆盘覆盖证书。证据在
`outputs/current/evaluations/m8_finite_outward_boundary_audit/`。

Random 是 seed-fixed 的无结构下界：不读取真中心，不使用 conical learner、CAD likelihood 或 spiral；它与其他非 Oracle 方法共享 capture/terminal、安全层和预算。单个 seed 可能幸运地形成短 chord，因此结论必须来自冻结的多点配对统计。

Strict 比较满足：同一初态、接触 nuisance、sensor seed、sensor-only handoff、双通道 terminal verifier、post-capture safety、hard limits，以及不可延期的 130 s accumulated simulated active-controller budget；`fallback_budget_extension_s` 必须为零。除 Oracle 外在线均不得读真中心/contact tag。单个视频只解释机制；成功率结论来自冻结多点配对统计。

### 5.1 Strict audit 中的 research / diagnostic branches

Proposed+ 不是把所有信息笼统称为“联合先验”：视觉给初始中心密度，
conical fingerprint 给方向/状态 likelihood，CAD 给 capture tube 与观测几何。
它先做与 Proposed 相同的有界短移动，再在剩余 belief 上执行 task-aware
grid segments。每个候选动作同时计算整段路径的 capture probability 与
endpoint 的 information gain，并扣除接触/路程代价；节点之间保持低下压力，
走空间直线、minimum-jerk 时间参数，只在 endpoint dwell 后读取 F/T 类别。

它与历史 entropy-grid 分支的区别是：后者从视觉/CAD 宽先验开始，纯粹按最大期望熵减
选点，并多次 unload/lift/touchdown；Proposed+ 先利用 learned shortcut，
节点间连续低力接触，且“整段未 capture”会软排除已经扫过的 CAD capture
tube。若 F/T 类别在当前 belief 下很意外，其 likelihood 会降温，避免仿真
观测模型把几何负证据覆盖掉。Proposed+ 不使用 local spiral；只有局部
task-grid 价值耗尽或 conditional nominal reserve 不再允许可选动作时，立即切换
full visual-support outward。该 reserve 只在跟踪误差不超过 0.25 mm、candidate
额外处理不超过 6 s、continuous-contact motion 开启且没有 unsafe interruption
等冻结条件下提供名义调度；它不是覆盖/成功保证，且绝不延长 130 s hard budget。

有限圆盘的离散覆盖单独做了证书：$R=4.8953$ mm，$r_{eff}=0.25$ mm，
action spacing 0.3359 mm；原 clipped disk 为 673 nodes，边界 padding 后为
749 nodes，解析和稠密检查的最坏 support-to-node gap 均为 0.2375 mm
$<r_{eff}$。它只证明 action-node geometry；只有实际访问所需节点/轨迹时才
得到条件几何交集，不能推出 tracking、cue、130 s 或 CAD-bottom 成功。

修正后的 Proposed+ 还加入一侧方向一致性：depth/F/T candidate 若满足

$$
(\mathbf x_c-\mathbf x_0)^\top\hat{\mathbf u}_{top1}<-r_{eff},
$$

则视为可能沿外侧边下降，不直接进入 insertion。它只使用初始/当前机器人
坐标、冻结 top-1 direction 和 CAD $r_{eff}$，不读真中心/contact tag。

冻结 seed-20260723 的 24 点 strict complete-loop 结果是：Proposed / Proposed+
/ Outward / Entropy-grid / Direct-first / Random / Oracle 分别达到
19/19/19/22/15/10/22 个 CAD-bottom；offline capture 分别为
23/23/23/23/18/12/24。所有方法均为 0 budget extension、0 budget violation。
Proposed+ 与 Random 各有 1 次 raw-physical torque hard stop；Outward 与
Random 各有 1 次 sensor handoff false positive。

Proposed 与 Proposed+ 为 18 个共同成功、各 1 个单独成功、4 个共同失败，
McNemar $p=1.0$。共同成功点上 Proposed+ 相对 Proposed 的 median 差为：
bottom time $+0.40$ s，actual/contact path $-0.07/-2.12$ mm，wear
$-6.09$ mJ。它支持效率/接触暴露 tradeoff，不支持 superiority。

Proposed+ 在 7/24 个 rollout 中为 nominal fallback schedule 提前换挡，在
8/24 中实际使用 global outward。它的 5 个非成功为 1 次 search exhaustion、
3 次 capture without valid bottom dwell 和 1 次 hard stop；没有 timeout。
这说明合理闭环可以失败，但失败必须由具名出口和硬预算收口。

Entropy-grid 的 22/24 高于 Proposed/P+ 的 19/24；在与 Proposed 的 18 个共同
成功上，它的 contact path/work 全部更低。因此它必须保留为强透明诊断，
不能因结果强而隐藏。即使 Oracle 也只有 22/24 bottom，说明“送到真中心”仍
不等于 terminal physics 已解决。

本批 126 个 strict success 全部由 `cad_depth_low_wrench_dwell` 完成；
signed axial-force branch 虽已实现并有单元测试，但没有被该批量实证覆盖。
selected `sobol_008` Proposed+ 用 5 次 grid observation，11.944 s capture、
16.209 s CAD-bottom、无 global outward；它只作因果机制示例。

### 5.2 Historical E3 capture-only evidence

<!-- FOUR_METHOD_AUDIT_BEGIN -->
冻结的 36 点 M8 配对审计是历史 capture-only E3 证据，不包含当前 strict
CAD-bottom success 合同，且其中 direct/retry 未计 simulator reset 运动。以下
数据只用于解释搜索机制，不能与上面的 strict 24 完整闭环表混算。

| 方法 | eventual / no-retry | 成功时间中位 | 成功 total/contact path | hard stop |
|---|---:|---:|---:|---:|
| Proposed | 34/36 / 34/36 | 12.47 s | 14.71/14.61 mm | 0 |
| Baseline A outward | 30/36 / 30/36 | 36.06 s | 77.11/77.07 mm | 1 |
| Baseline B direct/retry | 28/36 / 25/36 | 7.49 s | 5.36/5.36 mm | 0 |
| Baseline C random | 19/36 / 19/36 | 10.19 s | 20.82/20.82 mm | 1 |

Proposed/B 为 27 个共同成功、7 个 Proposed-only、1 个 B-only、1 个共同失败，McNemar $p=0.0703$。B 在共同成功点几乎总是更快、更短，符合 direct shortcut 的物理预期。13 个 direct-ray 几何 margin 为负的点，B 第一次成功率只有 0.154；margin 非负时为 1.000，说明失败机制确实是“方向误差乘径向距离使 ray 错过 capture width”，不是人为削弱。

Proposed/A 的 29 个共同成功点中，Proposed 更快 82.8%，total/contact path 和 work 更低各 86.2%。Proposed/C-random 为 both 18、P-only 16、C-only 1、both-fail 1，McNemar $p=0.000275$。Random 在幸运成功点可能更快，但 Proposed 在共同成功点中 66.7% contact path 更短，且 C 的 all-trial contact path 中位数达到 86.38 mm。它是无结构下界，不是强文献基线。

Proposed 使用了 4 次物理 reset + 全半径 outward search fallback。以上支持“一次 learned fingerprint + residual/global outward search 相对普通 outward/random 减少有限样本盲搜”，但不支持完整域覆盖、每个共同成功案例都更快更短或真机结论。
<!-- FOUR_METHOD_AUDIT_END -->

## 6. Direct-first 为什么只作机制诊断

方向正确时，direct ray 本来就应是最快的动作；如果 Proposed 在所有共同成功点都比它快，反而说明诊断项被削弱或比较不公平。Proposed 相对 direct-first 的合理卖点是：

- 不把“方向正确”误写成“距离也已知”；
- 运输距离允许有误差；
- residual outward spiral 在经验上处理 direction/radius bias；
- 失败时仍保留全半径经验性搜索机会，而不是只依赖有限重试。

因此报告必须把 direct-first 标成无 retry/reset claim 的机制诊断，并同时报告 strict CAD-bottom success、共同成功条件效率和完整失败类别。不能只挑 Proposed 成功的案例，也不能人为给 direct-first 注入错误方向。

## 7. 安全机制必须单独做 ablation

所有 strict 方法共享同一 post-capture guard 和 terminal verifier，保证搜索策略比较不被终端控制差异污染。安全机制的价值由另一组同轨迹配对消融回答：

- legacy post-capture controller；
- tip-RCC + funnel-health/energy/CAD shield。

当前 12 对 MuJoCo 结果是 hard stop 1 -> 0、最大 CAD 深度越界 1.00 -> 0.49 mm，但 bottom verified 5/12 -> 4/12，且 6 次干预都是 CAD-depth safe abort。它支持“避免继续做坏事”，不支持“提高完成率”“证明被动性”或“stall recovery 已有效”。

另有一个独立的极端机制演示：在相同 8° 残余倾斜命令和 0.6 mm 横向 TCP
偏差下，无漏斗分支实际倾角达到 9.69°，在 29.24 N / 128.59 mN m 时楔紧
hard stop；tip-RCC + supervisor 分支触发一次卸载/回撤/回正/重触，最终以
0.01° 到达 CAD bottom。M8 的 0.10 mm 单边间隙、4 mm 深度对应几何倾角尺度
仅 1.43°，因此该输入是刻意放大的 selected stress case。它证明仿真恢复动作
链能够执行，不改变上面的 12 对总体边界，也不代表真机故障分布或一般恢复率。

## 8. 采样和模型审查

### 8.1 已做对的地方

- 先做 symmetry/observability/collision audit，再训练 CNN；
- split base pose/trial 后才做 $C_6$ augmentation，避免旋转副本泄漏；
- 额外采真实 MuJoCo 六扇区 anchor，未把理想对称性当独立证据；
- P1v2 失败后只修 direction interface，P1v5 永久禁止再调参；
- 保存原始 6D F/T、pose、phase、controller/CAD metadata，便于真机复用 schema；
- CNN 只学方向和软状态，安全、距离和终点留给白盒模块。

### 8.2 还不能声称的内容

- 每个 P1v3 seed 的 fitting input 是 1716 条 grouped examples（1296 P0b-C6 + 180 P0d + 240 targeted）；它们不等于 1716 个独立位置；
- 完整 raw inventory 是 1674 个 rollout、651 个 unique XY；两组数字描述不同数据层级，不能混写；
- 仿真 $C_6$ 相关性不证明真实线缆、sensor frame 和机器人各向同性；
- M8 direction success 不证明 M10/M12 scale transfer；
- synthetic bias/noise/drift/delay 不是实测 sim-to-real distribution；
- APS coverage 是同仿真域校准结果，不是 real OOD guarantee。

### 8.3 当前是否还需继续大量采 M8 仿真

不需要先盲目再做 5000 条。M8 已有足够证据进入低风险真机校准阶段。新增仿真只应服务于明确缺口：

1. 在新 seed 上扩大 strict complete-loop 统计，而不是回用当前 24 点调参；
2. visual/CAD 边界 challenge；
3. 真机标定 dual-channel terminal verifier 的深度、wrench 与 dwell 阈值；
4. M10/M12 S2 已完成并失败；下一步只针对已归因缺口做小规模 learner
   calibration 与独立 controller/terminal commissioning。

## 9. M8/M10/M12 泛化

M8/M10/M12 是同一 hex topology，但不是严格相似缩放。绝对视觉误差、clearance ratio、chamfer、socket depth、力矩臂、制造偏心和接触刚度都会变化。

合理接口是无量纲化：

$$
\tilde z=z/a,\qquad
\tilde F=F/F_d,\qquad
\tilde\tau=\tau/(F_da),
$$

并加入 clearance/$a$、chamfer/$a$、depth/$a$ 的 CAD ratio。它提供 scale-transfer inductive bias，但 M8 训练结果不能替代 M10/M12 采样。

第一阶段 structured zero-shot 已完成。M8 P1v3 ensemble、15 deg 动作、
输入 normalization、APS 和所有门限均冻结；每个尺寸新采
$3$ states $\times2$ normalized radii $\times2$ off-grid angles $\times6$
physical sectors $=72$ 条动态轨迹：

| size | scan / hard stop | top-1 median/p90 | within 15 deg | APS coverage / mean size | state accuracy | context $>5\sigma$ |
|---|---:|---:|---:|---:|---:|---:|
| M8 reference | 72/72 / 0 | 1.75/9.25 deg | 1.000 | 1.000/2.63 | 1.000 | 0.014 |
| M10 zero-shot | 72/72 / 0 | 1.75/9.25 deg | 0.986 | 0.986/2.71 | 1.000 | 0.014 |
| M12 zero-shot | 72/72 / 0 | 1.75/9.25 deg | 0.972 | 0.972/2.94 | 0.986 | 0.153 |

这支持“无量纲 conical **方向接口**具有同拓扑跨尺寸迁移迹象”。它不支持
直接复用 M8 的 probability calibration、state-CAD likelihood、planner
radius、控制增益或 terminal threshold。M12 p90 raw force/torque 已升到
7.82 N/30.5 mN m，说明尺寸变化对执行控制仍有实质影响。rollout 中原本
写死的 `allen_bit_m8` body 也已改为从公共 `bit_tip` site 自动发现动态
body，并由三尺寸回归测试覆盖。

第二阶段 complete-loop S2 已完成。每个尺寸在其视觉 95% 范围内使用 30
个新 Sobol 位姿，M8 learner 完全冻结，CAD/capture radius/coverage pitch
按尺寸重算：

| size | Proposed / outward / oracle bottom | Proposed hard stop | executed control dir median/p90 | control dir $>90^\circ$ | common-success Proposed minus outward time/path |
|---|---:|---:|---:|---:|---:|
| M10 | 23/30 / 21/30 / 27/30 | 3 | 2.58/21.78 deg | 1 | -15.88 s / -46.16 mm |
| M12 | 21/30 / 19/30 / 30/30 | 4 | 3.58/24.76 deg | 1 | -17.86 s / -50.54 mm |

两尺寸都未通过冻结门：Oracle bottom >=0.90、Proposed bottom >=0.80、
Proposed hard stop=0、executed circular-mean direction $>90^\circ$=0。
每尺寸各有 2 个失败点的 control-direction error $>45^\circ$；其余失败
来自 search/terminal 或在方向较准时发生的 capture-release/reset/spiral
hard stop。因此不能把 S2 失败全部归因于 CNN，也不能只靠扩充训练集解决。

当前 30+30 条 S2 是永久 evaluation-only。下一步每尺寸另采 12--24 条
calibration，优先调 input affine/temperature/small adapter；同时用独立
oracle-guided 轨迹标定控制和 terminal。完成后必须用新 seed 生成 untouched
test。完整 backbone 重训仍是最后选项。

## 10. 低成本真机数据方案

硬件假设：KUKA iiwa 末端 -> 外置 F/T -> 刚性 adapter -> Allen peg。当前仿真只实现自由末端的笛卡尔导纳/tip-RCC 代理；真机前必须标定 flange-to-sensor、sensor-to-tip、gravity/bias 和 latency。

### H0：无接触和视觉标定

- 重复视觉定位，得到真实 $\hat\Sigma_v$ 和 95% ellipse；
- 6 个安全姿态采 F/T bias/gravity，估计 covariance、cross-axis 和 delay；
- 建立同一 raw-data schema 和时间戳对齐。

### H1：12 条低力 commissioning scan

- 六个物理方向，每个方向一个内/外径向点；
- 只验证 15 deg scan 可完成、回正、无 hard stop、phase 顺序正确；
- guard 先 shadow mode，不允许自动 transport。

### H2：24 条最小适配集

- $6$ 个方向 $\times3$ 个径向带 = 18 条 D-optimal anchor；
- 另加 6 条 boundary/day repeat 测重复性；
- 冻结 CNN backbone，优先只调输入 affine normalization、temperature、small adapter/head。

### H3：12 条永久 untouched 随机测试

- 从实测视觉 95% ellipse 随机抽点；
- 不进入 fine-tuning、threshold 或 planner 选择；
- 报 APS coverage、方向误差、回正漂移和 hard stop。

只有 H3 失败时，才从高 entropy、APS disagreement 或 action-boundary 中主动补 12 条；不做均匀大网格。首轮接触数据总量约 48 条，其中只有 24 条用于适配。

M10/M12 每个尺寸先做 18 structured zero-shot + 12 untouched random。失败后再加 12--24 条 calibration；若要发表“跨尺寸泛化”，每个尺寸仍需独立真机 test，不能把 calibration 当 test。

## 11. 冻结后的论文故事

一句话贡献应是：

```text
我们用一次相位结构化的低力触觉探测，把视觉残余下的二维盲搜压缩成
带预测集合的方向捷径；视觉/CAD 界定 support，有限宽度搜索处理距离误差，经典柔顺与
能量/CAD guard 独立约束执行风险，并提供具名 safe-abort/hard-stop 出口。
```

可主张：

- M8 仿真中 probe 对中心方向具有可用可观测性；
- direction posterior + soft CAD 能缩小 residual search region；
- Proposed 相对普通 outward 在配对条件下减少路径/接触功；
- entropy-grid 是强透明诊断，不能因为结果较强而隐藏或降格成弱基线；
- direct-first 可以更快，但只回答第一次 ray 的几何机制，不是物理 retry baseline；
- post-capture guard 展示 CAD-depth 防御取舍。

不可主张：

- CNN 找到了真实中心或精确距离；
- Proposed 全面优于 direct-first、historical direct/retry 或 entropy-grid；
- random C 在任何单个 seed 上都必然比 Proposed 慢或更差；
- 当前 MuJoCo 结果证明 wear、passivity、真机鲁棒性或 M8-to-M12 泛化；
- CAD bottom verified 等于完成拧松。

## 12. 最后判断

方案已经足够清楚，可以停止继续增加算法模块。Strict 仿真闭环应在冻结后由独立 seed 复验，同时转入真机低成本 commissioning。下一阶段的主要矛盾不是再换网络，而是：

1. 用实测视觉/F/T 分布替代工程先验；
2. 验证 15 deg probe 在 iiwa + adapter 上可回正且不伤工件；
3. 在 shadow mode 证明 posterior transport 不增加 capture loss；
4. 在 iiwa 上标定并冻结 dual-channel geometric terminal verifier；
5. 最后再接工具旋转和 first-turn coupling verification。
