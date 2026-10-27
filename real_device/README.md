# M12 真机事件触发闭环与锥形采样

这套代码把真机工作拆成两条互不混淆的工作流：M12 full-loop，以及按当前安装
peg 尺寸独立落盘的 M12/M10/M8 26 点采样与续采入口：

- **任务闭环**：触面、主动触觉观测/搜索、物理 capture、插入和独立 terminal
  verification；
- **commissioning 采样**：以确认后的启动 TCP XY 为本轮当前尺寸孔心，按预注册局部
  偏移采 26 个锥形指纹并保存真机原始证据；不加载模型、不在线训练，也不授予
  full-loop 权限。

当前三尺寸工具已经各完成一轮 26 点 fresh campaign；M12 又完成一轮正确程序的
完整采集。CAD `-15 deg` shape rotation 已烘焙进 v1.5 实体几何，不能在仿真、训练
或 runtime 再补一次。当前安装 yaw 仍是 operator-taught fixture `0 deg (mod 60 deg)`
假设，不冒充 as-built yaw metrology。当前
[`fixture_frame_commissioning_v1.json`](configs/fixture_frame_commissioning_v1.json)
故意保持 `PENDING/null/authorization=false`。四个 full-loop 仍按各自的 commissioning
边界执行；M12 采样入口明确使用 `OPERATOR_TAUGHT_RUN_LOCAL_M12_CENTER`，不把
手动孔心冒充 as-built 计量。所有入口仍保留 KST/Bota 新鲜度、命令所有者和硬保护
检查。若操作者已经把 MatlabToolboxServer 置为 Running，启动期异常清理只会
写入 ASCII `end` 退出应用，不会创建位姿、速度或力命令。静态通过不等于已经完成
新打印件的真机实验。

## 坐标和重新打印合同

fixture 坐标原点是 M10 孔心，`+X_F` 从 M10 指向 M12，`+Z_F` 为底座上表面法向；
照片从左到右是 M12、M10、M8，因此图像横向 `u=-x_F`。在从 `+Z_F`
俯视的数学绘图坐标中，右手系要求 `+Y_F` 指向图下方，即
`(u_right,v_up)=(-x_F,-y_F)`；若采用像素向下为正的相机坐标，则
`v_pixel_down=+y_F`。详细推导见
[`FIXTURE_TOP_VIEW_FRAME_AUDIT.md`](docs/FIXTURE_TOP_VIEW_FRAME_AUDIT.md)。

选定 peg flat 的法向、法兰 A/C 切向基准面的法向和 `+X_F` 对齐。fixture 长边侧面
跨越 `X_F-Z_F`，所以它和该 peg/法兰平面垂直；竖直 peg 棱只能辅助检查轴向，不能
唯一确定六角 yaw。控制最终使用实测完整 `WORLD <- fixture` 三维旋转，而不把照片
中的平行关系当作标定。

CAD 的 `-15 deg` 是 peg 相对法兰的构造旋转。socket yaw 和 canonical MuJoCo
peg/socket 相对 yaw 仍为 `0 deg mod 60 deg`，训练标签也不旋转。名义 CAD 的长度、
体积、质心和正六角面内惯性不因该中心旋转改变；但每次 FDM 重打仍须重新记录实体
实例、材料/填充、质量、CoM/惯量、42 mm 局部 tip 高度、径向跳动和低载荷复验。
完整 media-flange→tip TCP 必须另做 pivot/touch-off；不能把粉色件 42 mm 或当前约
92±4 mm 的尺量估计当成 certified 锥扫标定。当前 M12 采样会把 92 mm axial TCP
明确记为 run-local acquisition 假设，不提升其计量等级。

## 主入口

| 入口 | 方法 | 锥形指纹 | 初始输送 / 搜索 | 学习角色 |
|---|---|---:|---|---|
| `run_m12_full_loop_proposed_plus_entropy_grid` | Proposed+ | 1 次 | 学习 posterior × CAD 捷径 + TOP1 中心 80% corridor（45° 内不足则跳过）+ 95% credible residual EIG | `real_multisize_direction_v1` 获 M12 高层控权 |
| `run_m12_full_loop_proposed_local_spiral` | Proposed | 1 次 | 同一学习/CAD 捷径 + measured-endpoint posterior-sized local spiral | 同上 |
| `run_m12_full_loop_baseline_pure_outward_spiral` | Pure Outward | 0 次 | 从 touchdown 直接 outward spiral | 不调用模型 |
| `run_m12_full_loop_baseline_improved_direct_x3` | Improved Direct ×3 | 每次重扫，严格最多 3 次 | 每次新模型方向单 ray；垂向 `0,+0.25,-0.25 mm` anchor；三次 clean miss 后在当前位置正常结束，不重开 x3 | 每次锥扫后重新推理 |
| `run_m12_sampling_conical_fingerprints_26pt` | M12 数据采样 | 26 点各 1 次 | run-local 孔心；由打印 CAD 生成受约束短路线；不做 full loop | 不加载模型 |
| `run_m12_sampling_conical_fingerprints_26pt_resume(parentRunDir)` | M12 断点续采 | 只采父 run 中尚无完整锥扫证据的点 | 保留父孔心与 dataset lineage；child run 独立落盘 | 不加载模型 |
| `run_m10_sampling_conical_fingerprints_26pt` | M10 数据采样 | 26 点各 1 次 | M10 打印 CAD 独立点位/路线、逐点新鲜重力基线 | 不加载模型 |
| `run_m10_sampling_conical_fingerprints_26pt_resume` | M10 自动断点续采 | 只采 M10 lineage 中尚无完整锥扫证据的点 | 严格限制 M10 prefix、campaign、plan 和 peg STEP hash | 不加载模型 |
| `run_m8_sampling_conical_fingerprints_26pt` | M8 数据采样 | 26 点各 1 次 | M8 打印 CAD 独立点位/路线、逐点新鲜重力基线 | 不加载模型 |
| `run_m8_sampling_conical_fingerprints_26pt_resume` | M8 自动断点续采 | 只采 M8 lineage 中尚无完整锥扫证据的点 | 严格限制 M8 prefix、campaign、plan 和 peg STEP hash | 不加载模型 |

四个 full-loop 与三个 fresh-sampling 文件均为无参数入口；M12/M10/M8 续采可以无参数
自动选择严格匹配本尺寸 campaign/CAD lineage 的最新断点，也都保留显式父目录接口。
共享实现集中在
`execute_m12_contact_engagement_policy.m`，因此四种 full-loop 只替换 routing/search，
不会复制或漂移 touchdown、低力状态机、hard guard、capture、insertion 与 terminal。
旧文件名已移至
`archive/legacy_entry_aliases_20260724/`，只作 provenance；新实验与结果目录一律
使用上表名称。

当前 Proposed、Proposed+ 和 Improved Direct 使用
`real_multisize_direction_v1` 的三 seed real-first posterior，并要求
`CAPPED_SHORTCUT + hardware_control_sizes=["m12"]`。模型拥有方向、状态 posterior、
CAD-bounded shortcut/corridor/residual target 的高层控制权；capture、插入、terminal
和硬保护仍归 MATLAB 确定性控制器。Pure Outward 不调用模型。

方向路由与训练准入分开：弱但完整、新鲜、相位覆盖充分的 360° 指纹仍发出连续缩短的
有界动作；低 quality、APS 大或多峰只会缩短 shortcut、放宽 posterior/后续 CAD 搜索，
不会成为 validity veto。只有 incomplete、stale、nonfinite、frame mismatch、硬保护或
锥扫阶段断裂才禁止模型动作。模型推理失败会保存 checkpoint 并停止当前方法，不会静默
换成经典 heuristic 或 Pure Outward。使用锥扫的 full-loop 在连接 KST 前先用固定、
哈希绑定的 M12 trace 完成一次 venv→manifest/interface/promotion→normalization→三 seed
端到端 preflight；要求 `<3 s`，最终实测约 `1.26 s`。

## 任务闭环

四种方法只改变“如何从第一次触面位置到达可捕获区”；公共闭环是：

```text
fresh stationary local start + unloaded baseline
→ guarded WORLD -Z touchdown
→ 方法特定的有界 routing/search
→ sensor-only downward-progress capture candidate
→ shared capture requalification
→ upright 0.1 mm segmented insertion
→ measured depth band + persistent bottom-force event
→ KST end at bottom；异常则停止继续下发并显式 end
```

所有搜索都有统一的活动时间/动作半径边界。模型只提供状态/方向 posterior、APS、
ensemble disagreement 和可弃权 shortcut；它不判定 capture、到底或成功。插入
`2.5 N` 持续接触只表示倒角/孔壁导向，只有进入由实体 M12 top-to-bottom 深度标定
生成的候选深度带并满足持续 bottom event，才是 provisional terminal candidate。

当前 Bota WORLD* 仍是相对 unloaded baseline 的 commissioning 信号：力向量按姿态
旋转，尚未完成 measurement-origin 到 tip 的力矩平移、完整 payload/CoM/惯量重力
补偿或认证测量精度。因此真机 terminal 仍须与实体孔深、承扭/首圈耦合分开报告。

### 2026-07-27 历史 fast-but-bounded 配置（已被 2026-07-28 学习闭环替代）

以下条目保留作演化留痕，不能作为当前参数合同。当前 authoritative 合同见
[`REAL_DEVICE_TRACE.md`](REAL_DEVICE_TRACE.md) 的
“2026-07-28 — 23:03 real-first 多尺寸学习、CAD clocking 与 M12 模型控权闭环”；
尤其旧 classical/shadow、固定 `±12°` corridor、`0/-45/+45°` fan、固定 2 mm/s
和旧 shortcut 数值均已失效。

四个 full-loop 现在统一采用以下现场修复，wrapper 本身不各自漂移：

- 人工 air-start XY 只保留作审计；touchdown 每一步监测相对 air-start 的最大横向偏移。
  `>0.20 mm` 记录预警但继续，只有最大偏移 `>1.00 mm` 才在搜索前终止。这个有界放宽覆盖了同一
  KST 链路历史成功触地时约 `0.695–0.912 mm` 的多段耦合漂移，同时保留每步
  `0.30 mm` cross-track 门。通过后，Pure spiral、Proposed shortcut/grid/local
  spiral 和 Direct reset/ray 全部以**实测 touchdown XY**为锚；
- 锥扫低力 hold 需要同时满足低法向力和低总相对力。以 3.5 N 目标为例，名义门限为
  `normal <=0.875 N AND total <=1.75 N`，不再让仍有 3–5 N 横向接触载荷的样本反复
  进入/退出 hold；RAMP 与 SCAN episode 分别计数，前者恢复后仍可保留 SCAN routing，
  但任一 episode 都继续隔离训练；
- 锥扫力跟随仍以 `3.5 N` 为目标，但 deadband 从 `0.75` 收紧到 `0.50 N`，
  接触 admittance 从 `0.15` 提到 `0.20 mm/(N·s)`，最大下压速度/残差从
  `0.50 mm/s, 0.50 mm` 提到 `0.70 mm/s, 0.70 mm`。这是为了让真实指纹更接近训练
  接触工况；`12/9 N` 软卸载滞环保持不变，`≥18 N` 时冻结当前锥扫相位并立即增加
  一次 `+0.10 mm` 卸载，任务硬停止统一为 `25 N / 0.75 Nm`。发生过软卸载但完整
  扫完的指纹继续保存和参与 routing，标成 weak/pending offline QC，不直接进入训练；
- 表面 endpoint tolerance 为 `0.12 mm`。capture 检查先执行；到达 XY 后优先用最多
  3 次量化 Z 修正把法向力收敛到 `4.0±1.0 N`，不再把 `1.75–7.5 N` 宽带直接冒充
  4 N setpoint。只有校正量化、Z 停滞或有限 settle 次数耗尽后，且总 wrench 仍为
  新鲜接触并低于 `10 N / 0.30 Nm` soft-relief 门，才接受宽 continuation band 作为
  负观测继续搜索；
- 每个表面目标的动作上限由几何距离换算，再加 8 个恢复动作；XY/Z stagnation 和
  endpoint force-settle 连续上限从 5/5/12 收紧到 3/3/3。Z-only 修正累计横向漂移
  `>0.15 mm` 会终止；
- Pure 和 Proposed-local 共用的 Archimedean pitch 从 `0.40` 调到
  `0.50 mm/revolution`，路径弦长从 `0.15` 增到 `0.30 mm`；Pure 与 Proposed-local
  现在都覆盖到 `4.2 mm` 半径，离线几何均为 463 段。Proposed-local 不再因旧
  `2.0 mm` 残差半径在几何上漏掉较远捕获区；
- Proposed+ 的网格不是旧的固定 `0.55 mm / 2.0 mm / 25 actions`。当前由 M12 CAD
  尺寸显式推导：tip face AF `8.0 mm`、shaft AF `10.0 mm`、socket wall/mouth AF
  `10.6/13.0 mm`，得到 `0.50 mm` pitch、`2.50 mm` radius、`0.75 mm` 保守
  capture-likelihood radius，并完整枚举 81 个节点。合格锥扫先冻结一个方向，在
  `±12°`、CAD 圆盘裁剪后的中心优先蛇形 corridor 内连续覆盖；lane spacing 由
  `0.50 mm` 网格密度和扣除 `0.20 mm` 跟踪余量后的捕获半径共同限定。corridor
  真正完成后，才让完整 CAD 圆盘上的残余 EIG 接手。旧的“离当前点超过 1.20 mm
  就不可达”上层 veto 已删除；任意较远目标都由现有表面控制器自动拆成
  `0.30 mm XY / 0.08 mm Z` 的新鲜 Bota 观测微步。捷径范围提高到
  `0.75–1.50 mm`，弱指纹按质量连续缩短而非归零。若方向弃权则直接使用均匀
  CAD-bounded prior。当前 shadow CNN posterior 仍不接控制；
- Improved Direct 不再把已经走满的 4.2 mm 错向 ray 继续加长，而是每次独立重扫后
  执行透明 `0/-45/+45 deg` 扇形，并用 `0,+0.75,-0.75 mm` 横向锚点扩大覆盖；
- 表面 XY 请求速度统一为 `2.0 mm/s`，endpoint 分段与 spiral 弦长统一为
  `0.30 mm`；touchdown 仍为 `1.0 mm/s`，insertion 仍为
  `0.60 mm/s`。独立的 `25 N / 0.75 Nm` hard guard、`0.25/0.04/0.10 mm`
  capture 证据、`0.08 mm` 最大 Z 修正和 `0.10 mm` insertion 分段没有放宽。

冻结模型现在通过 `model_runtime/.venv/bin/python` 调用，并在 KST 连接前完成固定
trace 的端到端推理 preflight，避免系统
`/usr/bin/python3` 缺少 `torch` 导致在归一化和 CNN 之前失败。运行环境恢复不等于
控制放权：最新同位置真机指纹虽然重复性较高，但相对仿真训练 context 严重 OOD，
三模型 disagreement / simulation transport gate 也没有通过；fixture/WORLD 与
manifest 所写 upright-TCP adapter 的语义仍待带 CAD 真值的多位置真机样本标定。

实时轨迹图采用正交 `view(135,24)`：`+X_W` 左下、`+Y_W` 右下、`+Z_W` 向上，只保留
3 mm WORLD 三轴，不再绘制黑色 25 mm peg 轴。该改动只修正显示投影，不把照片当成
fixture/WORLD 标定。

实时图只用于监看，采用 deferred/no-blocking render，视觉上可能落后于机器人与传感器。
可用于科学分析的指纹是带时间戳的 `conical_directservo_trace.csv`、wrench/command/
state CSV 和 manifest，而不是屏幕动画本身。

带锥扫的 Proposed+、Proposed-local 和 Improved Direct x3 在每次 DirectServo
完整停止且模型推理返回后，额外批量更新一个 learned-decision 窗口；锥扫流内仍然
没有 `drawnow`、timer、FFT、第二个 subscriber、KST query 或额外文件写入。
窗口把实际获得控制权的 48-bin 多峰方向 posterior、离散 APS classes、TOP1 控制
方向、circular mean 诊断方向、三状态 posterior、CAD grid posterior、可信 corridor
和实际 shortcut/direct ray 同时画出，并列出 q95、capture margin、distance cap 与
最终命令距离。Pure Outward 按 baseline 定义不锥扫、不调用模型，因此不伪造学习
方向图。

决策输入仍是冻结 bundle 规定的原始七通道相位序列
`Fx,Fy,Fz,Tx,Ty,Tz,cmdZrel`；不会为了让曲线好看而改写模型输入。显示副本才使用
零相位的 1--6 阶圆周 Fourier 重构，并把原始曲线与高频残差的 robust MAD 带一起
保留。该阴影明确标为 `DISPLAY ONLY / NOT MODEL CI`：它只表示显示残差，不是
传感器置信区间，更不能替代方向 posterior/APS。冻结的
`REAL_FOURIER_LINEAR_V1` 本来就只使用 1--6 阶圆周谐波，所以当前没有必要在
在线控制链前再叠加一个会改变训练/部署分布的时域滤波器。

2026-07-29 的两次 Proposed-local 现场 run 暴露了接触卸载与自由空间 tracking
语义混用：第一次 0.5 mm shortcut 已完成后，一次 `+0.10 mm` WORLD-Z 软卸载
产生 `0.1517 mm` 横向弹性释放与 `0.2217 mm` 三维端点误差；第二次模型已将
shortcut 增至 `0.969 mm`，但峰值仅 `8.107 N / 0.4182 Nm` 时又发生第二次软
卸载，其实测位移 `[+0.3044,+0.0009,+0.2829] mm` 被 `0.35 mm` 自由空间式
tracking 门误停。

现在 soft torque trigger/resume 使用 `0.50/0.40 Nm` 双迟滞，`0.4182 Nm`
不会再产生无谓清载；谨慎速度仍从 `0.24 Nm` 起生效。真正触发的接触清载显式
使用 `CONTACT_RELIEF_MEASURED_REBASE`：`0.35 mm` 只保留为诊断阈值，控制器
在有限 `0.60 mm` 实测单步包络内接受精确 KST 端点，并立即复核 fresh Bota、
表面 Z 区间及 `0.60 mm` 累计横向释放预算，再从实测端点继续。普通表面、自由
空间、touchdown、capture confirmation 和 insertion 仍使用严格 tracking/progress/
cross-track 规则；`25 N / 0.75 Nm` 硬保护不变。同一多峰 posterior 的保守投影
为零时，CAD fallback 不再总取最近 0.5 mm 节点，而是在 TOP1 正向且不小于质量
相关 distance cap 的 75% 区域选择最高 posterior 节点；它仍受 q95、capture
margin、模型质量和 3 mm 总上限共同约束。

学习决策窗口中的三项状态名必须作为一个 scalar-cell 写入显示数据；裸的
三元素 cell 会让 MATLAB 把 `data` 隐式扩成 `1x3 struct`，随后在
`data.payload.cfg` 处报逗号分隔列表错误。当前窗口已固定保持 scalar struct；
该 best-effort 显示错误本身从不影响控制。

## M12 26 点实际打印 CAD 采样设计

M12 采样不再读取旧的 `fixture_multisize_conical_sampling_v1.json` 双环方案。活动计划、
点表和俯视图统一由一个离线生成器产生：

- [`m12_printed_cad_conical_sampling_v2.json`](configs/m12_printed_cad_conical_sampling_v2.json)
- [`m12_printed_cad_sampling_points.csv`](docs/m12_printed_cad_sampling_points.csv)
- [`m12_printed_cad_sampling_top_view.png`](docs/m12_printed_cad_sampling_top_view.png)
- [`m12_printed_cad_sampling_top_view.svg`](docs/m12_printed_cad_sampling_top_view.svg)
- M12 历史/复现实验图 [`m12采样规划图.png`](m12采样规划图.png)

## M10 26 点实际打印 CAD 采样

M10 复用已经真机完成 M12 26/26 的动作状态机、逐点原子 checkpoint、completion-first
策略和自动续采，但 CAD、数据目录、campaign ID、peg hash 与顶视图身份完全独立。
实际打印几何为 `D18.975 / mouth AF10.54 / wall AF8.54 / tip AF6.4 /
shaft AF8.0 / bore D0.5`，计划闭环路线为 `97.999 mm`，覆盖 supported search、
double overhang 和真正的 outer partial support。当前操作员图是
[`采样规划图.png`](采样规划图.png)，M10 canonical 产物为：

- [`m10_printed_cad_conical_sampling_v2.json`](configs/m10_printed_cad_conical_sampling_v2.json)
- [`m10_printed_cad_sampling_points.csv`](docs/m10_printed_cad_sampling_points.csv)
- [`m10_printed_cad_sampling_top_view.png`](docs/m10_printed_cad_sampling_top_view.png)
- [`m10_printed_cad_sampling_top_view.svg`](docs/m10_printed_cad_sampling_top_view.svg)

M10 peg 已安装时，先把直立 TCP 放在 M10 孔心上方一点点且不接触，然后运行：

```matlab
run_m10_sampling_conical_fingerprints_26pt
```

若中断，保持同一安装/fixture 后可无参数自动续采：

```matlab
run_m10_sampling_conical_fingerprints_26pt_resume
```

当前 M10 工具在 Bota 下方的重力不复用 M12 tare；启动时和每一点位都重新建立空气
baseline，原始六轴信号照常保存。生成/核验 M10 计划：

```bash
python3 tools/generate_m10_printed_cad_sampling_plan.py
python3 tools/generate_m10_printed_cad_sampling_plan.py --check
```

### M12 方案详细语义（续）

几何只来自实际送去打印的 v1.5 CAD 链：

- [`design_parameters.json`](model/parametric_tooling/design_parameters.json)
- [`generated/manifest.json`](model/parametric_tooling/generated/manifest.json)
- [`integrated_flange_peg_m12.step`](model/parametric_tooling/generated/integrated_flange_peg_m12.step)
- [`socket_base_hardware_easy_v2.step`](model/parametric_tooling/generated/socket_base_hardware_easy_v2.step)

生成与加载时会核对 design revision、manifest 和两个 STEP 的 SHA-256；MuJoCo 配置、
旧模型和 `visual_offset_sweep` 都不是点位几何输入。这里的“实际打印 CAD”指打印件的
设计权威，不冒充 as-built 计量；每件实体仍须记录实例 ID、尺寸检查、run-local 孔心
和 peg/fixture 相对 yaw。

三种接触状态由**直立 touchdown 时实际 AF8 tip-face 六角形**与
`D21.275` head 外圆、`AF13` mouth 的多边形面积关系定义：

- `supported_search`：AF8 footprint 不越过 head 外圆；它可以覆盖内孔，不能解释成
  “整个 tip 都落在实体环带上”；
- `double_overhang`：footprint 同时越过 head 外圆并覆盖 mouth；
- `outer_partial_support`：footprint 越过 head 外圆、已离开 mouth，但仍在 head 顶面
  保留非零支撑面积。

每点保存 head-top support、mouth overlap、outer overhang 三个面积比例和
`±1 deg` peg-yaw 包络。直立 AF8 状态标签不替代完整锥扫诊断：`15 deg` 锥扫还要按
打印件从 AF8 端面、`1 mm / 45 deg` peg 倒角到 AF10 杆体的完整几何，按
`Rz(-φ)Ry(15°)Rz(φ)` 在 360 个相位逐相位求精确凸包投影
和接触模式迁移。锥扫可以跨越 touchdown 状态边界；这些迁移、low-force、capture、
相位覆盖和时序完整性由真机 trace 另行记录，不能用 AF10 外接圆反过来改写 AF8
touchdown 标签。outer clean 同时覆盖接近落点边界和完整倒角扫掠仍与 mouth 分离的
诊断位置。

每轮仍为 26 次且没有中心方向样本：

- 16 个 `clean_candidate`：6 个 supported、6 个 double-overhang、4 个
  outer-partial；完整 trace 后也只进入 `PENDING_OFFLINE_QC`；
- 4 个 untouched `holdout`：永不参与训练、归一化或调参；
- 4 个 `quarantine`：分别覆盖 supported/double 与 double/outer 两条 CAD 边界；
- 2 个 `sentinel`：稳定 clean 点的精确重复，执行位置与原样本至少间隔 10 次，只作
  漂移和重复性检查。

路线和横向包络不再复制旧的固定索引或固定 `78.49 mm` 常数。生成器根据当前 26 点
确定受 sentinel 间隔约束的短闭环，把执行顺序和闭环距离写入 `execution`；MATLAB
加载后校验并直接使用。采样专用横向包络由最大计划半径、`0.25 mm` 落点容差、
`0.20 mm` 空中跟踪容差和额外 workspace margin 动态生成，只在
`DATASET_M12_CONICAL_FINGERPRINTS_26PT` 分支扩展，不改变四个 full-loop 的包络。

M12 campaign 每点执行：

```text
确认后的启动 TCP XY 作为本轮 M12 孔心，WORLD X/Y 作为 fixture X/Y
→ 首次离开孔心仍抬升 2.0 mm；每个采样点结束后先局部抬 0.75 mm 并立刻检查清载，未清载才每次增加 0.25 mm，最高不超过启动 Z + 3.0 mm
→ 按 v2 计划内生成并校验的受约束短路线，以 1.50 mm 分段、8.0 mm/s 在实测清载高度移动
→ 采 30 个新鲜点位空气 baseline
→ 锁定注册 XY，0.10 mm @ 1.0 mm/s guarded touchdown
→ baseline、未触地或落点超差可在清载回撤后原点重试，单点最多 3 次
→ 接触确认后保持位姿不动，只读至少 0.60 s 的新鲜 Bota 接触数据
→ 3.5±0.5 N、至少 0.5 s / 20 点且 std≤0.20 N 只作为优选质量等级
→ 任何未达到优选载荷的 fresh、未越 hard guard 接触都明确标记后继续锥扫
→ 完整 15 deg、40 Hz、6 s conical fingerprint；流内冻结界面
→ 立即回正、适应式清载回撤、两轮 wrench-clear
→ DirectServo 停止后批量保存 raw trace、baseline、settle 和质量状态
→ 下一点
→ 仅在状态已知且 clear 时返回初始非接触位姿
```

采样 touchdown 的 endpoint error `≤0.08 mm` 立即接受；`0.08–0.15 mm`
进入两次、每次间隔 `0.08 s` 的纯观测重读，不补发运动；实测 XY 仍另受
`0.25 mm` 落点门约束。touchdown 使用有限步数、每步实测 3D endpoint 和 fresh
Bota contact event，不再用单步 Z 进展阈值或通用三维向量比例门重复否决；其他运动
仍保留通用比例门。这样不会再因 KST 读取时序或亚分辨率位姿量化直接终止。

touchdown 后的 pre-cone 阶段已改为纯被动观测：不发 PTP、Z 微步、XY 纠偏或额外
KST pose query，只读取 fresh Bota，并逐样本保留 `25 N / 0.75 Nm` hard guard。
优选窗口写为 `PASS_PREFERRED_STABLE_LOAD`；其余载荷写为
`PASS_COMPLETION_FIRST_CONTACT_OBSERVED / NONPREFERRED_LOAD_PENDING_OFFLINE_QC`。
均值、标准差、最小/最大值、观测时长和 completion-first 标记写入逐点
JSON/MAT/CSV。软载荷质量不再有权跳过锥扫；锥扫自身的 40 Hz force-following、
low-force reacquisition 和 soft-relief 继续负责动态接触。

每点最多执行 3 次有界 acquisition。空气 baseline 暂时噪声过大时原地重采；
一次未触地或 XY 落点超差时，先清载回撤、重新锚定同一目标并采新的局部 baseline，
再下压。方向标签异常只把样本标为 quarantine，已经确认接触、落点合格且未检测到
capture/bore entry 的点仍执行诊断锥扫。capture/bore、连续三次仍无接触、硬力/力矩、
Bota stale、KST tracking/ACK 或 DirectServo 状态未知不会被重试逻辑吞掉。

锥扫完成和物理回撤现在是两个独立提交边界。DirectServo 停止且完整指纹完成后，
程序先批量导出该点 raw evidence、原子保存 attempt，并原子刷新 campaign checkpoint；
随后才发清载回撤。回撤后的 KST 状态读取即使失败，也不会再把已经完整的指纹误判为
采样失败或要求重扫。KST 客户端只接受精确 6/7 维有限数值回复，拒绝 `nak`、
`interrupted`、半包和错维回复；运动完成 ACK 后先留出协议 settle，若状态回复被
Sunrise 共享命令槽竞争吞掉，只重试 pose/joint **读查询**，绝不重发刚才的运动。

断点续采示例：

```matlab
run_m12_sampling_conical_fingerprints_26pt_resume( ...
    '/home/ubuntu20/robot_ws/group_meeting/real_device/data/M12_conical_fingerprints_26pt_20260728_004338_538')
```

续采会在连接 KST 前验证父 run 状态、计划/CAD/fixture/tool 身份、execution→sample
映射和每个已完成点的 MAT/CSV SHA-256。它创建新的 child run，父目录保持只读；
启动后使用父 run 的 M12 孔心和路线，但使用 child 的新鲜 stationary pose 作为本次
清载高度、Z 预算和最终返回位姿。任何运动前还会比较新鲜空气 baseline 与父 run
最近点位 baseline：小漂移只记录并继续，超过 `2.0 N / 0.10 Nm` 才拒绝。确认清载后，
父 run 中 `cone_completed=true` 的点在 UI 中恢复着色并直接跳过；例如上述 run 会
保留 1--17 点，从执行点 18 开始，不重放第 17 点锥扫。

方向真值由 `本轮 operator-taught M12 孔心 - 实测 touchdown 的 WORLD-aligned
fixture XY` 计算，不使用命令点，也不把启动孔心写成 as-built 计量。
M12 采样里，完整且新鲜、相位覆盖充分的 recovered low-force 指纹保留并标记信号
强度，不再仅因弱信号或旧模型不一致而 quarantine；真正的接触丢失/aperture、
capture、三次后仍落点误差 `>0.25 mm`、no-contact、污染 baseline、
传感器陈旧或锥扫/扶正/回撤不完整才降级。当前 DirectServo 锥扫内没有同步 KST pose
query；流内位姿明确标为 command-propagated，流前/后 KST pose 才标为 measured。
clean 数据只标为 `PENDING_OFFLINE_QC`，绝不在线进入训练；holdout、quarantine 和
sentinel 永不作为独立训练样本。
若 run 根目录存在 `RUN_QUARANTINE.json` 或 `DO_NOT_TRAIN.flag`，数据装载器必须整轮
排除；campaign-level disposition 优先于 attempt 中原有的
`offline_training_candidate=true`，不得仅凭原始 manifest 的 size/partition
重新纳入。原始文件与原始哈希仍保持不变，operator correction 只作为追加 sidecar
留痕。

在 `real_device/` 下生成或核验全部五个同步产物：

```bash
python3 tools/generate_m12_printed_cad_sampling_plan.py
python3 tools/generate_m12_printed_cad_sampling_plan.py --check
```

普通运行会同步写计划 JSON、点表 CSV、docs 下 PNG/SVG 和
`m12采样规划图.png`；根目录 `采样规划图.png` 当前留给正在执行的 M10 采样。
`--check` 会在内存中重新构造计划和图，并逐字节检查这些产物；
任一 CAD hash、点位、路线、图或操作者副本陈旧都会返回失败。

## 现场 commissioning

以下严格计量清单主要面向 full-loop 授权；M12 采样不会把未完成项伪装成已测量：

1. 新工具与 fixture 的物理实例 ID、材料和打印批次；
2. 实测右手正交 `WORLD <- fixture` 旋转和平移、as-built M8/M10/M12 孔心和顶面，
   并把各孔心不确定度压到 `<=0.25 mm`；
3. 实测安装后直立 TCP 旋转，且注册 `+TCP-Z` 与 `+Z_F` 反向；
4. pivot/touch-off 标定完整 media-flange→tip 三维向量，确认残差小于采样预算；
5. M12 STEP 哈希、相对 yaw、42 mm 局部 tip 高度、径向跳动；
6. 完整工具 mass/CoM/inertia、sensor measurement-origin→tip 变换、Bota baseline
   和低载荷锥扫/capture 复验；
7. 实测 M12 孔顶面→底面深度和底面 WORLD 点交叉核对；MuJoCo 6 mm 只作参考；
8. controller deployment / load handling：每次 Sunrise project/application/LoadData
   重新部署都生成新的 deployment UUID；所有 full-loop 路径都须在 KST 前提供
   `<=900 s` 的 running-app attestation，并匹配预登记的 station/IP/application、
   deployment UUID 与导出项目现场重算 SHA。随后二选一：核对非零 LoadData 的
   mass/CoM/inertia readback，或用同一 deployment、最终实体负载完成独立测量的
   representative full-cone tracking。attestation 会在 KST 连接前和首个运动前
   复验，且剩余 freshness 必须覆盖 `600 s` hard task horizon；JSON 容差只能
   收紧代码硬上限，不能放宽；
9. 独立、时间对齐的完整锥扫 measured pose/phase 证据；没有它可采诊断 trace，
   但不能把数据送入训练；
10. 仅对准备执行的入口设置相应 authorization。

电脑侧环境：

```bash
cd /home/ubuntu20/robot_ws
./prepare_matlab_kst_bota.sh start
```

确认 Bota 数据新鲜、没有其他 ROS/机器人命令所有者，再在 SmartPAD 选择
`MatlabToolboxServer` 并点击 `Running`。运行 M12 采样时，人工把直立 tip 放在
M12 孔心正上方的一点点非接触位置；确认后该 TCP XY 就是本轮孔心，fixture XY
假定和 KUKA WORLD XY 对齐。底座若旋转或重装，必须重新摆放并建立新的启动基准，
再按 wrapper 的 F5。调用这个专用采样入口本身就是该次运动授权；只要只读状态、
Bota、唯一命令源、KST 和静止起点检查通过，程序就自动开始，不再等待命令窗口口令。

所有 run 都写入 `data/`，包括 manifest、事件、命令、wrench、锥扫、机器人状态、
summary 和独立 end-state；采样入口还逐点写局部 baseline、contact-settle、完整 cone
CSV/MAT 和文件哈希，不生成模型输入/输出。任何 hard force/torque、stale sensor、
跟踪/KST 异常或未知状态都停止继续下发；未知状态不盲返。

完整主张边界见 [`CLAIM_BOUNDARY.md`](CLAIM_BOUNDARY.md)，逐次变更和验证见
[`REAL_DEVICE_TRACE.md`](REAL_DEVICE_TRACE.md)。
