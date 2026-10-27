# Real-device trace

## 2026-07-19 — MuJoCo → iiwa bridge v1

### 已确认事实

- 真机：KUKA LBR iiwa 14 R820；MatlabToolboxServer/KST 是默认在线命令通道。
- 传感器：Bota BFT-SENS-CAT2-B4 / SN001349，T15 产品量程证据已记录；SN-specific accuracy certificate 仍缺失。
- 操作者定义的折叠初始位姿已经冻结在 `test_real_device/kst_default_folded_start.json`。
- “向下”固定为 SmartPAD WORLD −Z。
- `kst_bota_guarded_world_z_50mm.m` 已实机证明：5 mm/s、0.5 mm 分段、2 N×3 fresh samples 可触发测量深度返回，并发送 KST end。
- `stop_matlabtoolboxserver.m` 已证明可使纯 Running 或 Ctrl-C 后的 MatlabToolboxServer 回到 Selected，并可按记录目标复位。

### 本次工程决策

1. Python/MuJoCo/PyTorch 负责仿真、训练与离线审计；MATLAB/KST 负责真机 FSM、Bota 采集、未来 ONNX inference、命令与 cleanup。
2. 学习器只占窄接口；经典控制负责 touchdown、probe 执行、搜索覆盖、capture 后柔顺、恢复和 terminal verifier。
3. 先完成 classical outward-search baseline，再开放 posterior shadow，最后才开放有界 shortcut。
4. `hardware_easy` socket 明确标成 intentional engineering bias，并要求 matched simulation 和 strict-profile 对照。
5. 规则、deadband、dwell 和 recovery 可以帮助任务做成，但不得改变 hard guard，也不得把 retry 伪装成独立 trial。
6. 当前 claim 冻结为 geometric engagement；motor coupling 与真正 unscrewing 独立排期。

### 新增资产

- `configs/unscrewing_real_v1.json`
- `run_unscrewing_real.m`
- `unscrewing_real_preflight.m`
- `unscrewing_contact_fsm_step.m`
- `tests/test_unscrewing_contact_fsm.m`
- `CLAIM_BOUNDARY.md`
- `README.md`

### 当前 gate

H0 静态/传感器无运动检查可执行。H1+ 在以下项目完成前保持未 armed：完整 wrench metrology；tool payload；fixture/WORLD、approach 与 depth datum；真实频率/延迟；task-frame guards；便携 ONNX contract。

### 验证留痕

本次只执行静态检查与离线测试；没有初始化 KST、没有连接 MatlabToolboxServer、没有发送任何机器人运动命令。测试输出和生成的 H0 summary 应存入本目录 `data/` 的新时间戳目录，不覆盖既有证据。

### 已接收的现场成功证据

操作者完成的最终 guarded 运行记录为：

`test_real_device/data/kst_bota_guarded_z_down50_20260719_043550_448/`

- `success=true`
- 命令 WORLD −Z 8.000 mm，实测下降 8.257 mm
- 相对基线力触发 12.535 N
- `sensor_control_observed=true`
- 触发后完成测量深度 WORLD +Z 返回，最终平移误差 0.123 mm
- `exit_reason=CONTACT_TRIGGERED_CONTROLLED_RETURN`
- `kst_end_request_sent=true`

summary 中 `smartpad_exit_verified=false`，因为这一字段没有自动读取 SmartPAD 状态；操作者随后报告现场结果达到要求。机器记录与人类观察保持分列，不把口头确认回写成原始 summary 字段。

### 本次静态验证结果

- MATLAB R2026a `checkcode`：4 个新增 `.m` 文件均为 0 issue。
- `test_unscrewing_contact_fsm`：shortcut、abstention、residual search、recovery、hard guard、active-time budget 和 KST end 路径全部通过。
- H0 静态 preflight：通过并列出 10 个 H1+ blockers；确认 `motion_commanded=false`、`kst_opened=false`、`ready_for_motion=false`。
- 最终 H0 记录：`data/H0_preflight_static_20260719_050807_999/`。
- skill `quick_validate.py`：通过。

## 2026-07-21 — 活动传感器链路切到 USB Binary

- 当前真机桥接不再使用历史 Bota EtherCAT 主机路径。KUKA 独占 USB-C 扩展坞网卡
  `enxf8e43b73f78c` 到 `192.168.1.147`；Bota BFT-SENS-CAT2-B4 / SN001349 独占
  `/dev/serial/by-id/...3885355D3532-if00`，协议 `Bota_Binary @ 2 Mbaud`。
- H0 每次先运行根入口 `detect`，再运行 `start`。前者只读验证 exact USB/KUKA routing；
  后者只启动 PC 侧 ROS sensor/viewer，并拒绝 robot command owner 与陈旧配置混用。
- 真实 USB 采集通过 600 accepted / 0 rejected / 全部 status ok；随后 start/status 分别测得
  约 199.7/200.0 Hz，窗口在桌面可见。证据、哈希和 claim boundary 见
  `docs/setup/evidence/BOTA_USB_BINARY_TOPOLOGY_20260721.md`。
- 活动 launch 的旧 shared-EtherCAT 旁路已封死。根入口现在还核验 driver/viewer 运行时
  config/qualification hash、KUKA MAC/carrier/route、ROS command owner、已知 host command
  process 和已建立 KST 30001 socket。现场曾捕获一个已有 MATLAB KST socket；入口按设计
  fail closed 且没有终止它。连接自行关闭后新 viewer 在
  `artifacts/experiment_preflight/ft_live_20260721_195758/` 成功启动并通过 fingerprint。
  该窗口随后又由 GUI 正常关闭；driver 保持约 200 Hz，最终 status 因没有映射窗口而正确
  返回非零。需要窗口时由操作者显式运行根入口 `start`，status 不改变 GUI 状态。
- 这只解除“当前 H0 数据通道是否真实存在”的疑问，没有解除 H1+ 的计量 gate：SN-specific
  certificate、measurement-origin→tip、完整工具 payload、fixture→WORLD、task-frame guards
  仍缺失。显示零点仍是固定姿态 baseline，不是完整重力补偿。
- 本次没有打开 KST 30001，也没有发送机器人运动指令；managed Bota ROS driver 与 viewer
  留在运行状态供下一步 H0 使用。

## 2026-07-21 — 两次实物打印反馈后的 CAD v1.4

### 现场反馈转化为设计决策

- 第一版 cartridge 肩盘需要支撑并产生明显拉丝；完整法兰本来就可以手动拆换，
  因此退役 receiver、三个 cartridge、U-clip 和 quick-change coupon。
- 新 M8/M10/M12 各自为完整的一体式法兰—peg 单实体，tip 均保持距蓝色
  基准面 42 mm。传感器侧法兰面朝下打印时，根部截面随高度单调缩小，默认
  无需支撑。
- 第二次实物观察确认原 `.98` tip scale 只有 0.06/0.08/0.10 mm 单边退让，
  被 FDM 层纹淹没。真机版改为 45° peg 倒角 0.60/0.80/1.00 mm，tip AF
  4.80/6.40/8.00 mm；没有采用更激进的 0.80/1.00/1.20 mm，以保留 M8
  端面并降低局部崩边/压强风险。
- `socket_base_hardware_easy_v2` 的深壁 AF 为 6.50/8.54/10.60 mm，
  socket 45° 倒角为 0.80/1.00/1.20 mm，外圆放大 1.15 倍；标签改为
  7 mm 粗体、0.8 mm 深凹刻。

### 物理边界

- 对齐六角 yaw、沿平边法向的组合导向余量近似为 1.65/2.07/2.50 mm，
  只是进入倒角导向区的几何估计，不是保证进洞半径。
- 深壁间隙带来约 ±9.75°/±7.59°/±6.63° 的理想居中 yaw 空程；更易
  插入的代价是扭矩接触更晚、方向指纹可能更弱。
- 42 mm 力臂仍存在；5 N 横向力约对应 0.21 N m 弯矩。首次上机必须低速、
  低 Fz、XY/倾角柔顺、横向力矩限幅，并用深度 + 横向 wrench + 有限时间
  和必要的小角度 torque gate 判定终态。
- 所有方法共用相同 easy-v2 工装可做公平真机内部比较，但不得声称严格
  nominal sim-to-real，也不得把机械捕获盆扩大表述成估计器精度提升。

### 产物与验证

- 最终打印件：三个 `integrated_flange_peg_*.step` 和
  `socket_base_hardware_easy_v2.step`；三个 socket coupon 建议先打印。
- v1.3 完整归档在
  `model/parametric_tooling/archive/v1_3_quick_change_2026-07-21/`。
- 九个 active STEP 均回读为单一有效实体；九个 STL 均通过 watertight、
  manifold、orientable、无自交检查。此次只做 CAD/网格验证，没有发送机器人
  命令。

## 2026-07-21 — 无打印件 H0.5 动作调试审计

### 现场与活动链路

- 操作者照片显示机器人位于记录的折叠构型，Bota 与安装转接堆栈仍在法兰上，
  下方有升降平台；打印实验工具和 socket fixture 尚未安装。
- `prepare_matlab_kst_bota.sh detect/status` 实测通过：KUKA 走
  `enxf8e43b73f78c`，Bota SN001349 走 USB Binary `/dev/ttyACM0`，ROS 原始
  wrench 约 200 Hz，viewer 可见，没有 ROS command owner 或已建立 KST 30001 socket。
- 2026-07-21 20:31 的五帧 raw sensor-frame wrench 约为
  `[-1.97, +0.26, -16.65] N` 与 `[-0.014, +0.100, +0.014] Nm`。这只证明 distal
  load 非零；它排除了“真正裸法兰”的说法，但不能替代完整机器人 payload 的质量/CoM
  测量，因为读数不包含传感器本体和 robot-side 转接件，并混有 bias/cable load。

### 新发现的阻断原因

- 活动 KST 1.7 源码 `MatlabToolboxServer.java:544-566` 在每次 attach tool 时把
  `LoadData` 固定为 `MASS=0`、CoM=`[0,0,0]`；源码 SHA-256 为
  `016850f6a10c20d4c755d11aaf79414c1392f5b912cf57d42ee93856c28214ca`。
- MATLAB `KST.None` 只是 0 m flange-height 常数；`KST.m` SHA-256 为
  `b1936d4e2f843b2eece2898f51ddc4921e279cb0fdd97e17cf3fb5538f2f3e85`，它只在
  connect 时发送几何 transform，不发送 payload。
- `PTPmotionClass.java:496-513` 的 WORLD relative `lin()` 只显式调用
  `setCartVelocity`；没有冻结 acceleration/jerk。源码 SHA-256 为
  `d02d958c01531d16e8c3adaed3ae47d0261aadc25e85e95a8e1057493d1649bf`。
  MATLAB client 随后阻塞等待 `done`，所以外层 45 s 计时不能抢占正在执行的 Sunrise move。
- 因此历史 50 mm 成功仍是重要的 connectivity/方向/回位证据，但不是当前安装堆栈
  payload 或 Cartesian acceleration 已合格的证据。H0.5 位移保持 BLOCKED。

### 落地文件与节奏

- 新增 `configs/air_motion_rehearsal_v1.json`：首个候选仅 WORLD `+Z 1 mm → -Z 1 mm`
  @ 5 mm/s；任一 prefix 不低于 fresh start，至少 15° joint margin，最终 1 mm / 0.03 rad
  回位门限。以后按独立 run 晋级 2 mm、X、Y 和几何 search shell。
- 新增 `configs/air_motion_qualification.json`：记录真实安装堆栈、源码哈希、raw wrench
  快照、payload/acceleration/connect-check blocker；不得靠历史成功自动置 true。
- 新增 `kst_air_motion_rehearsal.m` 与 `test_air_motion_rehearsal_contract.m`：
  `preview` 完全 command-free；`connect_check` 只有 connect、10 组/至少 2 s 站稳读数和
  `end`，零次 `move*`；`execute` 当前在 KST 连接前 fail closed。
- 根入口 `run_unscrewing_real` 新增 `air_motion_preview`、`air_connect_check` 和受锁的
  `air_motion_execute`。异常只请求 `end`，不盲返；raw TCP fallback 只允许在本次 KST
  ownership 已建立后使用。

### 验证与动作边界

- 当前配置匹配的最终 preview：
  `data/H0_5_air_preview_20260721_204314_284/`，两段累计轨迹为
  `[0,0,+1] → [0,0,0] mm`，`motion_commanded=false / kst_opened=false`。
- 离线回归随后用正确 execute token 尝试入口，按设计得到
  `AirMotion:MotionQualificationBlocked`；记录与 preview 分开，未初始化 ROS、未连接 KST、
  未发送机器人命令。
- 20:28 的旧六段 2 mm X/Y preview 只保留为设计迭代 provenance，不作为当前 hash 的通过证据。
- 本轮没有执行 `air_connect_check`，因为仍需操作者在现场明确确认 SmartPAD
  MatlabToolboxServer 已 Running；也没有执行任何位移。
- 最终复核还生成 `data/H0_preflight_static_20260721_204401_184/` 和
  `data/H0_preflight_live_sensor_20260721_204402_541/`。后者复用的 Bota run 为
  `test_real_device/data/kst_bota_guarded_z_down50_20260721_204402_756/`：MATLAB 接收率
  50.4 Hz、最大 force-axis std 0.0217 N、最大 torque-axis std 0.00051 Nm；两项 H0
  均明确 `motion_commanded=false / kst_opened=false`。

## 2026-07-21 — translation-only 任务编舞与首次连接尝试

- 操作者明确要求在没有打印件时，把接触、中心、capture 和入洞都作为已知事件，只看
  真机能否按“指纹漏斗 → shortcut → 熵网格 → 倒角 capture → 入洞”的节奏运动。
- 新增 `configs/air_unscrewing_choreography_v1.json`。它不伪造 15° conical contact：先
  WORLD +Z 5 mm 脱离平台，在抬升平面画 0.5 mm translation-only fingerprint ring，
  复现冻结 `sobol_008` 的 3.023 mm shortcut、grid endpoints 与 offline capture XY，
  再在原始 start Z 以上完成 virtual chamfer/insertion，最后回 fresh start。23 段，
  5 mm/s，最大目标半径 6.17 mm，固定姿态，Bota 只记录。
- `air_motion_qualification.json` 保留 payload/CoM、active LoadData 和 acceleration/jerk
  三项 UNVERIFIED，同时基于操作者此前同堆栈 50 mm 往返与 5 mm/s guarded run，只为
  `fixed-orientation / no-contact / no-below-start / radius<=7 mm` 建立公开 conditional
  exception。它不能晋级到接触、姿态、力控或物理 capture claim。
- MATLAB `checkcode` 三文件均为 0 issue；两种 profile preview 和 wrong-token no-KST
  回归通过。choreography preview 为
  `data/H0_5_air_preview_20260721_211211_801/`。
- 21:13 首次 execute 尝试先完成 Bota live preflight（51.8 Hz、force std 0.0247 N、
  torque std 0.00059 Nm），但 `192.168.1.147:30001` 的 KST tcpip/tcpclient 均无法建立。
  记录 `data/H0_5_air_execute_20260721_211302_361/` 明确为
  `kst_opened=false / motion_commanded=false / completed_segments=0`。最可能原因是 SmartPAD
  的 MatlabToolboxServer 当时未处于 Running/监听状态；没有改投 ROSSmartServo，也没有
  在无 ownership 时发送 raw `end`。
- 同步修正异常语义：普通 blocking PTP 没有 host 可抢占 hold，事件名改为
  `ERROR_END_REQUEST_NO_BLIND_RETURN`；连接未建立时跳过 end，而不是对空 connection
  重复调用。

## 2026-07-21 — F5 入口语义修正

- 操作者对参数化底层文件 `kst_air_motion_rehearsal.m` 直接按 F5 后报告机器人未动。
  最新记录 `data/H0_5_air_preview_20260721_212407_742/summary.json` 证明该次按设计运行了
  默认 `mode=preview / profile=axis_1mm`，其中 `commands_robot_motion=false`、
  `kst_opened=false`、`motion_commanded=false`；不是一次失败的真机命令。
- 保留底层文件的 fail-safe preview 默认值，新增零参数、名称明确且注释醒目的
  `run_air_unscrewing_choreography.m`。现场确认 MatlabToolboxServer Running、cell/platform
  净空和急停可达后，对该文件按 F5 等价于显式调用
  `air_task_choreography_execute + CELL_CLEAR_PLATFORM_CLEAR_SMARTPAD_RUNNING`。
- 快捷入口没有绕过执行器：PC/Bota/owner、fresh folded anchor、stationarity、逐段实测、
  正常回 fresh start 和 KST `end` 路径仍全部保留；真实接触、力控与 capture claim 仍未解锁。
- 同次只读复核发现并关闭 KST 握手的窄生命周期窗口：TCP 对象可能已写入 KST handle，
  但 tool-frame acknowledgment 尚未成功。现在该阶段异常会先尝试仅通过已有 KST object
  发送 `end`；未确认 ownership 时禁止 raw TCP fallback。`TFtrans` 明确返回 false 时，
  本地 KST client 已自动发送的 `end` 也会写入本次 run marker/summary。
- 本机 MATLAB R2026a 对 wrapper、根路由和执行器三文件 `checkcode=0`。离线合同回归通过：
  1 mm 与 23 段 profile 都只生成 preview，错误 execute token 在 ROS/KST 前拒绝；记录为
  `data/H0_5_air_preview_20260721_213509_382/` 与
  `data/H0_5_air_preview_20260721_213509_659/`。本轮验证没有连接 KST 或发送机器人命令。

## 2026-07-21 — M12 上表面中心人工示教工作位姿观察

- 操作者在 21:49 照片所示粉色三孔夹具上完成桌面摆放/对准，并明确声明当前工具位置
  “大概是 M12 孔上表面中心”。该声明保存为独立、不可覆盖的
  `data/H0_work_pose_capture_m12_socket_top_approx_20260721_214900_000/work_pose_observation.json`；
  它不是 anchor/home/recovery target，不会覆盖 `test_real_device/kst_default_folded_start.json`。
- 照片不能产生可重放坐标；当前 PC 侧没有 KUKA ROS state，也没有已建立的 KST 30001
  连接。因此观察先记为 `SEMANTIC_RECORDED_NUMERIC_KST_CAPTURE_PENDING`。操作者报告的桌面
  标定被如实保存，但 fixture-to-WORLD 数值变换、打印件精确 revision 与 flange-to-tip
  变换仍标为未机读验证，不能把法兰 XYZ 直接冒充孔中心 XYZ。
- 新增 `capture_m12_surface_center_pose.m`：只在操作者保持该姿态、SmartPAD
  MatlabToolboxServer Running 时由 F5 启动；代码没有 move 调用，读取 10 帧/至少 2 s
  稳定位姿与关节，写入新的 timestamped observation run 并请求 KST `end`；不会把近接触点
  自动晋级为 approach、motion target 或 fixture-to-WORLD 标定。

## 2026-07-21 — M8/M10/M12 共用粗中心与局部闭环 adapter

- 操作者补充声明：保持机器人当前姿态，以桌面记号人工移动并锁紧升降台，可以让
  M8/M10/M12 的孔上表面近似中心依次位于同一个机器人工作位下方；工具尖端可能略高于
  各表面。声明保存到
  `data/H0_work_pose_common_m8_m10_m12_operator_report_20260721_221000_000/work_pose_observation.json`。
- 该点只晋级为 bounded local commissioning 的候选 live-start gate；默认“初始位置”仍是
  固定折叠锚点。桌面记号、已知尺寸、匹配打印工具和 hardware-easy-v2 倒角都是公开的
  task-simplifying engineering bias，不是 fixture-to-WORLD 标定。
- 22:11 先执行了零运动 KST 数值采集。KUKA 网络、Bota USB 200 Hz、ROS master、viewer、
  command-owner 检查均通过，但 MatlabToolboxServer 的 30001 未监听，连接前失败；记录为
  `data/H0_work_pose_capture_m12_socket_top_approx_20260721_221118_685/`，其中
  `kst_opened=false / motion_commanded=false`。没有从照片估计坐标，也没有发送 raw end。
- `capture_m12_surface_center_pose.m` 已保留历史文件名但改成三尺寸共用语义；新增首选入口
  `capture_common_surface_center_pose.m`。成功时才发布
  `configs/common_work_pose_latest.json`；失败 run 不会被自动选作 anchor。
- 新增 `common_work_pose_contract_v1.json`、`common_center_guarded_demo_v1.json`、
  `common_center_demo_plan.m`、`kst_common_center_guarded_demo.m` 和
  `run_common_center_demo.m`。默认 F5 为 preview，完全不初始化 ROS/KST。
- 局部 adapter 的 `touchdown` 路径为：common pose/stationarity gate → Bota startup baseline →
  WORLD +Z 3 mm → clearance baseline 重建 → 尺寸对应实心环面 → WORLD −Z 0.25 mm 阻塞段
  + 每端 3 fresh samples → 测量式回 work pose → object `end`。
- `full` 路径在 fresh surface Z 上依次做 8 点 lift-shift-retouch 指纹、无学习器的透明
  inverse-force direction rule（低置信度弃权、尺寸 cap 0.78/0.95/1.00 mm）、最多 25 点
  有限信息排序网格、相对 surface 0.40/0.50/0.50 mm 的低-wrench capture candidate、
  `+0.10 mm unload → hold → same-depth retouch` 二次确认、0.10 mm 段的 1 mm 浅插入和
  relative-wrench dwell。claim 只到 provisional shallow geometric engagement，不到力控、
  承扭或 unscrewing。
- KST 相对直线调用经源码确认是 blocking；Bota 只在端点可判断。因此命令日志显式记录
  `max_unobserved_mm=0.25/0.10`。它不是实际/认证停止距离；20 N/0.75 Nm 也只沿用现有
  相对实验 halt，绝不表述为传感器 overload 或安全阈值。hard/stale/tracking/KST exception
  只停止继续命令并 object-end，不盲返；只有健康反馈下的普通 task failure 才沿测量路径
  垂直卸载、clearance XY、回 fresh work pose。
- MATLAB R2026a 验证：`common_center_demo_plan` 三尺寸合同测试通过；M8 preview 通过且
  `motion_commanded=false / kst_opened=false`；错误 token 与缺失 numeric common pose 都在
  KST 连接前 fail closed。对应 preview 为
  `data/H1_H3_common_center_m8_preview_20260721_223053_734/`。本轮没有执行任何真机运动。

## 2026-07-21 — 共用中心局部 demo 的首次真机前 fail-closed 加固

- 最终静态审查识别出五个实际风险：默认 M8 可能与已安装尺寸不一致；三尺寸共用 token；
  dwell 只看末尾样本而未锁存早先连续接触；孔内/接触后回撤误用 0.5 mm @ 5 mm/s
  自由空间参数；0.8 mm 全局 tracking tolerance 大于 0.10/0.25 mm 微段，且 full 没有要求
  同尺寸 touchdown 证据。
- 入口现默认 `SIZE="" / MODE="preview" / TOKEN=""`，尺寸未明确时在 ROS/KST 前拒绝；
  M8/M10/M12 使用不同 touchdown/full token。实心环参考中心还必须越过 socket-mouth
  circumradius 至少 0.50 mm，不能只检查 AF/2。
- contact debounce 现在在整个 dwell 内锁存首次连续事件。接触后先用 0.10 mm @ 0.75 mm/s
  的 +Z 微段逐端观察，累计至少 0.30 mm 且连续两个 dwell 清载后才切回 0.5 mm 自由空间
  回撤；hard/stale/diagnostic/tracking 仍不盲返。
- tracking 改为阶段门限：0.35 mm free、0.15 mm touch、0.07 mm insertion，并检查至少 35%
  沿命令方向进展、0.20 mm 横向漂移和相对 live-start 的 12 mm lateral / +5/−4 mm Z
  局部包络。180 s 上限中预留最后 30 s 给健康反馈下的测量式回位。
- 新增 `confirm_common_center_touchdown_selected.m`。full 必须先有同尺寸 touchdown 的
  `task_completed + return_completed + KST end`，再由操作者看到 SmartPAD Selected 后用
  尺寸专属确认词生成 release；release 绑定 config/work-pose/tooling/qualification 和
  touchdown summary 哈希。任一变化都会使 full 在 KST 前拒绝。程序不再仅凭写出 `end`
  就置 `success=true`。
- 23:08 再次只读检查 `192.168.1.147:30001` 仍为 NOT_LISTENING，因此没有重试 KST 位姿
  读取或真机运动。加固后 MATLAB R2026a 的八个相关文件 `checkcode=0`，三尺寸 plan、静态
  fail-closed、原 FSM/空中编舞回归和 M8/M10/M12 no-KST preview 全部通过；显式尺寸 token
  在缺失 numeric work pose 时也于 KST 连接前拒绝。

## 2026-07-21 — 共用中心采集成功与首次 touchdown 启动竞态

- 23:23 操作者在 MatlabToolboxServer Running 下成功完成共用 M8/M10/M12 工作位只读采集：
  法兰 `[472.7828, 0.7667, 562.9786, 3.141390, -0.000953, 3.133175]` mm/rad，关节
  `[-0.000643, 0.1956, 0.000025, -1.5843, 0.0086, 1.3627, -0.0022]` rad；10 帧站稳跨度
  `0.000103 mm / 1.87e-7 rad / 1.92e-7 rad`。不可覆盖观察位于
  `data/H0_work_pose_capture_common_m8_m10_m12_approx_20260721_232301_477/`，并发布为
  `configs/common_work_pose_latest.json`。该次没有 move，KST object `end` 已写出。
- 23:29 M12 preview 正常完成，打印尺寸专属 touchdown token；`motion_commanded=false /`
  `kst_opened=false`。23:32 的空 token 被 gate 在 ROS/KST 前拒绝；随后一次正确 token 又因
  viewer 未运行而在 root status 阻断，仍为零运动。
- viewer 启动后的 23:34 正确 token run
  `data/H1_H3_common_center_m12_touchdown_20260721_233421_330/` 通过 operator gate，但 MATLAB
  新建 ROS 节点在固定 4 s 内没有完成 `/diagnostics` TCPROS 建链；summary 明确
  `kst_opened=false / motion_commanded=false`。现场同期 shell 读取证明 Bota driver 正常：
  exact `BFT-SENS-CAT2-B4:SN001349`、diagnostic Level 0、约 1 Hz，wrench 约 199.9 Hz；因此
  这是 MATLAB subscriber 启动竞态，不是传感器掉线或诊断内容不合格。
- 启动路径现保留全部 fail-closed 条件，只把握手编排改为：`rosshutdown` 后 0.75 s 清理，
  先订阅低频 diagnostics，最多等 12 s 并继续严格检查 name/hardware_id/Level=0/age<=1.5 s，
  再订阅 200 Hz wrench 并在 KST connect 前验证 frame/finite/age<=0.10 s。每个控制样本原有
  diagnostic 检查、所有相对阈值和异常不盲返语义均未放宽；等待时间、尝试数和接收路径
  会写入 `events.csv`。

## 2026-07-22 — M12 touchdown 通过与 full 首次微米级进度误触发

- `data/H1_H3_common_center_m12_touchdown_20260722_005506_579/` 完成了 M12 实心环面
  fresh surface touch、测量式返回和 KST object `end`：relative dF 峰值 3.456 N、dT 峰值
  0.01120 Nm、回位误差 0.522 mm / 0.001092 rad，73 个命令完成。操作者看到 SmartPAD
  `Selected` 后于 01:01 生成同尺寸 full release；该 release 绑定的五项哈希当时一致。
- 首次 full run `data/H1_H3_common_center_m12_full_20260722_010321_982/` 再次通过 Bota、
  KST、共用工作位、baseline 和实心环 surface touch。指纹第一点前，前一条 0.252145 mm
  Z 对齐已把实测高度带到目标附近；重复 Z gate 只剩 `0.008887 mm` 请求。KST 实测沿程
  `0.002 mm`，小于统一的 35% 比例门槛约 0.00311 mm，遂触发
  `CommonDemo:CommandProgress`。端点误差门槛此前已经通过，Bota/force/torque/诊断均健康；
  这是低于有意义分辨尺度的比例判据假阳性，不是碰撞或实际跟踪失控。
- 异常按既定策略没有从边沿盲返：停止后续命令、写出 KST `end`，SmartPAD 回到
  `Selected`。现场照片与记录一致，末端停在 M12 实心环参考点上方约 3 mm；该 run
  `task_completed=false / return_completed=false`，不能作为 full 成功证据。
- 修复明确公开为数值工程死区：长度 `<=0.02 mm` 的残余对齐不发送机器人命令并写
  `MICRO_MOVE_SKIPPED_WITHIN_DEADBAND`；若命令长度不大于其阶段端点容差，端点误差通过后
  不再应用比例进度门槛，并写 `SUB_TOLERANCE_PROGRESS_ACCEPTED`。0.10 mm insertion、
  0.25 mm touch、0.5 mm free-space 段仍需原 35% 进度检查；所有 tracking/cross-track、
  Bota、workspace、joint、time 和异常不盲返门限保持不变。
- 此项策略写入主配置，配置哈希因此变化。旧 M12 full release 被保留为 provenance，
  但会在 KST 连接前因 config hash 不匹配而失效；复位到共用工作位后必须重新完成
  touchdown、人工确认 SmartPAD Selected，并生成新 release，才能再次运行 full。

## 2026-07-23 — 收束为人工近孔起点的单入口 M12 事件链

- 操作者明确改变 commissioning 节奏：先人工移动并锁紧升降台/孔座，让 M12 孔口位于
  peg 下方的近孔区域；不再要求固定 common-work-pose、尺寸 release、多脚本跳转或先去
  远处实心环面。这个 task-local start 只服务本轮，不替换默认折叠初始位，也不声明孔中心
  或 fixture-to-WORLD 标定。
- 审计 `mujoco_unscrewing_probe_new` 后，将任务终点限定为 geometric engagement：
  guarded surface event → 一次完整 probe/fingerprint → bounded shortcut/abstention →
  finite residual search → capture handoff → candidate CAD-bottom dwell。它不包括 first-turn、
  反转、螺距一致性或拔出。MuJoCo 活动 rollout 本身从已给定 ambiguous contact 开始，
  所以真机入口另外补了短 touchdown；MuJoCo contact tag 不进入在线逻辑。
- 审计 `Xue_ws/5.KMPC`、`6.GTpHRI`、`7.DMPC` 后保留了前人的可靠节奏：经典微段控制是
  唯一命令主干；事件层只锁存/切 phase；透明慢层只给有限、有界 shortcut；每阶段有
  budget、deadband、dwell、completion latch 和不可覆盖日志。没有把 NN/RL 放进首轮真机
  command path。
- 新活动文件只有 `run_m12_event_trigger_insert_demo.m`。提交态固定
  `MODE="preview" / TOKEN=""`；execute 把最后一帧稳定 live pose 当本轮起点，不读取旧
  work-pose 或 release。人工起点合同要求一个小空气隙，并让 tip 下方仍有一点实心上表面
  支撑；若当前 XY 在 `1.2 mm` WORLD `-Z` 内没有 `2 N × 3 fresh` 触面事件，程序不猜
  surface Z，测量式回本轮 start 后 `end`。
- 触面后强制完整执行 8 点、半径 `1.85 mm` 的 lift-shift-retouch 圆环，再使用
  inverse-force circular resultant：置信度低则弃权，执行时最多 `1.00 mm`。miss 后在
  `0.55 mm` pitch、`2.00 mm` radius 的有限 M12 网格上按二值 open/blocked likelihood
  最大化 expected information，最多实际探测 25 点；Bayesian 模型只路由任务，不参与
  hard guard，也不是学习结果。
- 所谓“锥形试探”在当前硬件资格下明确实现为 fixed-orientation translation-only proxy。
  真 15° tip-centered nutation 需要已测 tip 虚拟转心、完整工具质量/CoM/惯量和匹配
  Sunrise LoadData；若错误绕法兰旋转，42 mm 杠杆在 15° 下会产生约 11.3 mm 横扫，因此
  本轮没有下发任何姿态变化。
- provisional capture 需要在 fresh surface Z 以下低相对 wrench 前进 `0.50 mm`，随后
  `+0.10 mm unload → hold → -0.10 mm retouch` 仍保持低 wrench。插入保持固定姿态、
  `0.10 mm @ 0.60 mm/s` endpoint-only；只有 `5.7–6.2 mm` 候选深度带与连续 5 个 fresh
  samples `>=2.5 N` 同时出现，才写
  `provisional_m12_bottom_event_in_depth_band_candidate`。6 mm 来自
  `allen_m12_hardware_easy_v2.json`，该文件仍标
  `hardware_aligned_profile_ready=false`，不是实体孔深计量或认证到底。
- 相对 hard guard 仍为 `20 N / 0.75 Nm`，插入 soft guard 为 `8 N / 0.25 Nm`。
  Bota 启动和健康退出都核对 Binary interface、精确 config SHA、app mode/submode、
  zero configured offsets、sensor counter/rate/frame age/new/published/rejected；wrench
  Header.Stamp 只按 ROS host publish time 使用，不冒充传感器采样时钟。
- KST blocking 仍意味着 Bota 只能在每个端点之后判断；`0.10 mm` 是最大未观察命令，
  不是认证停止距离。tracking 比例门只在
  `command distance > endpoint tolerance + 0.02 mm deadband` 时生效，关闭了 07-22 的
  微米级假阳性而不放松端点误差门。自由段、预测/实测局部包络、cross-track、关节裕量、
  240 s 总预算和 35 s 正常回位预算仍独立存在。
- 普通 no-contact/search/capture/terminal miss 在 robot+Bota 健康时做
  contact-scale WORLD `+Z` 卸载、clearance XY、回 fresh start、object `end`。hard/stale/
  tracking/KST exception 不盲返，只请求 `end`。raw TCP `end` 只在本 run 已成功握手、
  object end 又失败时允许；attach reject 不允许 raw fallback。写出 end 与 SmartPAD
  人工看到 `Selected` 继续分开记录。
- 旧 common-center v2 在本轮审计中确认是半集成状态：planner 已读 v2，但 executor/
  confirmer/tests 仍有 v1 字段/哈希，且新的 progress helper 未真正接入；不能再拿旧
  `run_common_center_demo.m` 真机 F5。所有旧 `.m`、旧 tests 和 sync-conflict trace 已完整
  移入 `archive/legacy_multientry_20260723/` 与 `archive/trace_conflicts/`；历史 `data/`
  和 `configs/` 保持原路径不变。
- 初次默认 preview 因 `mfilename('fullpath')` 不带 `.m` 后缀而在 source hash 阶段失败；
  它发生在 ROS/KST/sensor/motion 之前。修为显式 `.m` 后，MATLAB R2026a
  `checkcode=0`。最终 source SHA-256 为
  `c79ab470ad2015f5cd4215cae2dea67a174f8a59fcf2d9c4c3914e0abbb56e7e`；
  14:34 的最终无运动验收为
  `data/M12_event_trigger_insert_preview_20260723_143449_090/`。MATLAB batch 断言通过：
  `motion_commanded=false / kst_opened=false / kst_end_request_sent=false`，
  terminal reason 为 `PREVIEW_COMPLETE_NO_ROS_NO_KST_NO_MOTION`；对应 `summary.json`
  也保存了同一 source hash 与零命令计数。
  本轮没有运行 execute、没有连接 KST、没有发送任何真机运动。

## 2026-07-23 — 人工起点高度放宽为约 20 mm

- 操作者补充：人工把孔座/升降台调到合理近孔位即可，peg 与上表面间隙最多约 2 cm，
  不要求精确保持 1.2 mm。活动入口已将 touchdown 上限从 `1.2 mm` 改为 `20.0 mm`，
  task-local 下行包络同步设为 `20.0 mm`。
- 下探仍保持 `0.10 mm` blocking endpoint 段、`1.0 mm/s` 和每端点 Bota 相对力事件检查；
  因此放宽的是人工起点容差，不是取消接触检测或硬 `20 N / 0.75 Nm` 保护。
- 无触面时最多下探 20 mm 后测量式回本轮 start 并请求 KST `end`；触面后立即进入
  lift-shift-retouch、shortcut/grid、capture 和插入链。当前代码与 README/preview 文案已同步。
- 放宽参数后的当前源码 SHA-256 为
  `c18c1d9a51c414968b5ff1c10be8113c3cc89cc9915f7217c6c57a82825432a7`；最后一次 MATLAB
  `checkcode` 仍为 0。此前 14:46 preview 已确认 `WORLD -Z <= 20.00 mm` 且没有 ROS、KST
  或机器人运动；仅最后一行 preview 文案随后做了同义修正。

## 2026-07-23 — 改为 F5 后一气呵成的 live-direct 主线

- 按操作者要求删除活动入口中的 `MODE`、`TOKEN`、preview、release 和普通健康回位分支。
  `MatlabToolboxServer` 已在 SmartPAD 进入 `Running` 时，对
  `run_m12_event_trigger_insert_demo.m` 按 F5 就直接初始化一次 KST/Bota 并执行完整任务。
- 主线固定为：
  `touchdown → 完整八点锥形指纹 → 计算/执行捷径 → 熵网格 → 倒角捕获 → 插入 → 触底 → end`。
  八点全部执行完才计算捷径；捷径不再因低置信度弃权，而是以 inverse-force resultant
  决定方向，以置信度把距离透明映射到 `0.25–1.00 mm`。若圆形合力恰好抵消，则用最小
  接触响应节点的方向作为确定性 fallback。
- 捷径只改变局部搜索原点，不能直接宣布 capture，也不能跳过熵网格。原八点证据先换算到
  shortcut-endpoint 坐标系以初始化 posterior，随后在 `0.55 mm` pitch、`2.00 mm`
  radius 的有限网格上按 expected information 选点，最多 25 个实际探测。
- touchdown 后的操作 clearance 改为 fresh surface Z 上方 `0.60 mm`，不再每点返回最多
  20 mm 高的人工起点，也不再重复重置 baseline；本轮相对 wrench 始终以动作前的 unloaded
  baseline 为参考。这让动作节奏更接近 `Xue_ws/7.DMPC` 的单次初始化、单主循环、统一收尾。
- 成功终点改为洞底原地结束：在 `5.7–6.2 mm` 带内再次确认连续 5 个 fresh samples
  `>=2.5 N` 后，不发任何返回运动，直接 object `net_turnOffServer()`；操作者看到
  SmartPAD `Selected` 后再人工恢复。touchdown miss、网格耗尽、未触底以及 hard/stale/
  tracking/KST 异常同样只停止未来命令并在实测当前位置请求 `end`，不做盲返。
- 删除“深度带之前首次 2.5 N 持续接触就退出”的旧分支。倒角导向或轻微孔壁摩擦在
  低于 `8 N / 0.25 Nm` insertion soft limit 时只记录并继续 0.10 mm 下行；`2.5 N`
  只有与 5.7–6.2 mm 候选深度带同时成立时才可作为 provisional bottom event。
- 没有删除必要的单命令所有者、Bota 身份/新鲜度、局部 workspace、关节裕量、
  `20 N / 0.75 Nm` hard guard 和 KST blocking 端点观测。总活动时间由带 35 s 回位预留的
  240 s 改成无回位预留的 600 s，以容纳完整八点和最多 25 点网格；WORLD `-Z` 局部包络
  同步改为 28 mm，覆盖最多 20 mm touchdown、0.5 mm capture probe 和约 6 mm 插入，
  不会在接近 20 mm 的合法人工空气隙下把插入本身误拦截。
- 本轮只编辑和静态验证代码，没有调用真机函数、连接 KST 或发送运动命令。
  MATLAB R2026a `checkcode` 返回 `0` issue；活动源码 SHA-256 为
  `b1a6eff5beae0f87447a50b83bd49d5bb96f0667d83bc2156e3f14f87ab1b305`。

## 2026-07-23 — 操作者否决八点平移 proxy，恢复真正 15° tip-pivot 锥形指纹合同

- 操作者明确更正：正式任务必须保留工具轴真实倾斜约 `15°` 并完成一整圈 nutation；
  可以在相位离散、事件切换、重力基线和捷径规则上使用透明工程技巧，但不得再用八点
  WORLD-XY 平移冒充锥形指纹。因此当前
  `run_m12_event_trigger_insert_demo.m` 的 fixed-orientation 八点段不满足最终任务合同，
  在改写完成前不能被报告为 conical fingerprint。
- `mujoco_unscrewing_probe_new` 的活动采样/闭环路径固定 nominal tip，并执行：
  upright settle `0.80 s`；smoothstep 倾到 `15°` 用时 `10/15 s`；以
  `roll=15° sin(theta), pitch=15° cos(theta), theta:0→2pi` 扫描 `52/15 s`；
  用 `22/15 s` 回正；闭环另保持 upright `0.20 s`。原始序列最终周期重采样为 360 个
  phase bins，方向网络使用 `Fz / Tx / Ty / actual tip-z` 四通道。该 RPY 圆的真实轴倾角
  为约 `14.9569–15°`；若真机采用严格常倾角 SE(3) 圆锥，差异必须作为实现差异留痕。
- KST/Sunrise 源码复核确认 `KST(..., Tef_flange)` 会通过 `TFtrans` 把非单位变换注册为
  default TCP；后续 `getEEFPos`、`movePTPLineEEF` 和 Cartesian DirectServo 实际都作用于
  该 TCP。因此正确路径是先注册物理 peg tip，再保持 tip 的 WORLD XYZ 不变、改变连续
  ABC；Sunrise 负责法兰补偿。继续使用 `eye(4)` 时旋转中心是 media flange，不是 tip。
- 当前唯一已知的 42 mm 是蓝色 sensor-side 安装面到 tip 的 CAD 基准，不是完整
  media-flange-to-tip 变换。`configs/common_work_pose_latest.json` 中
  `T_flange_to_tip` 仍为空，白色转接件、Bota 本体和蓝色工具的完整装配栈尚未实测；
  仅 42 mm 在 15° 下就对应约 `10.9 mm` 横向杠杆运动，不能直接填入 KST 冒充完整 TCP。
- Bota 当前资格仅覆盖固定姿态 baseline-relative binary contact event。15° 旋转会改变
  工具自重在六轴上的投影；正式实现应使用同相位无接触基线或经实测质量/CoM得到的
  orientation-aware gravity compensation，并重新限定 phase-indexed hard/event gates。
- 本次仅完成 MuJoCo、KST MATLAB client 与 Sunrise Java server 的只读交叉审计并记录
  正确合同；没有连接 KST、没有发送机器人命令，也没有把缺失的 TCP 数值猜成真值。

## 2026-07-23 — 真实 15° 连续锥形主线、装配尺寸链更正与低带宽证据

- 活动入口
  `run_m12_event_trigger_insert_demo.m` 已由历史八点平移 proxy 改成注册 tip TCP 的
  Cartesian DirectServo 连续锥形：`0.8 s` 直立稳定、`10/15 s` 平滑倾至 `15°`、
  `52/15 s` 保持常倾角连续转满 `360°`、`22/15 s` 回正、`0.2 s` 直立核验。
  接触前先以相同姿态时间表做一次离面 unloaded cone；接触圈按实际姿态选择最近的
  unloaded 样本作相对 wrench 参考。该方法是姿态匹配的 commissioning 补偿，不冒充
  完整 mass/CoM/inertia 与 Sunrise LoadData 标定。
- DirectServo start 返回后立即安装独立、幂等 stop cleanup。正常结束和异常展开均调用
  `realTime_stopDirectServoCartesian()`；active/stopped marker 使重复 stop 可判定。
  锥形 CSV 在进入实时循环前只打开一次，循环内直接 `fprintf`，不再每 tick 反复
  `fopen/fclose`。日志 40 列同时记录 source stamp/age、参考类型/姿态误差、tick
  lateness、命令/实际 tip pose、raw wrench、WORLD* wrench 与相对范数。
- MATLAB 实时窗口限频约 `8 Hz`，复用控制循环已有 pose/wrench：显示实际/命令 tip
  轨迹、WORLD/TCP 三轴、peg 轴与 Bota 测量原点处 WORLD* 六维时序。力/力矩只旋转，
  未知 sensor-origin→tip moment arm 时不平移 torque；关闭窗口只停绘图。
- 尺寸语义按操作者更正彻底拆分。真实顺序固定为：
  `KUKA media flange → 白/银适配法兰 → Bota robot-side plane`
  `→ Bota tool-side/wrench plane → 水泥灰适配法兰`
  `→ 粉色 CAD 安装面 → 粉色法兰+peg tip`。
  Bota 官方两安装面间距为 `38.00 mm`；粉色交付 STEP 的安装面到 tip 为 `42.00 mm`
  且已包含粉色法兰。蓝色连接件是另一独立参考件，人工尺量 `10–11 mm`，完全不进入
  当前粉色 TCP 方程，也不再作为粉色 CAD 面的别名。
- 白/银适配法兰与水泥灰适配法兰的单件厚度仍未计量。当前 `92±4 mm` 只允许解释为
  一次端到端临时观测：
  `KUKA media flange→粉色 CAD 安装面约 50±4 mm`
  `+ 粉色 CAD 安装面→tip 42 mm`。
  该 50 mm 整段若后续确认并非量到灰色件/粉色件的真实接合面，就必须废止而不能拆项
  重构。水泥灰法兰底面到桌面约 `570 mm` 是当前拍照姿态的场景高度，不是灰色法兰
  厚度、不是 TCP 项，也不能与 38/42 mm 相加。
- 活动工具合同已删除错误的 `blue_sensor_plane_to_tip` 别名，新增上述 ordered chain、
  两个未决装配子变换和 570 mm 场景观测。Python loader 同步拒绝蓝/粉 datum 混用。
- 离线验收：MATLAB R2026a `checkcode=0`；锥形 trace header/formatter 均为 `40` 列；
  `real_tooling_v1.json` 通过 JSON 解析；Python compileall 通过；工具合同 unittest
  `2/2` 通过。活动主程序 SHA-256 为
  `0d4130c033265ca51f1e175726b08d95ad4aba4531ab5aa8c394db2d2bd05c10`，
  工具合同 SHA-256 为
  `565834dbd17bda4ed28799ae88284e0e9f880bf6ea45271dd3190885d8a57a86`。
- 本轮没有初始化 ROS/KST、没有连接 MatlabToolboxServer、没有读取实时 Bota 数据、
  没有发送任何机器人命令。上述 `92±4 mm` 仍是动作调试值，不是 pivot/touch-off
  计量结果。

## 2026-07-23 18:57 — 默认折叠位的水泥灰法兰桌面高度观察

- 操作者在“用户指定默认折叠实验初始位姿”下报告：水泥灰适配法兰盘底面到白色桌面
  的垂直距离约为 `59 cm`（约 `590 mm`）。
- 该数值只绑定当时的折叠姿态、安装堆栈与白色桌面，是人工近似场景观察；不作为
  水泥灰法兰厚度、KST TCP、flange-to-tip、WORLD Z 标定或运动目标。
- 此前约 `570 mm` 的观察属于另一拍照姿态，两者并存且不得互相覆盖。工具、传感器、
  适配法兰、机器人姿态、桌面或安装关系改变时，需要重新观察。
- 本次只更新语义合同和留痕，没有连接 ROS/KST 或发送机器人命令。

## 2026-07-23 19:21 — M12 主程序基线接收节奏误判修复

- 操作者从人工放置的近孔位运行 `run_m12_event_trigger_insert_demo.m`。程序读取到稳定
  live start：tip TCP `xyz=[472.656, 1.695, 316.770] mm`，但在第一条运动命令之前
  因 MATLAB 基线消费率门限退出。因此本轮 `motion_commanded=false`，随后通过活动
  KST 对象写入 `end`；SmartPAD Selected 仍由操作者观察。
- Bota 驱动诊断实际为 `199.956 Hz`、`rejected_frames=0`，身份、配置哈希、消息新鲜度
  均通过。原门限以 150 个 source stamp 的首尾平均值计算出 `30.7 Hz`。离线复核同批
  数据得到间隔中位数约 `19.94 ms`（持续消费节奏约 `50.15 Hz`）；少数图窗首次渲染及
  逐样本反复 `fopen/fclose` 停顿把首尾平均值拉低，最大一次约 `0.705 s`。这不是
  Bota 200 Hz 驱动链路降速。
- 主程序现将 raw/derived/WORLD* 三份高频 wrench CSV 每次运行各打开一次，由
  `onCleanup` 在正常、异常或 Ctrl-C 路径统一关闭；逐样本原始证据没有删减。三个日志
  共用每个样本的一份 wall/elapsed 时间。
- 图窗在基线测量前强制完成第一次 `drawnow`。40 Hz 门限保留，但改为检查正向
  source-stamp 间隔中位数所代表的持续 MATLAB 消费节奏；首尾平均率、P95 间隔和最大
  间隔继续写入 summary/event，不能被隐藏。
- 驱动约 200 Hz 诊断、消息新鲜度、单调时间戳、硬力/力矩门限、DirectServo 超时和
  异常不盲返均未降低。证据目录为
  `data/M12_event_trigger_insert_live_20260723_192101_880`。
- 同时修正新 run CSV header 的换行写法，避免表头末尾保留字面量 `\n`；历史原始
  run 不改写。
- MATLAB R2026a `checkcode -id` 返回 0 个问题。本次修复后只做静态/离线复核，没有
  再次连接或运动机器人；活动源码 SHA-256 为
  `a0448499923bd73ac9d37ee20cfdcef75b9710177b4cbfd445527c6644282ab9`。
  下一次真实 F5 run 才能验证优化后的在线持续节奏。

## 2026-07-23 19:35 — 图窗强制首帧阻塞修复

- 第二次 F5 run 捕获到同一稳定近孔起点
  `tip xyz=[472.656, 1.695, 316.770] mm` 后没有运动。只读检查显示
  `raw_wrench.csv` 仍只有表头，说明程序尚未进入基线循环。
- 上一版为隔离首次渲染抖动而在 `initializeDashboard` 末尾加入的无条件
  `drawnow` 阻塞了 MATLAB Figures 前端。19:36:00 cleanup 通过活动 KST 对象写入
  `end`；本轮没有任何 command 记录，也没有生成任务 summary。
- 修复后，`makeContext` 不再创建图窗：先以持久打开的 CSV 完成纯 Bota 基线并通过
  节奏/噪声检查，再在第一条机器人命令之前创建 dashboard。初始化路径不再调用完整
  阻塞式 `drawnow`；任务中的限频更新仍使用 `drawnow limitrate nocallbacks`。
- 新事件 `DASHBOARD_INITIALIZING_NO_MOTION` 与 `DASHBOARD_READY_NO_MOTION` 明确区分
  可视化准备和首条运动命令。MATLAB R2026a `checkcode -id` 返回 0 个问题；活动源码
  SHA-256 为
  `fc3931782705fdab574b346892cc4d0867d9ecb44295714acecab70bac972e46`。
- 本轮诊断和修复没有发送机器人命令。证据目录：
  `data/M12_event_trigger_insert_live_20260723_193455_161`。

## 2026-07-23 19:41 — 首个 touchdown 微段后的 MATLAB 绘图阻塞

- 第三次 F5 run 已证明前两项修复有效：Bota 基线持续/平均消费率为
  `66.7/66.4 Hz`，P95/max source-stamp gap 为 `0.0250/0.0301 s`，力/力矩噪声
  `0.0242 N / 0.00057 Nm`，dashboard 对象在约 `0.22 s` 内创建完成。
- 首个 `TOUCHDOWN_001` 命令为 WORLD `-Z 0.10 mm @ 1 mm/s`；实测位移
  `[+0.0415, -0.0011, -0.0633] mm`，端点误差 `0.0555 mm`，在 `0.15 mm`
  tracking tolerance 内。随后一帧端点相对力仅 `0.0337 N`、相对力矩
  `0.00056 Nm`，没有触发接触或 hard guard。
- 该帧已经写入 raw/derived/WORLD* CSV，但下一帧没有到达日志。执行路径唯一剩余位置
  是 `updateDashboard` 的首次 `drawnow limitrate nocallbacks`；Ubuntu MATLAB Figures
  前端仍可在这个“限频”调用中无限阻塞。因此机器人看起来只动一下。
- 操作者 Ctrl-C 后，19:43:49 cleanup 通过活动 KST 对象写入 `end`；没有盲返。证据
  目录：`data/M12_event_trigger_insert_live_20260723_194102_328`。
- 主程序现明确采用
  `DEFERRED_NO_DRAWNOW_IN_ROBOT_COMMAND_THREAD`：继续以最高 8 Hz 更新内存中的
  trajectory/wrench graphics objects，但控制线程绝不调用 `drawnow`。MATLAB 可自行
  opportunistic repaint，完整图在函数回到 prompt 后仍可呈现；动作和原始日志不再等待
  Figures 前端。manifest 同步记录该 render policy。
- MATLAB R2026a `checkcode -id` 返回 0 个问题；活动源码 SHA-256 为
  `661a0afb355e41cf31ff1ad7f7533b522d2384547098f0d83714de41fbfd39c9`。
  本次代码修复没有额外发送机器人命令。

## 2026-07-23 19:45 — F5 进入后卡在无日志预检区

- F5 实际已创建
  `data/M12_event_trigger_insert_live_20260723_194540_237`，但事件只到
  `CONICAL_QUALIFICATION_BOUNDARY`；没有 KST connection、ROS global node、Bota
  subscriber 或运动命令。`kst_end_state.json` 保持 `NOT_OPENED`。
- 终端独立运行同一个 `prepare_matlab_kst_bota.sh status` 用时约 `10.96 s` 并全部
  PASS，Bota 约 `200.0 Hz`，因此不是 PC/ROS/Bota 环境失败。
- 该无日志区此前包含无条件 `addpath(KST Matlab_client)`、根 status、命令所有者检查
  和 MATLAB ROS node restart。VS Code MATLAB language server 正在索引工作区时，重复
  `addpath` 可能触发耗时重索引，且原程序无法显示具体停点。
- 主程序现在仅在 `KST` class 尚不可用时执行一次 `addpath(...,'-end')`；已加载的会
  明确记录并跳过。根 status 增加 `/usr/bin/timeout 30`，并为 KST path、root status、
  command-owner check、MATLAB ROS restart 的 begin/pass 分别写即时事件。
- MATLAB R2026a `checkcode -id` 返回 0 个问题；活动源码 SHA-256 为
  `c3f0a16ffd3394c605e059e787d223741cc40ab60e8e520d1a21868094453e5e`。
  本轮诊断与代码修改没有连接或运动机器人。

## 2026-07-23 20:13 — MATLAB shell 缺少 ROS_DISTRO 导致 root status 早退

- 新 run 已依次通过 KST client path 加载并进入 `ROOT_STATUS_BEGIN_NO_MOTION`，随后
  `prepare_matlab_kst_bota.sh status` 在 source ROS Noetic 时失败：
  `/opt/ros/noetic/etc/catkin/profile.d/1.ros_distro.sh` 在调用方 `set -u` 条件下读取了
  尚不存在的 `ROS_DISTRO`。该异常发生在 ROS global node、KST connection 和任何运动
  之前。
- 终端此前能通过而 MATLAB 失败，是因为终端继承了已经设置的 ROS 环境，MATLAB
  `system()` 启动的 shell 没有 `ROS_DISTRO`。这不是 Bota、KUKA 网络或
  MatlabToolboxServer 故障。
- 共享入口 `src/iiwa_lab_bringup/scripts/setup_env.sh` 现在只在变量完全不存在时先
  `export ROS_DISTRO=""`，随后仍 source 官方 Noetic setup。若调用方已经设置其他发行版
  值则原样保留，ROS 官方脚本仍会报告发行版混用，未绕过该检查。
- 两项复核通过：
  1. `env -u ROS_DISTRO` 且 `set -u` 的 shell source 入口后得到
     `ROS_DISTRO=noetic` 与正确 `ROS_MASTER_URI`；
  2. MATLAB R2026a 内 `setenv('ROS_DISTRO','')` 后通过 `system()` 执行同一 bounded
     status，返回码 `0`，Bota 实测 `200.0 Hz` 且全部 PASS。
- 主程序 MATLAB `checkcode -id` 为 0。验证只运行命令自由的 status，没有探测 KST
  30001、没有启动机器人控制器、没有发送运动命令。

## 2026-07-23 20:21 — F5 已执行；ROS nounset 链与遗留 KST socket 修复

- 操作者看到 VS Code 下方面板空白，但面板当时选中的是 `OUTPUT`，并非
  `TERMINAL` 中的 MATLAB 会话。F5 实际已经执行并生成
  `data/M12_event_trigger_insert_live_20260723_202133_776`。
- 该 run 在 `ROOT_STATUS_BEGIN_NO_MOTION` 后约 `0.13 s` 结束：
  `/opt/ros/noetic/etc/catkin/profile.d/10.roslaunch.sh` 在调用方 `set -u`
  条件下读取尚不存在的 `ROS_MASTER_URI`。`motion_commanded=false`、
  `kst_opened=false`，没有连接或移动机器人。
- 20:13 的单变量 `ROS_DISTRO` 补丁只修复了生成脚本链中的第一个未定义变量，现已
  被通用修复取代：`setup_env.sh` 仅在 source ROS Noetic 与工作空间生成的 setup
  文件期间暂时关闭 nounset，随后恢复调用方原本的 nounset 状态。空环境同时移除
  `ROS_DISTRO/ROS_MASTER_URI/ROS_IP/ROS_HOSTNAME/ROS_VERSION/ROS_PYTHON_VERSION`
  后复测，source 成功得到 `ROS_DISTRO=noetic` 与
  `ROS_MASTER_URI=http://192.168.1.41:11311`，且 nounset 恢复为开启。
- 只读 status 随后正确前进到独立的 command-owner 检查，并发现当前 MATLAB
  PID 2274058 仍持有 `192.168.1.147:30001`。证据追溯到未完成的
  `artifacts/kst_recovery/return_then_end_20260723_202012_710`：它只记录到
  `CONNECTING`，没有 summary；之后两次 recovery 均因 tool setup 未获确认而只写
  `end`。
- 根因是 KST 的现代 `tcpclient` 路径原先只写 `end\n`，没有删除 transport；KST
  handle 也继续保存该对象。因此 SmartPAD 生命周期请求与 PC socket 释放被错误地
  混为一件事，后续 F5 会被活动连接门拒绝。
- 已修复 `Matlab_client/net_turnOffServer.m`：现代 `tcpclient` 写入 `end` 后立即
  `delete`；并修复 `KST.net_turnOffServer()` 在成功后清空其 `t_Kuka` 属性。当前
  MATLAB 已加载的是旧 class definition，必须重启该 MATLAB 会话一次以释放这条
  历史 socket；之后的新 run 会采用修复后的关闭语义。
- 主任务文件 `checkcode -id=0`；KST helper `checkcode -id=0`。KST class 的 12 条
  analyzer 信息均为原上游未使用变量/不可达代码/矩阵求逆建议，不是本次解析错误。
  两个用户附件中数百条 severity 4 信息来自第三方 `Matlab Comment Checker`，属于
  缺少注释提示，不会阻止 F5。
- 本次只读检查和源码修复没有发送机器人命令。修复后 SHA-256：
  `setup_env.sh=86fa4ebea3638b64b64743b61fc1ceed48c5d256a3fc76c7114b3f931b53ed6d`，
  `KST.m=2345cd1225847c257e2fbb45198bfeb0ede4707b267cfb0998e9aa3a22186c88`，
  `net_turnOffServer.m=3d5abbf2683a391bcd43053a061d6ed1a063ea80c321c8c74f04a2180f10291d`。

## 2026-07-23 20:42 — 首次完整 touchdown 成功；接触释放误用孔内跟踪门

- run `data/M12_event_trigger_insert_live_20260723_204145_676` 已完整通过 root
  status、单一命令所有者、MATLAB ROS node、Bota 诊断/新鲜度、KST tool-frame
  连接和稳定起点检查。Bota 驱动约 `198.4 Hz`，MATLAB 基线持续/平均消费率
  `66.5/54.6 Hz`，基线最大力/力矩标准差 `0.0234 N / 0.00056 Nm`。
- 机器人从本轮 task-local start
  `xyz=[472.672, 1.695, 316.653] mm` 以 WORLD `-Z 0.10 mm @ 1 mm/s`
  完成 23 个 blocking touchdown 微段；累计实测路径约 `2.022 mm`。在
  `Z=314.480 mm` 得到连续接触事件，峰值相对力 `2.777 N`、相对力矩
  `0.00373 Nm`，远低于 `20 N / 0.75 Nm` hard guard。下压与 Bota event-trigger
  本身已验证工作。
- 随后的第一条接触释放命令为 WORLD `+Z 0.10 mm @ 0.6 mm/s`。其 full-vector
  endpoint error 为 `0.0854 mm`；方向没有颠倒，也不是 KST `nak`，但
  `contactAwareRetractToZ` 错误复用了孔内插入专用的 `0.0700 mm` 门，因此在向上
  卸载阶段过早终止。cleanup 已写入 KST `end`，未盲返，SmartPAD Selected 仍由
  操作者确认。
- 修复只把 `contactAwareRetractToZ` 的接触释放微段切换到已经用于 touchdown 的
  `touch_tracking_tolerance_mm=0.15`。实际 M12 孔内插入仍保持
  `insertion_tracking_tolerance_mm=0.07`；WORLD +Z 方向断言、0.10 mm 最大未观测
  微段、0.20 mm cross-track、工作空间/关节裕度、Bota freshness 以及
  `20 N / 0.75 Nm` hard guard 均未改变。
- MATLAB R2026a `checkcode -id` 为 0；没有额外发送机器人命令。修复后主文件
  SHA-256 为
  `9773a8bd62d93e9542251e1580935bbeb5968c89d4c63ddb6e4ea0a8f6c99ddd`。
- 本轮异常结束时工具仍处于表面受力位置。下一次 F5 前必须由操作者在 Impedance
  中手动抬到无载、近孔、至多 20 mm 空隙的 task-local start；不能直接把当前受力
  状态当成新基线，否则会把接触载荷吸收到 baseline。

## 2026-07-23 20:51 — touchdown 与 2 mm 空中参考完成；旧部署不支持 DcSeCarEEfP_ 反馈

- run `data/M12_event_trigger_insert_live_20260723_205141_150` 再次通过全部
  PC/ROS/Bota/KST 起动门。task-local start 为
  `xyz=[472.731, 0.912, 315.888] mm`；Bota 基线持续/平均消费率
  `66.7/70.1 Hz`，最大力/力矩标准差 `0.0355 N / 0.00068 Nm`。
- WORLD `-Z` touchdown 在第 21 个 0.10 mm 微段确认，接触面候选
  `Z=314.458 mm`，峰值相对力 `4.165 N`，远低于
  `20 N / 0.75 Nm` hard guard。接触释放先执行 3 个 0.10 mm 慢速卸载微段，
  随后执行 4 个约 0.481 mm 自由空间段，最终实测
  `Z=316.684 mm`。因此照片中的上抬量相对接触面是 `2.226 mm`；目标为
  2.0 mm，阻塞相对运动的累计端点偏差为约 `+0.226 mm`，不是厘米级异常抬升。
- `UNLOADED_CONE` 的 Cartesian DirectServo 已被 Sunrise 正常确认启动，但第一条
  `sendEEfPositionGetActualEEFpos` 调用返回的内容不能解析为六维位姿；
  `conical_directservo_trace.csv` 只有表头，证明程序在第一条锥形样本之前终止。
  独立 DirectServo cleanup 随即得到 stop acknowledgement，之后 KST `end` 成功；
  没有盲返。
- 本地 KST 1.7 源码标明 `DcSeCarEEfP_` 是 2019-05-12 新增的可选
  “命令+实际笛卡尔位姿”协议；当前控制柜部署的 MatlabToolboxServer 很可能是更早
  的实现。KST 官方教程、`junyuan/Control_Main_2MPIC.m`、`5.KMPC`、
  `6.GTpHRI` 和 `7.DMPC` 均使用原始 `DcSeCar_` /
  `sendEEfPositionf` 流式通道。
- 主程序现改为每拍通过 `sendEEfPositionf` 发锥形目标；每 5 拍使用旧部署一直支持
  的 `getJointsPositions` 读取真实关节，并用 KST iiwa14 + 92 mm TCP 正运动学重建
  实际 tip TCP。其余拍只传播上一实测 tracking offset，并在
  `conical_directservo_trace.csv` 的 `feedback_observed` 字段明确标为非实测；
  tip/orientation tracking 与关节裕度断言只在实测拍执行，Bota freshness 和
  `20 N / 0.75 Nm` hard guard 仍每拍执行。
- 用本次起始关节离线复算，正运动学 TCP 与控制器记录的平移误差仅
  `0.000032361 mm`，姿态旋转误差约 `1.69e-7 rad`，验证了模型、iiwa14 变体和
  92 mm TCP 链的一致性。MATLAB R2026a `checkcode -id` 为 0；源码中已不存在
  `sendEEfPositionGetActualEEFpos` 调用。
- 修复后主文件 SHA-256：
  `1a832747e71c5616371f19c13a0792e2894a439d02b89e6e49114e80e3272e28`。
  本次分析、离线正运动学和源码修改没有连接或运动机器人。

## 2026-07-23 22:39 — 移除正常路径上抬；接触锥形、下压力捷径与熵网格贯通

- run `data/M12_event_trigger_insert_live_20260723_210914_098` 再次验证了短程
  touchdown：第 12 个 WORLD `-Z 0.10 mm` 微段在
  `tip Z=314.482 mm` 得到连续接触事件，峰值相对力 `2.581 N`。此前两次有效
  touchdown 分别约为 `2.777 N` 与 `4.165 N`，而三次起始基线最大力轴标准差约为
  `0.0226..0.0355 N`。因此本轮继续使用 `2.0 N × 3` 个新鲜样本作为接触事件，
  并使用约 `3.0 N` 作为接触面运输目标；这些是数据导出的实验门限，不是认证安全值。
- 21:09 run 的实际终止原因不是接触阈值。旧 pipeline 先从接触面上抬，再进入
  `UNLOADED_CONE`；DirectServo 内一次同步关节查询耗时约 `0.18 s`，使下一拍 lateness
  达 `0.1797 s` 并触发 `ConicalLoopOverrun`。DirectServo cleanup 和 KST `end`
  均已执行，没有盲返。
- 正常 pipeline 已改为：
  `touchdown -> 原接触位连续 15° tip-centered 360° cone -> 约 3 N 下压力捷径
  -> 同一下压力的 task-aware entropy grid -> 倒角向下进展事件
  -> 冻结 XY、只用 -Z 的二次确认 -> 0.10 mm 分段插入 -> 深度带+持续触底事件
  -> KST end`。源码中已不存在 `AIR_REFERENCE_RETRACT`、`UNLOADED_CONE`、
  `RETOUCH` 或 `contactAwareRetractToZ` 正常路径。
- 锥形流使用 `40 Hz` 的 `DcSeCar_ / sendEEfPositionf`。每个 command tick 仍读取一帧
  新鲜 Bota wrench，但同步 `getEEFPos/getJointsPos` 只在 DirectServo start 前和 stop
  后执行；流内轨迹明确标为 command-propagated，不能被误称为实测 tracking。
  接触锥形用同一 run 的直立原始力基线和当前姿态分别旋转到 WORLD 后相减，以抵消刚体
  重力力矢量；缺少完整 tool CoM/inertia，因此力矩只作 guard，不参与捷径方向。
- 捷径与熵网格共用 endpoint-observed 接触外环：XY 每段至多 `0.15 mm`，Z 每次至多
  `0.05 mm`，目标法向力 `3.0 +/- 0.4 N`。捷径末端若没有捕获事件，会立即作为真实
  negative observation 更新 posterior 并把局部原点标为已测，避免熵网格重复第一点。
  后续 action utility 为“信息增益 + 线段捕获概率奖励 - 行程代价”，未捕获会沿实际
  计划线段更新 posterior。
- 倒角捕获不由模型标签或低总力单独宣布。候选必须同时满足：相对接触面累计向下
  `>=0.25 mm`、最近三步仍向下 `>=0.04 mm`，且在 `8 N / 0.25 Nm` soft envelope
  内；随后固定 XY，禁止任何 `+Z` 命令，并要求再获得 `>=0.10 mm` 实测下沉且法向力
  `<=5 N`。捕获位置记录改为实测 pose 相对本轮起点的 XY，而不是计划端点。
- 捷径方向的符号也经冻结仿真数据复核。只读重放
  `mujoco_unscrewing_probe_new/outputs/current/datasets/cross_size_s4_adaptation/m12/trials/*.npz`
  中 120 个有效 M12 conical trial，按当前 force/downward-command cue 启发式计算：
  原 `+axis` 对真中心方向的中位误差 `114.69°`、`<90°` 比例 `16.7%`；改用
  `-axis` 后为 `65.31° / 83.3%`。在 supported-search 子集上分别为
  `139.17° / 0%` 与 `40.83° / 100%`。因此当前中心运输采用 `-TCP-axis` 映射，
  但明确记录为 M12 commissioning heuristic，不冒充已训练 posterior 或普适物理定律。
- 结束语义已拆开：触底后只有 KST end 写入成功才记录
  `...KST_END_SENT_PENDING_SMARTPAD_SELECTED_OBSERVATION`；写入失败不再误报 sent。
  SmartPAD `Selected` 仍需操作者目视确认，所以 `success=false` 保持证据边界，并另有
  `task_completed`、`completion_status` 和
  `human_selected_confirmation_pending` 表达真实状态。
- 可视化 peg 轴改为已注册 TCP 的 `+Z`；在当前直立姿态它对应 WORLD `-Z`，与
  touchdown/insertion 方向一致。轨迹图把 DirectServo 流内轨迹标为
  `observed / command-propagated`，不再统一写成 actual。
- `group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md` 已同步：
  旧八点/抬空/retouch 页面明确标为 superseded；活动流程改为原接触位 15° cone、
  约 3 N 捷径、同力熵网格、no-lift capture confirmation、插入触底和 KST end。
- MATLAB R2026a 对最终主文件执行 `checkcode -id` 返回 `0` 个问题。最终活动源码
  SHA-256 为
  `c78819706a441bb8dea8416e0dc4dc087244859623b00dee7afb76bb26ade095`。
  本轮只做日志分析、冻结仿真离线重放、源码修改与静态检查；没有连接或运动机器人。

## 2026-07-23 22:54 — 接触事件有效；DirectServo 被 MATLAB 图形写入拖慢后误判超时

- run `data/M12_event_trigger_insert_live_20260723_225440_600` 的 touchdown
  在第 19 个 `0.10 mm` WORLD `-Z` 微段确认：接触候选
  `tip Z=315.406 mm`，峰值相对力 `2.903 N`，起始基线最大力轴标准差
  `0.0301 N`。`2.0 N × 3 fresh samples` 接触门已经触发，且接触位约
  `3.0 N` 的 WORLD* 法向反馈稳定；本次停止不是阈值过高或 event trigger
  未配置。
- MatlabToolboxServer 已确认 `CONTACT_CONE` Cartesian DirectServo start。
  `conical_directservo_trace.csv` 记录的前三拍 lateness 为
  `0.00017 / 0.00321 / 0.11648 s`，随后达到 `0.2049 s` 并触发
  `M12Demo:ConicalLoopOverrun`。独立 DirectServo stop cleanup 与 KST end
  均成功，所以 SmartPAD 回到 Selected。
- 根因是 DirectServo command thread 虽未调用 `drawnow`，却仍通过
  `receiveWrench -> updateDashboard -> addpoints/set/title` 在控制拍内修改
  MATLAB Figures。现场图形前端一次写入可阻塞约 `0.1..0.2 s`；旧绝对时钟又把
  这一笔延迟重复带到后续拍，造成同一抖动被连续判罚。该问题属于控制线程调度，
  不是接触力学事件。
- 活动主程序现在仅在 DirectServo 锥形流内部暂停全部图形对象写入；每拍 Bota
  freshness、WORLD* wrench guard 与 `conical_directservo_trace.csv` 仍保留，
  流结束后 dashboard 在捷径/熵网格/插入阶段恢复更新。
- 锥形时钟改成“几何样本一个不跳、孤立大抖动只计一次并平移剩余时基”：
  lateness `>0.12 s` 触发一次 schedule rebase；单次 `>0.50 s` 或超过
  8 个真正连续 late tick 才终止。这样不会把同一笔 GUI/ROS 延迟重复判罚，
  又保留了持续卡顿时的 DirectServo stop。
- touchdown 后额外接触 settle 从 `0.80 s` 缩为 `0.20 s`；KST start 本身仍有
  约 `0.5 s` settle，随后直接执行真实 `15°` tip-centered continuous cone，
  不增加上抬、空中参考或 retouch。接触阈值 `2.0 N × 3`、接触运输目标
  `3.0 N`、`20 N / 0.75 Nm` hard guard 均未放宽。
- MATLAB R2026a `checkcode -id` 返回 `0` 个问题。修复后的活动源码 SHA-256：
  `39103fe3c58059da6dad3681532fdc5bec59fe5ee9e0b41882befe30bf20bb0c`。
  本轮只读取既有真机日志、修改源码并静态检查；
  没有连接或运动机器人。

## 2026-07-23 23:13 — 接触事件正常；15° RAMP 因几何压入达到 20.669 N

- run `data/M12_event_trigger_insert_live_20260723_231311_441` 再次完整通过
  PC/ROS/Bota/KST 起动门。task-local start 为
  `xyz=[474.658, 0.901, 316.009] mm`；WORLD `-Z` touchdown 在第 10 个
  `0.10 mm` 微段确认，接触面候选 `Z=315.411 mm`、峰值相对力
  `2.818 N`。因此 touchdown 的 `2.0 N × 3 fresh samples` event trigger
  已正常工作，不是本轮停止原因。
- `CONTACT_CONE` DirectServo 已在 `40 Hz` 正常启动。已落盘的 31 个锥形样本
  仅包含 `SETTLE` 和 `RAMP`，最大 tick lateness 只有 `0.007659 s`，证明上一轮
  GUI/时基问题已经消失；程序尚未进入 `SCAN`，所以没有机会旋转一整圈。
- 真实停止原因是 RAMP 时保持 TCP 中心 Z 不变，8 mm across-flats 六角端面随
  倾角把边缘压入平面。已落盘的最后一帧在命令倾角 `13.647310°` 已达到
  `18.616135 N`，WORLD* `Fz=16.437675 N`；下一新鲜 Bota 帧达到
  `20.669 N`，独立 `20 N` hard guard 按设计停止 DirectServo 并发送 KST end。
  本次不是“阈值没触发”，而是旧残差环的 `0.30 mm/s、±0.25 mm` 卸载能力远小于
  倾斜平端面所需的几何抬升。

## 2026-07-24 00:37 — 几何前馈、冻结相位卸载与完整证据链

- 15° cone 的动作顺序保持完整：
  `0.40 s SETTLE -> 3.00 s RAMP -> 6.00 s / 360° SCAN
  -> 3.00 s RETURN -> 1.00 s UPRIGHT_VERIFY`。没有改成八点近似，也没有在
  touchdown 后抬空重来。
- 根据活动工具合同中的 M12 六角端面 `AF=8.0 mm` 和
  `hex_flat_normal=+X`，每个 DirectServo 样本现在计算六个真实顶点的支撑函数。
  15° 全倾角的一圈所需 WORLD `+Z` 几何抬升随相位为
  `1.035276..1.195434 mm`。该透明 feedforward 先消除已知平端面压入，
  再由残差力环修正打印、TCP 和柔顺误差。
- 残差力环仍以约 `3 N` 下压力为目标，但改为非对称：
  WORLD `+Z` 卸载最多 `1.25 mm/s、+0.75 mm`，重新下压最多
  `0.40 mm/s、-0.40 mm`。当总相对力进入 `5.5 N` 以上时，控制器使用
  `max(法向力, 六维中的总力范数)` 预卸载，避免“横向边缘力很高而法向力较低”
  时仍继续下压。
- 总力达到 `8.0 N` 时，不提高 hard guard，而是冻结当前倾角和锥形相位；
  只允许 WORLD `+Z` 或零位移卸载。总力连续 3 个新鲜样本
  `<=5.5 N` 后才从同一几何样本继续，因此不会跳过 360° 覆盖。单次恢复按真实
  wall time 限为 `3.0 s`，所有恢复累计 wall time 限为 `20.0 s`；
  `20 N / 0.75 Nm`、Bota freshness、任务局部工作空间、关节裕度和
  DirectServo `onCleanup` 都未放宽。
- 修复了两个后续鲁棒性/证据问题：
  1. 锥形结束若仍有小 Z 偏置，表面闭环只按原有 `0.05 mm` 微步重新贴合，
     不会由绝对高度夹紧生成一次较大的 Z 命令；
  2. 每个锥形 wrench 样本先写入 `conical_directservo_trace.csv`，再检查 hard
     guard，所以真正触发停止的 force/torque 双通道峰值也会留痕。异常 summary
     还会从 trace 和异常文本回填峰值，不再错误沿用 touchdown 峰值。
- 启动期 contract 现在还会验证“最大六角几何抬升 + 最大残差卸载”能够装入
  task-local WORLD `+Z` 包络；恢复 episode/tick/真实 wall time 都会写入顶层
  `summary.json`。DirectServo start 调用若在“控制器可能已接受、MATLAB 尚未返回”
  的极小不确定窗口抛异常，也会先做一次幂等 stop 尝试再保留原始异常。
- MATLAB R2026a 对活动主文件执行 `checkcode -id` 与 `mtree` 解析均通过；
  六角支撑函数离线检查得到 `1.035276..1.195434 mm`。本轮只分析已有日志、
  修改源码并做离线/静态验证，没有连接或运动机器人。活动主文件 SHA-256：
  `ebca193d7e47c709a63c7dc11f61aea8fa4c35dedb07a9f46f7278171638df33`。

## 2026-07-24 00:45 — 完整锥形已通过；捷径末端分辨率死循环已解除

- run `data/M12_event_trigger_insert_live_20260724_004512_158` 已完整执行真实
  `15°` tip-centered 连续锥形并进入 `SHORTCUT_CONTACT_TRANSPORT`。因此这次
  “锥形后一直 Running”不是锥形、接触 event trigger 或 hard guard 失败，而是
  后续捷径闭环的终点判据与真机可测分辨率不匹配。
- 本次 touchdown 在第 15 个微段确认，surface `Z=296.063 mm`、峰值相对力
  `3.273 N`。连续锥形留下 537 个样本
  （SETTLE 17 / RAMP 120 / SCAN 240 / RETURN 120 / UPRIGHT_VERIFY 40），
  最大 tick lateness `0.006288 s`，峰值相对力/力矩
  `6.756 N / 0.12797 Nm`；停止 DirectServo 后 tip/orientation 误差
  `0.039 mm / 0.010°`。这组证据明确支持“锥形整圈已完成”。
- 旧捷径要求 XY 残差 `<=0.020 mm`。本次在末段约 `0.0236 mm` 残差附近，
  机器人/KST 位姿反馈已基本不再产生可测进展：step 36..132 的平均命令约
  `0.023675 mm`，平均沿命令方向实测进展仅约 `0.0000055 mm`
  （约命令的 `0.023%`），其中 45 次甚至为负进展。程序因此共发出 133 个捷径
  命令、完成 132 个，却无法满足 `0.020 mm` 的旧交接门；最终由人工 Ctrl-C
  结束，cleanup 正常发送 KST end。
- 该停滞不是恒力闭环不稳定。旧 run 在捷径 step 24..132 的 327 个 endpoint
  dwell 样本全部位于旧 `2.25..3.75 N` 目标带内；WORLD* Fz 平均约
  `3.283 N`、标准差约 `0.107 N`，总力范数仅 `2.734..3.715 N`，峰值力矩
  `0.02358 Nm`。真正应修的是分辨率、有限步数和无进展语义，而不是继续抬高接触力。
- 活动程序将捷径/表面输送交接改为与已观察真机分辨率相容的
  `0.10 mm` XY endpoint tolerance；单次 XY 仍不超过 `0.15 mm`，Z 力调节量化为
  `0.06..0.08 mm`。连续 3 次低于 `0.010 mm` 的可测 XY/Z 进展会明确转入
  “已在交接容差内且接触持续”判定，或以可诊断异常退出，不再无限重复同一微命令。
  同时增加 `45 s / 80 actions / 8 force-settle actions` 的有限预算。
- 恒力输送的继续条件现在必须同时满足：接触事件仍持续、法向相对力位于
  `2..6 N`、总相对力 `<8 N`、总相对力矩 `<0.20 Nm`。软卸载采用
  `8 N / 0.20 Nm` 触发、`6 N` 恢复和有限恢复次数；捕获候选与确认必须位于同一
  软包络内，不能先于软卸载被误判。`20 N / 0.75 Nm` hard guard、Bota freshness、
  关节裕度、task-local workspace 与独立 KST end cleanup 均未放宽。
- 捷径到熵网格的交接、倒角捕获确认和插入阶段都补上了“命令已发但实测不动”
  的独立进展门。插入仍用 `0.10 mm` 微段；连续 3 次进展低于 `0.010 mm` 会终止
  并留下明确原因，成功必须同时满足深度带、持续触底事件和插入软力包络。修复了
  捕获确认后继续使用旧 pose，以及最后一次确认动作未重新评估的两个状态衔接问题。
- touchdown 法向符号不再取最后一帧，而是取连续接触确认窗口的 WORLD* Fz
  中值，避免单帧侧向冲击把后续锥形和表面恒力方向判反；完整窗口及最终符号会写入
  `events.csv`。
- 用本次旧日志做固定观测反事实回放时，新逻辑在第 6 个循环已进入
  `0.10 mm` endpoint settle，并会在约第 14 个循环、13 个 action 后满足有界接触
  交接；这只能证明旧死循环已被消除，不保证新真机 run 的力学轨迹完全相同。
- 当前活动 MATLAB 真机闭环**没有调用**仿真训练的神经网络。仓库中虽有 M12
  PyTorch `circular_cnn_best.pt` checkpoint，但尚无冻结 ONNX、完整
  preprocess/postprocess 合同或 MATLAB/Python golden-vector parity 证据。
  因此当前方向判定仍是透明的锥形相位/力矩/下降进展启发式，之后进入
  sensor-updated finite entropy grid。程序与 summary 明确记录
  `trained_model_inference_used=false`，不把启发式冒充 sim-to-real 模型。
  正确的后续接入顺序是：冻结 ONNX 与数据合同，先 shadow inference 留痕，
  通过 parity 后才允许模型建议不超过既有 `1 mm` 的捷径；模型缺失或低置信时仍
  回退到透明规则和熵网格。
- MATLAB R2026a 对最终主文件执行 `checkcode -id` 返回 `0` 个问题，`mtree`
  解析成功。最终活动源码 SHA-256：
  `ac2651f3213a50ceff02d0c87903174337596d4741ce7c28218e560c6283f6a4`。
  本轮只读取既有真机日志、修改源码、做静态检查与离线回放；没有连接或运动机器人。

## 2026-07-24 05:05 — 最新三 seed 模型接入真机 shadow；恢复仿真后验链而非只搬方向箭头

- 当前冻结模型选为
  `hardware_easy_v2_multisize_robust_direction_v4`，模型合同 SHA-256 为
  `0a73a9067b0ec60a26e0f76f6cbda6eba6727c8d510353bb0982c9658d484674`。
  它使用三组 seed 的 circular CNN，将一圈锥形指纹重采样为
  `4 x 360 [Fz, Tx, Ty, tip_z]`，同时输出三类接触状态 posterior、48 类圆周
  方向 posterior、APS prediction set、圆周集中度和 seed disagreement。部署时逐个
  核对 checkpoint、normalization、metrics、adaptation report 与 interface contract
  哈希；训练机留下的 `/home/njy/...` 绝对路径只作为旧 provenance，运行时绑定到
  当前 workspace 的冻结模型根目录。
- 新增 `real_device/model_runtime/infer_conical_direction.py` 和
  `deployment_manifest.json`。运行器内嵌与冻结 checkpoint 精确对应的
  `4->24->48->64` circular Conv1d 架构，不导入 MuJoCo、sklearn、joblib 或训练脚本；
  隔离环境固定为 CPU `torch==2.4.1+cpu / numpy==1.24.4`。它不打开 ROS、Bota、
  KST 或机器人 socket，只接受一个已经完成的锥形指纹，写一个 JSON 后验。
- MATLAB 仍是唯一真机命令所有者。完整 15°/360° cone 结束后，主程序只做一次
  `/usr/bin/timeout 20` 的 phase-boundary 推理，不在 DirectServo tick 内跨进程
  RPC。超时、缺文件、哈希不符、非有限输出或模型异常只写
  `LATEST_MODEL_SHADOW_INFERENCE_FALLBACK` 并回退透明 classical policy，不会让一个
  物理状态正常的 trial 仅因学习模块不可用而终止。
- 当前部署 authority 固定为 `SHADOW_ONLY`。仿真 gate 通过也不等于真机 promotion；
  `hardware_control_allowed=false`。模型在下一次真机 run 中只生成
  `learned_conical_fingerprint_input.json` 和
  `learned_conical_direction_shadow.json`，实际 shortcut 仍由既有透明启发式执行。
  未来即使将 manifest 字符串改成 `CAPPED_SHORTCUT`，没有第二把、单独冻结并哈希的
  real-trace promotion contract，运行器也不会授予控制权。
- 对齐的不只是“模型给一个方向”。MATLAB 现在会保留 learned state/direction
  distributions、APS、ensemble concentration/disagreement，并用 soft CAD state
  likelihood 与 soft visual prior 形成完整离散 center posterior；shadow planner
  复现仿真中的方向质量、reverse mass、10–25% 下分位、capture margin、35–65%
  radius cap 和显式 abstention。零置信时 shortcut 可以严格为 `0 mm`，不再强迫至少
  `0.25 mm`。
- 修复了会放大 sim-to-real 偏差的 belief/search 结构：center hypotheses 永远锚定
  在本轮原始接触点；shortcut 只改变当前机器人位置，不平移整个搜索域；其真实走过
  的线段与末端 negative/capture evidence 都更新同一个 posterior；模型用过以后不把
  posterior 重置为 uniform。熵网格只选距当前点不超过 `0.85 mm` 的连贯局部动作，
  并加入正 travel penalty，避免旧实验中“向错误方向跳走后再乱搜”的结构性放大。
- 两条不可变真机 cone trace 已用最终 runtime 离线重放：
  - `M12_event_trigger_insert_live_20260724_022333_540`：
    `outer_partial` 概率 `0.6507`，任务系方向 `261.41°`，
    APS size `4`，concentration `0.9866`，seed top-1 span `22.5°`，
    simulation transport gate 拒绝；
  - `M12_event_trigger_insert_live_20260724_023025_306`：
    `outer_partial` 概率 `0.5771`，任务系方向 `275.64°`，
    APS size `3`，concentration `0.9964`，seed top-1 span `7.5°`，
    simulation transport gate 通过。
  两次均明确写出 `hardware_control_allowed=false`，回放证据分别保存在对应 run 的
  `latest_model_shadow_replay.json`。
- 每次 live shadow 的 `summary.json` 还会保存模型融合后的 25 个全局锚定 grid
  hypotheses、归一化 posterior、影子 shortcut 的方向质量、reverse mass、分位数和
  cap 证据。这样下次实验可以直接比较“模型本来想怎么走”和 classical 实际动作，
  不必事后从一个方向角反推。
- 对“是否马上采真机数据 fine-tune”的当前结论是：先不要用 fine-tuning 掩盖合同
  错误。两个本应在 M12 孔口附近的起点都被判成较高概率 `outer_partial`，而模型仍给
  出很尖锐的方向 posterior；这首先指向 phase/sign、任务系到 WORLD 旋转、
  Bota measurement-origin 与 tip wrench、baseline/gravity、TCP/实际锥形运动或训练
  support 的 mismatch。先用已知 fixture center 给 real shadow trace 标出真方向，
  做 frame/origin/golden-vector 对照；这些通过后，才考虑 calibration 或小规模
  fine-tuning，并按实体 start/fixture session 分割数据。
- skill 已同步蒸馏这组教训：删除“ONNX 是唯一可接入形式”和“必须等 H3 全部完成才可
  做任何 H4”的过时绝对规则；保留 ONNX parity 优先，同时允许哈希绑定、无机器人
  socket 的 phase-boundary CPU PyTorch bridge 做 commissioning。明确 shadow 可与 H3
  改进并行积累证据，但 H5 控制 promotion 仍必须有稳定 classical residual/capture/
  terminal fallback；禁止强制最小 shortcut、禁止平移搜索域、禁止后验用后归零，并
  要求在 fine-tuning 前依次排查 identity、frame、origin、preprocess 与 training
  support。
- 离线验证全部通过：MATLAB R2026a `checkcode -id` 返回 `0` 个问题；Python
  `py_compile` 通过；两个 JSON 通过 `json.tool`；KUKA skill 通过
  `quick_validate.py`。本轮没有连接 KST、没有打开命令 topic、没有运动机器人。
  当前关键文件 SHA-256：
  - 主 MATLAB：
    `c2908fd93a53654a521c77be8684907d050d0807b7fb9d6560a9488dc10af8f9`
  - Python inference：
    `d5410f213934dba4ee44421d456ff7baa52f32cb771af05a7676e5e90d0ee67c`
  - deployment manifest：
    `5a0018c024fd379e46035967aa673944732b362b364fa59ae26241a5f374d04f`
  - real-device config：
    `2cbe19b9be5503396b0207a061e955bd78b66b3dbedf42b47beed117c087b7a8`

## 2026-07-24 — 新 clocking 工具后的坐标、数据、模型与四方法真机骨架闭合

### 物理坐标判断与实例边界

- 待重打的一体式 M8/M10/M12 flange–peg 使用 v1.5 `-15 deg` CAD construction
  clocking；孔座和升降台不移动。这个旋转只使选定 peg flat 法向与法兰 A/C 切向
  基准法向重合，不改变正六角的名义体积、质心或面内惯量。真实 FDM 质量、CoM、
  惯量、tip 高度和 runout 仍须按新 tool instance 重测，旧安装不能自动继承资格。
- “面1、棱1、面2都平行”不是正确的三维约束。peg 竖直棱只检查轴向/roll–pitch，
  不能确定 yaw；选定 peg flat 与法兰基准共同定义 yaw。孔座长边侧面跨越
  `X_F-Z_F`，因此它与该 peg/法兰平面垂直。最终控制只接受实测、右手、正交的
  `WORLD <- fixture`，不从照片发布 frame。
- fixture 原点仍为 M10 as-built 孔心，`+X_F` 指向 M12；照片从左到右是
  M12/M10/M8，故 `u_photo=-x_F`。名义 `[-45,0,+45] mm` 只作 CAD 初值，各孔实际
  中心和顶面必须重测。CAD `-15 deg` 不再加到 MuJoCo、历史方向 label 或动作上；
  旧打印件、未知 frame/clocking trace 保留但隔离，不通过“旋转标签”混入训练。

### 最小采样与污染防线

- 新增确定性的三尺寸俯视采样设计：
  `configs/fixture_multisize_conical_sampling_v1.json`、CSV 及 PNG/SVG。每尺寸
  26 次：16 个两圈错相 clean candidate、4 个冻结 holdout、4 个边界 quarantine、
  2 个 exact-repeat sentinel；三尺寸共 78 次，中心点不产生方向 label。离线点只是
  CAD-seeded local offsets，执行时先重定位到实测 as-built 孔心，再经实测
  `WORLD <- fixture` 变换。
- 新增 `run_m12_conical_sampling_campaign.m`。每点严格执行：
  `>=3 mm lift + two clear dwells -> air XY -> guarded touchdown -> full cone
  -> immediate upright/retract -> two clear dwells`，最后只在状态已知且 clear 时回到
  初始非接触位。原始 wrench/pose/phase、MAT/JSON/CSV/checkpoint 与点间 dashboard
  全部保存；不在线更新权重。
- 方向 label 只由 `measured as-built centre - measured touchdown` 在 fixture frame
  内计算。命令点、照片、模型预测和 capture outcome 都不能充当真值。tracking
  `>0.25 mm`、计划方向偏差 `>5 deg`、near-centre、no-contact、capture、guard、
  stale sensor、锥扫/回正/回撤不完整全部 quarantine。
- 当前 DirectServo 环内为 command-propagated pose/phase；pre/post robot pose 不能
  证明整圈轨迹。新增独立 cone phase/pose 资格门：只有时间对齐的 measured robot
  pose 或 external tip tracking 证明最大 phase error `<=1 deg`、tip-XY error
  `<=0.20 mm`、timestamp error `<=10 ms`，clean 点才可成为
  `PENDING_OFFLINE_QC` training candidate；否则自动 quarantine。
- 联合 lateral label budget 在 KST 前计算：
  `sqrt(socket-centre uncertainty^2 + TCP lateral uncertainty^2) + runout`，
  必须同时不超过 `0.30 mm` 和最小 clean 半径对应的 `5 deg` 方向偏差。这样单项
  “合格”但合成后超预算的数据不会进入训练。

### 独立真机计量与模型权限

- 保持 simulation source-of-truth
  `mujoco_unscrewing_probe_new/configs/real_tooling_v1.json` 逐字节不变，
  SHA-256=`03878ec282b8e9284b8733062d9ecf9132834069a83cad573755b0120617b356`。
  所有 post-print/post-install 字段迁到
  `configs/tool_metrology_commissioning_v1.json`，避免真机 null/qualification 状态
  污染冻结仿真几何谱系。
- 所有入口在 KST 前要求：一致的 physical tool/fixture/installation IDs、完整
  media-flange→tip 3-vector pivot/touch-off、TCP uncertainty/residual、runout、
  payload mass/CoM/SPD inertia、sensor measurement-origin→tip rigid transform、
  as-built centre/top plane、直立 TCP rotation，以及入口对应 authorization。当前
  模板仍为 PENDING/null/false，因此按设计拒绝真机运动。
- 干净仿真方向模型 v7 只用 canonical valid M8/M10/M12 数据训练；capture/incomplete
  分别剔除 93/144/166，failed-audit development 和 fresh final samples 均未回流。
  fresh8 的 M12 hardware-easy 域 PASS：167/180 valid，median/p90
  `0.93/3.29 deg`，accepted `>90 deg` 为 0；但 M8-source 域 p90
  `174.67 deg` 且有 2 个 accepted `>90 deg`，所以全局为
  `REJECTED_GLOBAL_AUDIT`。
- 真机 manifest 仅暴露 `M12_DEVELOPMENT_SHADOW_ONLY` 与 `SHADOW_ONLY`，没有
  promotion lane。MATLAB 现在以常量 SHA-256
  `1716bb00e1dfcd7b6863524d14d2e41cbbf7f572f49d7ce20a53369526e9acb9`
  在 KST 前钉住 manifest；manifest 再钉住 inference runtime、requirements、
  13 个模型工件和 5 个审计证据。模型只记录 posterior，默认运动永远由 classical
  cue 执行。

### 四方法共享任务闭环

- 新增三个薄入口：
  `run_m12_proposed_insert_demo.m`、
  `run_m12_pure_outward_insert_demo.m` 和
  `run_m12_improved_direct_retry3_demo.m`。它们与默认 Proposed+ 共用 touchdown、
  hard force/torque/state guards、physical capture requalification、0.1 mm 分段插入、
  terminal verifier 和同一 70 s policy budget。
- Proposed 是一次独立 cone + 授权 posterior shortcut + 2.0 mm local outward
  residual；Pure Outward 无 cone/模型，从原接触锚点执行 0.40 mm/rev、最大
  4.2 mm 的 Archimedean 搜索；Improved Direct 最多三次独立 cone/inference/ray，
  使用 `0,+0.25,-0.25 mm` 垂直调整。需要 learned motion 的两个入口在当前
  shadow manifest 下会在 KST 连接前失败，不会把 shadow 结果冒充 baseline。
- 这些文件完成的是“动作—观测—capture—插入—终止”软件结构闭合；闭环允许
  bounded failure，不保证成功。没有新打印件真机 run、成功率、承扭或反转拆卸主张。

### 验证与当前哈希

- 31 项 Python 静态/authority/sampling/baseline tests 全部通过；
  sampling generator `--check` 通过，验证 78 点、0 centre attempts；MATLAB
  7457 行 block stack 平衡。当前机器无 MATLAB/Octave，因此没有执行 `checkcode`
  或 MATLAB runtime；本轮未连接 ROS/Bota/KST，未发送机器人命令。
- 关键 SHA-256：
  - 主 MATLAB：
    `8168254919e69e072ceff268028eb330cd2ea8723ab8b9290b8b07605083b47f`
  - tool metrology template：
    `c6b0e604d53ba787b603589fc4e754bf596e2cd4b9586ca1b886aa9795d4dff2`
  - fixture commissioning template：
    `bd684115d8580e90694b26a33c8f1a4e3e3de014909349a594ce46c2a27797bf`
  - 78-point plan / top-view PNG：
    `4f84f2f117f1679d9cb19f6b50a1350dca31a5ceee9ea7c244a925cf2aac0c89` /
    `94577bfe6f94224643ce1ba80cd58417b85ad6ed7bc36e832f589938acf9c863`
  - deployment manifest / inference runtime：
    `1716bb00e1dfcd7b6863524d14d2e41cbbf7f572f49d7ce20a53369526e9acb9` /
    `762fb4e89f3c3754ae22907440ba08a293c00fdab99e9a17c5f8d19d4be0f1a4`
  - clean-data audit / untouched fresh8 final：
    `895f2e821b487c175370854de596210ade8e4a983d5ecf354820dff032782181` /
    `f12764904cd30b953e9ba627058a1f442fe1471870f3cc56f801e1b94fa4355d`

## 2026-07-24 — 最终 fail-closed 加固：实体底面、controller load 与 Sunrise 源码证据

### 实体 terminal 不再借用 touchdown 或 MuJoCo 深度

- `fixture_frame_commissioning_v1.json` 新增实体 M12 顶面到底面深度、标准不确定度和
  独立 bottom WORLD 点。M12 顶面点按
  `t_WF + R_WF * [45,0,0]^T mm` 计算，而不是误用 fixture 原点；bottom 交叉核对沿
  实测 `-Z_F` 扣除孔深，允许残差不超过 `0.25 mm`。
- 插入的绝对深度定义为
  `d = D_measured - (p_tip - p_bottom)^T z_F`：顶面约为 `0`，实测底面约为
  `D_measured`。插入命令沿 `-Z_F`，terminal band 由
  `[D_measured-0.30, D_measured+0.20] mm` 生成，并在 band 内重新做持续 bottom-force
  dwell。MuJoCo `6 mm` 只保留为不可变 CAD 参考，touchdown 高度不再充当深度零点。
- 物理底面、力事件、soft/hard wrench envelope 和 terminal 时间窗必须同时通过；
  到达命令深度、保护性停止或异常 `end` 都不计成功。

### payload 识别不再被偷换成 controller compensation

- 所有 full loop 在 KST 连接前都须完成一次共同的 controller-deployment
  attestation：每次 Sunrise project/application/LoadData redeploy 生成新 UUID，
  预登记 station/IP/application/deployment UUID/项目源码 SHA，从导出项目包现场
  重算 SHA，并要求 running-app readback 同时绑定实体工具和安装 session。readback
  必须晚于本次安装与 deployment、不得未来日期，并在运行前 `<=900 s`。
- 同一 attestation 在 Bota/ROS preflight 后、KST 连接前，以及 baseline/dashboard
  完成后、首个机器人命令前各重新验证；deployment UUID、evidence SHA 和 project
  SHA 在同一 run 内不得变化。首个运动前还要求当前 readback age 加 `600 s` hard
  task horizon 不超过 `900 s`，避免“检查时未过期、任务中途已过期”的边界穿透。
- 共同 attestation 通过后，controller-load handling 至少一条路径合格：
  1) Sunrise 非零 LoadData；或 2) 最终实体负载下、独立测得的代表性 full-cone
  phase/tip tracking。两条路径是包含式 OR；两条都通过也合法。tracking evidence
  必须与当前 attested deployment UUID 相同，不能在 controller redeploy 后复用。
- 非零 LoadData 路径还须预登记 LoadData 名称与部署 UTC，要求 readback 的
  application/load/source SHA、实体工具 ID、安装 session 和 payload evidence hash
  全部一致。readback 还须记录 `RUNNING` 状态、方法、UTC 和 controller station
  identity。
- mass、CoM 和 inertia readback 必须与 hash-qualified payload evidence 数值一致。
  可编辑 JSON 容差只能比代码硬上限更严，不能超过 `1 g / 0.5 mm /
  1e-6 kg m^2`。当前模板仍为 `PENDING/null/false`，因此按设计在 KST 前拒绝 full
  loop；本次没有伪造 Sunrise 项目包或 readback。
- tracking 路径绑定当前 physical tool instance、installation session、活动 MATLAB
  源码 SHA、锥扫倾角、命令频率、扫描时长、phase bins 和 feedback policy。旧工具、
  旧源码或 command-propagated 轨迹不能静默继承资格。

### 数据、模型和四方法的最终权限边界

- 78 点采样仍按每尺寸 `16 clean + 4 frozen holdout + 4 quarantine +
  2 sentinel` 分区；label 只来自实测 as-built center 与实测 touchdown。没有独立
  measured pose/phase 证据时，所谓 clean 点只能 quarantine/diagnostic，绝不自动进入
  fine-tuning。
- clean-v7 的全局 fresh8 结论仍是 `REJECTED_GLOBAL_AUDIT`；只允许
  `M12_DEVELOPMENT_SHADOW_ONLY`，manifest 没有 promotion lane。默认 Proposed+
  执行 classical cue，v7 只记录 posterior；需要 learned motion 的 Proposed 与
  Improved Direct x3 在当前 manifest 下会在 KST 前失败。Pure Outward 虽不依赖模型，
  仍受相同实体计量、capture、insertion 和 terminal gate 约束。
- 因此现在闭合的是可失败、有限时间、可追溯的
  `observation -> routing -> capture -> measured-bottom terminal` 软件/证据链；不是
  新打印件已经真机跑通，更不是拧松任务已经完成。

### 最终静态与离线验证

- 32 项 sampling/runtime-authority/baseline 静态测试通过；78 点 generator
  `--check` 通过；`configs/` 下 12 份 JSON 及 deployment manifest 可解析；Python
  `py_compile` 通过。主 MATLAB 共
  7935 行，独立只读审查确认 block stack 平衡。
- v7 的 M12 固定 trace 再回放与冻结 JSON 逐字节一致，SHA-256 =
  `0f7a65b9fcf581e7e91aa4f8e0f3929af8b7fa425c3746adc9214321d2c548fa`，
  `hardware_control_allowed=false`。
- 中英文 Marp 各 39 页均可生成 HTML；英文 CJK 数为 0。由于本机没有
  MATLAB/Octave，也没有 KST/机器人运行环境，本次未做解释器级执行、Sunrise
  readback 或真机运动。
- 当前关键 SHA-256：
  - 主 MATLAB：
    `727c7d32717eaae1085fb27b7655d45f361bd123e479c26acb8d99a963810823`
  - tool metrology / fixture commissioning：
    `f469197cb1c7239e3ca342ac6a3c0809ead2c6ffc225704e9724555f191aaa2a` /
    `656ca0e217e6a48d39ee482f1d3e4d01109545a72b93398d7a7a123c2b310a7f`
  - 78 点 plan / top-view PNG：
    `4f84f2f117f1679d9cb19f6b50a1350dca31a5ceee9ea7c244a925cf2aac0c89` /
    `94577bfe6f94224643ce1ba80cd58417b85ad6ed7bc36e832f589938acf9c863`
  - deployment manifest / inference runtime：
    `1716bb00e1dfcd7b6863524d14d2e41cbbf7f572f49d7ce20a53369526e9acb9` /
    `762fb4e89f3c3754ae22907440ba08a293c00fdab99e9a17c5f8d19d4be0f1a4`
  - 中文 / English Marp：
    `a4f85542f0e2f112c953a46276c749db4f491646e8a93772ed49a6cac5a6cf64` /
    `421ad4d07430373d8c6804371ed2c56376615ccd58604c291e65483d912381b9`

## 2026-07-24 — 去除活动 `CAPPED_SHORTCUT` 前提；经典锥扫方向直接执行

### 动作权限简化

- `run_m12_proposed_insert_demo.m` 现在选择
  `FULL_LOOP_PROPOSED_CLASSICAL`：一次完整锥扫，经透明
  `force + commanded downward progress` 启发式得到 `0--1 mm` 有界捷径，再执行
  `2.0 mm` local outward residual。
- `run_m12_improved_direct_retry3_demo.m` 现在选择
  `FULL_LOOP_IMPROVED_DIRECT_RETRY3_CLASSICAL`：最多三次独立
  `cone -> classical direction -> bounded ray`，锚点仍为
  `0,+0.25,-0.25 mm` 垂直调整。
- active mode 列表中不再有 `*_CAPPED` 入口，也不再在 KST 连接前要求
  `CAPPED_SHORTCUT` promotion contract。clean-v7 仍可在每次锥扫后做 shadow
  对照，但模型 manifest/hash/推理失败只记录 `learned.error`，不会阻断或改写经典动作。
- hard force/torque、stale wrench、command owner、完整 TCP/payload/fixture/depth、
  controller deployment、有限时间/行程、physical capture、分段插入和 bottom terminal
  gate 全部保留。这次只移除了未获证据支持的 learned-motion 前提，没有移除防撞和
  坐标资格化。

### 内侧单点支撑的新增物理风险

- 如果 peg 只搭在 socket 内侧倒角/内缘，锥扫最低支撑点转入孔内时可能失载。
  当前 `runConicalFingerprint` 只有高载荷 phase hold；低载荷时仍会追踪 `3 N`，
  最多允许 `0.40 mm` residual downward press，因此现版本可能把失载变成未显式分类的
  下沉/捕获，不能把该段无条件当作完整方向 fingerprint。
- 合理的下一控制改动应是低力接触模式机：
  `CONTACT_SCAN -> LOW_FORCE_PHASE_HOLD -> RECONTACT 或 APERTURE_CANDIDATE`。
  触发时冻结 cone phase/XY，限制一次很小的重新接触 Z 行程；若接触恢复则带标签继续，
  若持续失载并伴随下沉则提前结束 fingerprint，转交 upright physical capture
  requalification。采样 campaign 中此类 incomplete/capture trace 必须 quarantine，
  不得进入 direction loss。
- 在该低力模式实施和真机小载荷验证前，内缘起步的全锥扫仍属于已识别 commissioning
  风险；静态测试通过不等于该接触模式已在真机安全跑通。

### 验证与哈希

- 32 项 Python sampling/runtime-authority/baseline 静态测试通过。当前机器无
  MATLAB/Octave；本轮未连接 ROS/Bota/KST，也未发送机器人命令。
- 主 MATLAB 共 7895 行，SHA-256：
  `ef8b5e1fcf4191f508c71a19de371dd57ba4a117c94c0c6f33db42d4707b87c7`。
- Proposed / Improved Direct wrappers：
  `c0114a01f072c8f9aa29e2a2765753a930ab9e6d3a08d1315dc61b109a64b061` /
  `45044854d8f149d1f0d0e08d2bde72acdcc60db1b52620fc7555911530a8c1ee`。
- 中文 / English Marp 已同步经典动作权限表述，SHA-256：
  `776d4c0142d4ccd229ffd8030f6709618f83170cfdea4689246045d09c66bcf4` /
  `499b0c3218d7229fe1888e6049780565d4e3b9f4f7ec878da0d1405c81e45477`。

## 2026-07-24 — 锥扫低力接触模式闭环

- `runConicalFingerprint` 新增
  `CONTACT_SCAN -> LOW_FORCE_PHASE_HOLD`：首次低法向力即冻结锥扫相位、
  WORLD XY 和姿态；第二个连续低力样本后，才以 `0.20 mm/s` 做最多
  `0.25 mm` 的 WORLD -Z 重获接触。阈值不是固定猜测，而由本次 150 点静止
  Bota 基线噪声与已登记 `3 N` 目标共同计算；恢复使用更高阈值和 3 点滞回。
- 若在不超过 `0.05 mm` 的重获位移内快速恢复，则从原相位继续并标记
  `QUICK_RECONTACT_RESUMED_TAGGED`；这种轨迹可完成在线经典路由，但采样 campaign
  一律转入 quarantine，不能成为方向训练样本。持续低力或有意义的下沉改判为
  `APERTURE_CANDIDATE`，不再补完 360° 指纹。
- aperture candidate 会保持当前 WORLD XYZ，在触发方位上平滑减小倾角至直立；
  DirectServo 停止并取得新鲜实测位姿后，才使用已有的“持续 Bota 接触 +
  相对表面实测深度 + 额外 no-lift 下沉”做物理复核。复核通过才可进入共享
  insertion verifier；失败则停在当前位姿。Proposed/Proposed+、Improved Direct
  retry 和 sampling 三条消费者均在 learned/classical direction 之前检查该状态。
- 低力阶段若横向力或测量原点力矩越过“基线噪声 + 目标力/目标力臂”尺度的软阈值，
  只发一次 `+0.10 mm` relief，随后由既有 DirectServo cleanup 停止；原
  `20 N / 0.75 Nm` hard guards、stale-wrench、时限和空间边界均未改变。
- 锥扫输出现在显式保存 `schedule_completed`、
  `direction_fingerprint_valid`、`training_fingerprint_valid`、
  `mode_transition_outcome`、低力阈值/次数/位移，以及独立 post-stop 位姿。
  提前扶正时，数组只保留真实执行的 schedule 行，不再以未执行行或扶正后位姿
  污染最后一个扫描相位。
- 新增 `tests/test_m12_conical_low_force_guard_static.py`；全部 39 项 Python
  静态/权限/采样测试通过。当前机器没有 MATLAB/Octave，本轮没有连接
  ROS/Bota/KST，也没有发送任何机器人命令；解释器级检查和小载荷真机调参仍待现场完成。
- 当前主 MATLAB SHA-256：
  `94a0931dd5b94290bec0cffdbb51dd2939ed79248663e9fd6a355a6be082e460`。

## 2026-07-24 — 五个清晰真机入口与右手俯视坐标修正

### 入口重构

- 活动真机接口改成五个无参数、可分别 F5 的公开入口：
  `run_m12_full_loop_proposed_plus_entropy_grid.m`、
  `run_m12_full_loop_proposed_local_spiral.m`、
  `run_m12_full_loop_baseline_pure_outward_spiral.m`、
  `run_m12_full_loop_baseline_improved_direct_x3.m` 和
  `run_m12_sampling_conical_fingerprints_26pt.m`。
- 8529 行公共物理实现移入 `execute_m12_contact_engagement_policy.m`。每个入口只
  选择一种内部 mode；touchdown、低力接触模式、hard guards、capture requalification、
  insertion 和 terminal verifier 不复制。旧五个文件名保留为无参数短 alias。
- 内部 mode 和输出目录前缀同步改为方法语义名。每个 run 的 result/manifest 记录
  public entrypoint、绝对路径与 SHA，避免文件名与实际 policy 脱节。
- Proposed+ 是一次透明经典捷径加有限 entropy grid；Proposed 是同一透明经典捷径加
  local outward spiral；Pure Outward 无锥扫/无模型；Improved Direct 最多三次
  独立锥扫/经典方向/ray。四个 full-loop 均不要求 `CAPPED_SHORTCUT`，v7 只做
  可选 shadow。旧的未调用 CAPPED verifier 已从活动 engine 删除。

### 俯视坐标修正

- fixture 定义不变：`+X_F` 从 M10 指向 M12，`+Z_F` 从桌面向上，
  `+Y_F=+Z_F×+X_F`。观察者位于 `+Z_F` 一侧俯视时，`+X_F` 在图中向左，
  所以 `+Y_F` 必须向图下。
- 数学显示坐标改为
  `(u_right,v_up)=(-x_F,-y_F)`；只有像素向下为正的图像坐标才满足
  `v_pixel_down=+y_F`。生成器、JSON/CSV 和 PNG/SVG 已一起重生。
- 该改动只翻转照片投影字段；`local_offset_fixture_mm`、`fixture_xy_mm`、
  向心真值与 `WORLD←fixture` 动作字段保持不变，动作相关字段前后 SHA 均为
  `15a53189f74cd334b6ab71605f080b4806f251ec8bcddcb66418a4c88cb1f9e8`。
- 新增 `docs/FIXTURE_TOP_VIEW_FRAME_AUDIT.md`，并把同一修正版图片继续用于
  中英文 PPT。

### 数据治理与验证

- 采样入口现在明确：任一 `LOW_FORCE_PHASE_HOLD`、`QUICK_RECONTACT` 或
  `APERTURE_CANDIDATE` episode 都 quarantine，不进入 direction training；
  full-loop 仍可在新鲜物理复核通过后继续，诊断 cue 不替代 terminal verifier。
- 41 项 real-device 静态测试通过；78 点生成器 `--check` 通过；MuJoCo 工程
  356 项测试通过、2 项历史 deck snapshot 跳过；相关 Python `py_compile` 通过。
  中英文 Marp 均为 40 页，英文 CJK 数为 0。
- 当前机器没有 MATLAB/Octave；本轮没有连接 ROS/Bota/KST，也没有发送机器人命令。
  五个入口已完成静态可执行结构验证，但新打印件的 post-install commissioning 仍需
  现场填写后才能越过 KST 前门。
- 当前关键 SHA-256：
  - shared engine：
    `f26b4e526f1b645a754da26926b2ac6d6d4fec02b79ae4c7512ef0dd34b1dcf0`
  - 五个公开入口：
    `fea07c1039836fc6fc7ac875b4051d7ade37d2727725b670611c3fc897c31996` /
    `3f8deb37d8e015f63ca85f21d1df6c1fe19d3cb4a1e8c8cca86f3a57b50c80c4` /
    `e48e941857ff00615583b8ec906b50324033a0c7a31f2e288426183c06004f59` /
    `791852b66b5de479fb471dc78d08807fb6f7425225594725d8cc0c5935a8ee35` /
    `31073c95796d6477a67354757e9a0caaa4a2ae8a10e3dbd882e138c737d5974e`
  - 78 点 plan / 修正俯视图：
    `d728c57f63e80d6259ffe1522ecd32e3d9902b39e685a9c8d1e1c647d3685fb2` /
    `1d6bdba1cac56a6304e7c531def33b140dbb0f87586cc90fa8c9f63a6cdfb42b`
  - deployment manifest 未变：
    `1716bb00e1dfcd7b6863524d14d2e41cbbf7f572f49d7ce20a53369526e9acb9`
  - 中文 / English Marp：
    `f0e8df496f2a23d325a060865e7c81f9061c809372ade0aa6a8e1acee2de3f4f` /
    `9d1749a9c5a2f7bfe731354812272cc426b4da073dbbdd2af34a0b1f0dc88a9d`

## 2026-07-24 — 活动 MATLAB 根目录收敛为六个文件

- 将五个没有独立控制逻辑的旧名 wrapper 移至
  `archive/legacy_entry_aliases_20260724/`：
  `run_m12_event_trigger_insert_demo.m`、
  `run_m12_proposed_insert_demo.m`、
  `run_m12_pure_outward_insert_demo.m`、
  `run_m12_improved_direct_retry3_demo.m` 和
  `run_m12_conical_sampling_campaign.m`。
- 活动根目录现在严格只含一个共享核心和五个公开入口：
  `execute_m12_contact_engagement_policy.m`、
  四个 full-loop 方法入口与一个 26 点采样入口。没有复制或删除 touchdown、
  低力模式、guard、capture、insertion 或 terminal 实现。
- `README.md`、`archive/README.md` 和静态测试已同步。测试现在明确断言旧名不在
  活动根目录、归档文件存在且仍只转发到对应的新入口。
- 41 项 real-device Python 静态测试全部通过。本次未连接 MATLAB/Octave、
  ROS/Bota/KST，也未发送机器人命令。

## 2026-07-24 — Proposed+ 启动期 JSON null 与 Running 生命周期修复

- 现场复现：
  `run_m12_full_loop_proposed_plus_entropy_grid` 在首个 KST 连接之前进入
  `validateM12Tooling`，并由 `strtrim` 抛出
  `MATLAB:strtrim:TooManyDimensions`。该次记录为
  `data/M12_proposed_plus_entropy_grid_20260724_153320_700`，其中
  `motion_commanded=false`、`kst_opened=false`，所以“F5 没反应”不是控制器拒绝运动，
  而是本地 commissioning JSON 验证提前失败。
- 在现场 MATLAB R2026a 上确认：`jsondecode(null)` 是 `double [0x0]`；
  旧写法 `char(string(value))` 会生成 `char [0x0x0]`，随后 `strtrim` 因输入超过二维
  而崩溃。共享执行器新增 `scalarTextOrEmpty`，把 JSON null 明确正规化为空字符行，
  并拒绝非标量文本；所有 commissioning 身份、操作者、UTC、证据路径/hash 和
  controller readback 文本都改走同一入口。
- 修复后，当前模板会按真实状态给出
  `M12Demo:ToolMetrologyPhysicalIdentityMissing`，不会再冒出 MATLAB 内部维度错误。
  `configs/tool_metrology_commissioning_v1.json` 与
  `configs/fixture_frame_commissioning_v1.json` 仍明确为
  `PENDING/null/authorization=false`；本次没有伪造实体 ID、TCP/payload/CoM/惯量、
  sensor-to-tip、fixture frame、孔深、Sunrise readback 或 full-loop authorization，
  因而该入口仍应在任何运动前 fail closed。
- 启动期异常清理补齐：即使尚未创建 KST 运动对象，只要操作者已经把
  MatlabToolboxServer 置为 Running，执行器会对 `192.168.1.147:30001` 只写 ASCII
  `end`，不写任何 pose/velocity/force 命令；SmartPAD 是否回到 Selected 仍由操作者
  观察并在 run record 中标记为 pending。
- 验证：
  - MATLAB R2026a `checkcode` 完成，0 个解析错误；只有既有风格/未用变量提示；
  - MATLAB 现场确认旧转换的尺寸确为 `[0 0 0]`；
  - `test_m12_real_baselines_static`、`test_m12_sampling_campaign_static` 与
    `test_m12_conical_low_force_guard_static` 共 36 项通过；
  - 本轮没有调用任何真机入口，没有连接 ROS/Bota/KST，也没有发送机器人运动命令。
## 2026-07-24 — Proposed+ 公共入口恢复现场跑通的直接执行内核

- `run_m12_full_loop_proposed_plus_entropy_grid.m` 不再进入依赖未填写
  tool/fixture commissioning JSON 的新共享引擎；F5 直接调用已在同一
  KUKA、Bota、MatlabToolboxServer 和近孔人工起点条件下跑通过的单文件任务内核。
- 直接流程保持为：WORLD -Z 微段触地、连续 15° tip-centered 锥形指纹、
  约 3 N 接触运输捷径、有限熵网格、倒角捕获、0.1 mm 分段插入、持续洞底事件、
  KST `end`。没有 preview、token 或正常自动回折叠位。
- 归档内核改名为与文件一致的
  `run_m12_event_trigger_insert_old_demo`，并修正归档后工程根目录解析；运行数据仍写入
  `group_meeting/real_device/data/`。
- 打印件 STEP 绑定更新为当前
  `362018e71e4bb80ea07640d6fb7bc0ddbe6178c25046f0283741e5a49e52a4fe`。
- 未删除运行时保护：单一 KST/ROS 命令所有者、Bota 身份/配置/新鲜度、
  有限数、20 N/0.75 Nm、关节余量、局部工作空间、任务超时，以及成功或异常的
  KST `end` 均继续执行。

## 2026-07-24 — Proposed+ 切回新共享内核并采用旧内核现场兼容层

- `run_m12_full_loop_proposed_plus_entropy_grid.m` 已不再调用归档的旧单文件内核；
  公开入口现在直接调用
  `execute_m12_contact_engagement_policy('FULL_LOOP_PROPOSED_PLUS_CLASSICAL_ENTROPY')`。
  因而连续 15° tip-centered 锥形、接触跟随捷径、有限熵网格、倒角捕获、
  分段插入、底部持续事件、实时图表与统一 run record 均由新共享内核执行。
- full-loop 新增明确的 predecessor-compatible commissioning lane：
  沿用已跑通旧内核的 92 mm 轴向 TCP（声明不确定度 ±4 mm）、人工静止近孔起点、
  起始姿态 WORLD 映射 `diag([1,-1,-1])`、当次 touchdown/capture 建立局部表面基准、
  6 mm 分段插入。缺失物理 tool/session ID、完整 payload/CoM/惯量、
  Sunrise LoadData/readback、精确 pivot 和 fixture metrology 会如实写入 manifest，
  但不再阻止该 full-loop 入口连接 KST 和执行运动；26 点数据采样入口仍走严格门。
- 将 shared policy 的 70 s 时钟改为 480 s。此前真实锥形、捷径和熵网格运行可超过
  70 s，旧时钟会把仍在正常推进的任务当成超时；独立 600 s 整体运行上限仍保留。
- 没有把未完成标定伪称为 qualified：连接日志改写为 active/provisional TCP，
  WORLD wrench 映射标为 old-kernel provisional，manifest 同步写入兼容 authority
  与 claim boundary。
- 仿真训练模型仍仅作为 shadow telemetry；缺少 PyTorch、模型推理失败或 trust-anchor
  不通过时会被捕获并回退到 classical cone-direction + shortcut + entropy-grid，
  不会阻断真机动作。本轮没有把尚未通过 sim-to-real 验证的模型提升为运动 authority。
- 仍保留：Bota SN001349 身份、驱动配置与数据新鲜度，单一 ROS/KST 命令所有者，
  20 N / 0.75 Nm 硬限，关节余量，任务局部工作空间、跟踪与有限值检查，
  以及正常/异常 KST `end`。
- 无运动验证：
  - MATLAB R2026a `checkcode` 成功解析共享内核，入口和内核均解析到活动目录；
  - 三组 real-device 静态测试共 37 项通过；
  - 本轮未连接 ROS/Bota/KST，未发送机器人运动命令。
- 本轮文件 SHA-256：
  - `execute_m12_contact_engagement_policy.m`：
    `8c58263ac6741f8bb7648198f231c627019f51c23481ce344dc1a815bc709161`
  - `run_m12_full_loop_proposed_plus_entropy_grid.m`：
    `101690d33bb27a225ca63d95f2c9bf7970fc977cbf2f53e64813feb6035134df`
  - `tests/test_m12_real_baselines_static.py`：
    `dd189b9c1c0daa3a7a8b036b7256c4f8186f6c9a9ef029f912c48cca0b2c81b3`

## 2026-07-24 — 首动作前 KST tcpclient 无限等待修复

- 现场 run
  `data/M12_proposed_plus_entropy_grid_20260724_231937_682`
  已通过 root/Bota/owner/KST/start-pose/baseline/dashboard，并记录
  `OLD_KERNEL_COMPATIBILITY_FIRST_MOTION_READY`，但机器人没有动作。
- 证据边界很明确：`commands.csv` 只有表头，没有首个
  `TOUCHDOWN_001,ISSUED`；KST TCP 仍为 ESTABLISHED，最后一次收发停在进入
  `moveOne` 的首个 `Eef_pos` 状态请求。因此这不是接触阈值、运动包络或
  event-trigger 拒绝，而是任何运动命令发出前的 KST 请求/回复阻塞。
- 本机没有可用的 legacy Instrument Control Toolbox `tcpip`，KST 实际走
  MATLAB R2023+ `tcpclient` 回退；旧工具箱却继续混用 `fprintf/fgets`，
  且回退只设置 `ConnectTimeout`，未设置响应 `Timeout`。
- KST 客户端兼容层已更新：
  - `tcpclient` 现在显式使用 LF terminator 和 8 s 有限回复超时；
  - 新增 `kstWriteLine` / `kstReadLine`，tcpclient 使用原生
    `writeline/readline`，legacy tcpip 仍使用原接口；
  - tool attach、EEF/joint getter、blocking relative line move，以及
    DirectServo start/stream/stop 的关键通信均改走这两个统一入口。
- shared core 的 `moveOne` 复用上一条已测 post pose；第一条命令复用几秒前
  刚采集的最终 stationary pose，删掉首命令前多余的 `Eef_pos` 往返。每个动作
  完成后的 EEF/joint 实测、tracking/workspace/joint checks 仍保留。
- 验证：
  - 38 项 real-device 静态测试通过；
  - MATLAB 确认 `tcpclient` 提供 `Timeout`、`writeline`、`readline`；
  - shared core、公开入口和 KST 修改文件均完成 MATLAB 解析；
  - 本地回环服务端测试因本机缺少 Instrument Control Toolbox 的
    `tcpserver` 无法建立，仅此项未执行；
  - 本轮诊断和验证没有发送机器人运动命令。
- 关键 SHA-256：
  - shared core：
    `df70a17f2e8248e346bd4f3f58dd22c729cb77f3e714a6f97c36ee0d9c4d5fed`
  - `net_establishConnection.m`：
    `3174046404733e1d32858fa145de6ed0cba18e7e86cfc160c8a4a231afe414dc`
  - `kstWriteLine.m`：
    `7cddd5039c77a24d572e1ce97ad67dd3989efa10b3ea5c4ab9b01ef46908ebb7`
  - `kstReadLine.m`：
    `eacb62243ddfa4e1e907a529612427e9d17cdaba61b0e97fc532b3a410b46cd9`

## 2026-07-24 — Bota 已恢复但 latched stale 诊断导致两步后误停

- 现场 run
  `data/M12_proposed_plus_entropy_grid_20260724_234543_044`
  已越过首动作卡点并完成两条真实 WORLD -Z 指令：
  `TOUCHDOWN_001` 与 `TOUCHDOWN_002` 均在 `commands.csv` 中记录
  `ISSUED/DONE`，累计实测下降约 0.173 mm。
- 随后停止原因为
  `M12Demo:DiagnosticStale: diagnostic level 2: Bota frame stream is stale`，
  不是接触阈值、力硬限、tracking 或 KST acknowledgment。
- 该停止是诊断时间尺度不匹配造成的假阳性：
  Bota 驱动以 1 Hz 发布并 latch `/diagnostics`，一次超过 0.25 s 的 USB
  主机侧帧间隔会把 `Bota frame stream is stale` 保持到下一次诊断；
  但本次 `raw_wrench.csv` 在 `23:46:11.931` 已收到 age 0.0136 s 的新鲜
  wrench，程序却在 `23:46:11.997` 立即读取到尚未被下一条 1 Hz 诊断覆盖的
  stale 状态。现场只读复查时驱动已恢复 level 0、约 199.97 Hz、
  frame age 0.000091 s。
- shared core 现在只对身份匹配、诊断时间戳新鲜、level 2 且消息精确为
  `Bota frame stream is stale` 的单一情形执行最多 2.0 s 的有界恢复确认。
  下一条健康诊断到达后记录 `TRANSIENT_BOTA_STALE_RECOVERED` 并继续；
  若 stale 持续超过窗口则仍终止。
- hardware ID、配置契约、其他传感器错误、非有限 wrench、wrench 年龄/时间戳、
  20 N/0.75 Nm 硬限及所有运动跟踪检查没有放宽。最终洞底 hold 同样使用该
  有界恢复确认，避免任务已经完成后被一条旧的 latched stale 否决。
- 验证：
  - 三组 real-device 静态测试共 39 项通过；
  - MATLAB R2026a 成功解析 shared core；`checkcode` 仅有既有风格提示，
    新增诊断恢复段没有解析告警；
  - 本轮只读诊断与离线测试没有连接 KST，也没有发送机器人运动命令。
- 本轮文件 SHA-256：
  - `execute_m12_contact_engagement_policy.m`：
    `2316c691f294e697bb17f5e4f8ab808a2346e5fe484d400b6169944cd2bb3757`
  - `tests/test_m12_real_baselines_static.py`：
    `b0b6e755eafdd543189957798b0944ecae66a2261bcc08b25ff9b46194402d4a`

## 2026-07-25 — 锥形起步短暂卸载不再误判孔口，放宽可恢复接触链

- 现场 run
  `data/M12_proposed_plus_entropy_grid_20260725_000357_220`
  已完成 touchdown：表面候选 Z `294.707 mm`，峰值相对力 `4.043 N`，
  确认窗口 WORLD* Fz 中位数 `3.908 N`。随后并非力硬限、KST、
  tracking 或 Bota stale 终止；锥形在 RAMP 仅 `0.07595 deg` 时发生一次
  短暂卸载。
- 旧状态机的恢复分界过窄：法向力在有界下压 `0.115 mm` 后已经恢复，
  但因为超过旧 `0.050 mm` quick-recontact 分界，被误标为
  `APERTURE_CANDIDATE_UPRIGHT_HANDOFF`；回正后的实际深度只有
  `0.040 mm`，物理孔口复核自然失败，任务因此提前结束。
- 低力状态机现在采用与旧真机代码/DMPC 工程节奏一致的
  “持续判定 + 滞环 + 饱和恢复”：
  - 瞬时低力需连续 5 个新鲜样本才开始下压恢复；
  - 恢复需达到目标力 55% 并连续 4 个样本；
  - 完整 `0.25 mm` 有界恢复带内只要恢复接触，就标记后回到原锥形相位，
    继续完成 15 度连续一圈；不再把 `0.05--0.25 mm` 的夹具/打印件顺应
    瞬态当成孔口；
  - 只有压满 `0.25 mm` 仍持续失载时才提出孔口候选，并继续走现有的
    实测 Z + Bota 物理复核。
- 用本次真实 `conical_directservo_trace.csv` 离线重放新判定：
  旧逻辑在 `0.115 mm` 返回孔口候选；新逻辑在 hold 第 22 个样本、
  约 `0.105 mm` 恢复位移、法向力 `2.161 N` 时得到 4 连续恢复样本，
  结果为 `QUICK_RECONTACT_RESUME_TAGGED`，应回到 RAMP 继续转圈。
- 同步放宽后续“可恢复”包络，避免完成锥形后又因打印件顺应或小抖动
  逐步卡住：
  - 接触保持目标 `3.5 N`，死区 `0.75 N`；
  - 贴面捷径/熵网格软卸载为 `10 N -> 7 N` 滞环，延长到
    120 个 transport actions / 75 s，允许 5 次连续微小进度；
  - capture 软包络为 `12 N / 0.40 Nm`，插入软包络为
    `12 N / 0.35 Nm`；
  - 触探/插入 tracking 容差 `0.20 mm`，插入最小进度
    `0.005 mm`，允许 5 次停滞后才判定不可推进。
- 独立的 `20 N / 0.75 Nm` 硬限、有限值、通信新鲜度、单一命令所有者、
  关节余量和任务局部工作空间没有取消。软包络触发时先卸载/重试；
  硬限或不可恢复通信错误仍结束 KST application。
- 无运动验证：
  - 新低力/软包络静态测试 8 项通过；
  - 全部 real-device 测试 45 项通过；另 1 项不可执行，因为冻结的旧
    shadow replay 清单仍引用不存在的
    `/home/njy/njy_ws/.../conical_directservo_trace.csv`，与本轮控制修改无关；
  - MATLAB R2026a `checkcode` 成功解析 shared core，
    `parser_like=0`；
  - 本轮仅重放既有日志并做静态检查，没有连接 KST 或发送运动命令。
- `00:34` 首次重跑在连接 KST 前被
  `ConicalReliefWorkspaceContract` 拒绝：15 度六角支撑几何最大抬升
  `1.195 mm` 加新残余卸载 `0.900 mm`，合计 `2.095 mm`，超过仍为
  `2.000 mm` 的旧 task-local +Z 包络。现将该包络同步为 `2.250 mm`；
  这只消除配置自相矛盾，力/力矩硬限及其他守护未变。
- `00:36` 真实重跑已到锥形 RAMP `12.369 deg`，并非刚开始或机械卡死；
  六角支撑补偿已把 TCP 从 touchdown 随倾角抬升约 `0.734 mm`，说明几何
  卸载通道正常。退出原因为单帧横向力 `3.580 N` 刚超过旧
  `3.500 N` 障碍阈值，代码主动再抬 `0.100 mm` 后结束；当时总相对力
  仅 `4.288 N`、力矩 `0.0589 Nm`，远低于硬限。
- 将锥形低法向力阶段的横向障碍阈值由目标力 `1.0x` 改为 `2.0x`
  （本次为 `7.0 N`），力矩阈值由目标 tip moment `0.5x` 改为 `0.9x`，
  且由单帧触发改为连续 3 个新鲜样本才执行一次 +Z relief 并结束。
  按本次 trace 重放，`3.580 N` 不再触发；同期法向力已经连续 4 帧恢复，
  状态机会回到原 RAMP 相位继续完成锥形。

## 2026-07-25 — 00:42 锥形低力死区修复与三倍软窗口

- 现场 run
  `data/M12_proposed_plus_entropy_grid_20260725_004205_101`
  已完成 touchdown，并在锥形 RAMP 到达 `10.254 deg`；退出不是力/力矩硬限、
  KST、tracking 或 Bota stale，而是
  `M12Demo:ConicalLowForceEpisodeClock`。
- 根因是低力恢复只看 WORLD* 法向分量。倾斜后的实测总相对力长期约
  `3.3--3.5 N`，表明接触仍然存在，但法向分量落在
  `0.875--1.925 N` 的滞环死区；旧代码既不恢复，也不累计未恢复样本去执行
  `0.25 mm` 有界再接触，最终在 3 秒观察钟结束。
- 状态机现在采用“法向或总接触载荷任一恢复”：
  法向达到 `1.925 N`，或总相对力达到目标 `3.5 N` 的 `70%`
  (`2.45 N`)，连续 3 个新鲜样本即回到中断的锥形相位。未恢复样本采用
  连续计数，不再悬停在死区。
- 按同一真实 trace 离线重放，新状态机在低力 hold 第 4 个样本、
  `0.0818 s` 内恢复；当时法向 `2.279 N`，无需下压到完整再接触带，
  因而不再触发 3 秒退出。
- 可恢复软门限同步放宽：
  - 单次低力观察 `3 -> 9 s`，累计预算 `20 -> 60 s`；
  - 横向阻塞 `7.0 -> 10.5 N`，力矩阻塞约 `0.2898 -> 0.483 N m`，
    仍需连续 3 个新鲜样本；
  - 锥形软卸载滞环 `10/7 -> 12/9 N`，单次/累计卸载预算
    `3/20 -> 9/60 s`。
- 独立硬限仍为 `20 N / 0.75 N m`，通信有限值/新鲜度、单命令所有者、
  关节和任务局部工作空间守护未改变。
- 无运动验证：低力专项 8 项通过，real-device 静态测试 40 项通过。
  当前 shell 没有可调用的 MATLAB executable，因此本轮不能从 shell 执行
  `checkcode`；修改均为局部 MATLAB 表达式/结构体字段变更。
- 活动源码 SHA-256：
  `b9fff22c23c424facd5979c6420b7c21f58bb9c3906666ff9e753c5fa55ad0b1`。

## 2026-07-27 — 两次实机失败的共同根因与四方法统一 fast-but-bounded 修复

- 本轮交叉审计了用户报告的两次现场运行：
  - Proposed+：
    `data/M12_proposed_plus_entropy_grid_20260727_180316_740`；
  - Pure Outward：
    `data/M12_baseline_pure_outward_spiral_20260727_180543_283`。
  `18:02:29` 的 Proposed+ 目录在本轮分析中只作启动旁证，不和完整的
  `18:03:16` 运行混为一谈。
- Proposed+ 的故障链不是单一“阈值太高”：
  - 人工起点为 `X=472.622 mm`，touchdown 为 `X=473.321 mm`；14 个名义纯 Z
    touchdown 动作累计净横向漂移约 `0.699 mm`。旧代码仍以人工 air-start XY
    作为 shortcut/grid 中心，而锥扫 pivot 已是 touchdown pose，导致两个算法阶段
    实际使用不同原点；
  - 锥扫记录了 `280` 次 low-force episode、`880` 个 hold ticks 和
    `22.621 s` hold wall time。多数样本只是 WORLD-Z 法向分量低，而总相对力仍约
    `3–5 N`；旧逻辑仅凭低法向触发，又立即凭总力 `>=2.45 N` 恢复，形成高频 chatter；
    对同一 trace 的 RAMP/SCAN 样本作只读离线重放，新 normal/total 双门只命中
    `13` 个孤立样本，最大连续命中长度为 `1`；
  - 该 trace 的 `training_fingerprint_valid=false`，旧代码仍执行了置信度
    `0.058`、距离 `0.058 mm` 的经典 shortcut；它低于实机 endpoint 分辨率，却被
    当作一次可解释动作/负观测；
  - entropy grid 第 3 个目标已经接近 endpoint，随后连续 5 次下发
    `0.06 mm` Z-only 修正，但实测 Z 几乎不动且伴随 XY 漂移，最终以
    `M12Demo:SurfaceZNoProgress` 停止。最大相对力 `9.019 N`、力矩
    `0.1146 Nm`，均没有触及 `20 N / 0.75 Nm` hard guard。
- Pure Outward 暴露的是同一共享 surface controller：
  - 13 个名义纯 Z touchdown 动作也产生约 `0.695 mm` 横向漂移，旧 spiral
    仍绕人工 air-start 而不是实测 touchdown；
  - 到 `PURE_OUTWARD_0014` 时，一个高层 spiral waypoint 消耗了 60 个动作，
    大量是 `0.06/0.08 mm` Z 修正与 sub-resolution progress；用户在
    `PURE_OUTWARD_0016` 过程中人工停止，因此目录没有 `summary.json`，也没有
    把人工停止误写成算法 failure。
- 四个 full-loop wrapper 不复制控制器；修复全部落在
  `execute_m12_contact_engagement_policy.m` 的共享路径：
  - 保存 `manualStartXY` 只作审计；touchdown 逐段计算最大与累计横向运动，
    最大偏移 `>0.20 mm` 时以 `M12Demo:TouchdownLateralDrift` 在任何搜索前
    fail closed。通过后统一把 measured touchdown XY 绑定为 Pure spiral、
    Proposed shortcut/grid/local spiral 和 Direct reset/ray 的 search anchor；
  - low-force hold 改为同时满足低法向和低总力。3.5 N 目标下名义门为
    `normal<=0.875 N AND total<=1.75 N`；恢复仍可由法向或总接触载荷完成，
    且 outcome 记录 trigger normal/total 与恢复通道；
  - `training_fingerprint_valid=false` 时 Proposed/Proposed+ 以零 shortcut
    abstain 后继续几何残差搜索；低于 `0.12 mm` endpoint 分辨率的 shortcut
    同样归零。Improved Direct 还要求经典方向置信度 `>=0.15`，否则禁止完整
    4.2 mm ray；
  - surface endpoint tolerance 改为 `0.12 mm`。capture 检查仍优先；没有
    capture cue 但终点存在新鲜 `2–6 N` 法向接触且低于
    `10 N / 0.30 Nm` soft envelope 时，直接形成负观测，不再追逐精确 3.5 N；
  - 每个 surface target 的动作预算改为
    `ceil((初始XY距离-0.12)/0.15)+8`，并保留全局 120 上限；
    XY/Z stagnation 从 5/5 收紧为 3/3，force settle 从 12 收紧为 3；
    Z-only 修正累计横向漂移 `>0.15 mm` 即终止。不同轴的无动作不再清零另一个轴
    已累计的 no-progress evidence；
  - shortcut、grid posterior、local spiral center 都使用实测 endpoint，不再把
    planned endpoint 偷换成已实现动作；
  - Pure/Proposed-local 共用 pitch 从 `0.40` 调到
    `0.50 mm/revolution`，实际 Archimedean 几何 waypoint 仍为
    `<=0.15 mm`。离线复算 Pure 4.2 mm 半径为 924 段，Proposed-local
    2.0 mm 半径为 210 段；没有用 0.30 mm 长弦换取速度；
  - `makePlan`/`plan.csv`/manifest workflow 改为 mode-specific。Pure 不再声明
    cone/model/Bayesian hypotheses，Proposed-local、Direct x3 与 Proposed+
    分别只列自己的真实路径。
- 明确保留的边界：
  - surface/touchdown 物理速度仍为 `1.0 mm/s`，insertion 仍为
    `0.60 mm/s`；
  - hard guard 仍为 `20 N / 0.75 Nm`；
  - capture 仍要求 `0.25 mm` 总下沉、`0.04 mm` recent drop 和
    `0.10 mm` extra confirmation；
  - insertion 仍使用 `0.10 mm` 分段及原 depth/bottom-event verifier；
  - shadow model 仍无运动权限。
- 可视化继续采用 deferred/no-blocking render，画面延迟不进入控制状态机，也不能
  作为科学指纹。科学证据是带时间戳的 conical/wrench/command/state CSV 与 manifest。
- 无运动验证：
  - 三个 shared-controller 静态/几何测试模块共 `47` 项通过；
  - 其中离线几何测试确认 `0.50/0.15` 配置下 Pure/Local 分别为
    `924/210` 段，终点半径精确为 `4.2/2.0 mm`；
  - 全量 discovery 仍有一个既有环境错误：
    `/usr/bin/python3` 缺少 `torch`，所以
    `test_model_runtime_authority.py` 无法导入；这与本轮 MATLAB 控制修改无关，
    也与实机中 shadow inference 的相同非阻塞缺包现象一致；
  - 当前 shell 没有 MATLAB/Octave executable，未执行 `checkcode`；
  - 本轮没有启动 MATLAB、ROS、Bota 或 KST，没有发送任何机器人运动命令。
- 本轮关键 SHA-256：
  - shared core：
    `4fc124db9dac06b8cb6878a55a780236c208485a85f3db9285a4b96e91abc55b`
  - `tests/test_m12_real_baselines_static.py`：
    `b8d787238737f945eac03a3660b65c0be0a564d31cefeac5c891dd5349bef820`
  - `tests/test_m12_conical_low_force_guard_static.py`：
    `179bc7aea684bde0c3a35d1da47f38a82d5f40516d268642f27dd3d3b03b7695`
  - `tests/test_m12_sampling_campaign_static.py`：
    `893b4e8a11102d9d72aa9e6dfe32e7f0bbd35bc56f77e50fd8efbb9251b40608`

## 2026-07-27 — 20:35/20:36 touchdown 提前停止取证与有界放宽

- 只读取证运行：
  - `data/M12_proposed_plus_entropy_grid_20260727_203536_032`；
  - `data/M12_proposed_plus_entropy_grid_20260727_203646_680`。
- 两次都不是力、力矩、Bota stale 或 20 mm touchdown 行程门：
  - 均只执行 4 个 `-0.10 mm` touchdown 命令；
  - 20:35 实测下降 `0.420070 mm`、XY 漂移 `0.209870 mm`；
  - 20:36 实测下降 `0.259158 mm`、XY 漂移 `0.202854 mm`；
  - 两次分别因 `0.210/0.203 mm > 0.200 mm` 触发
    `M12Demo:TouchdownLateralDrift`；
  - 最大相对力仅 `0.049638/0.062782 N`，最大相对力矩仅
    `0.000898/0.001103 Nm`，说明停止时尚未接触。
- 20:36 停止点仍比 18:03 历史首次接触 Z 高约 `0.452 mm`，比连续 2 N
  确认接触 Z 高约 `0.502 mm`。18:03 成功触地的 touchdown 横漂为
  `0.699 mm`；其他同链路成功触地记录达到约 `0.912 mm`。因此新加的
  `0.20 mm` air-start 最大偏移门与该 KST 链路已知的逐段小耦合不兼容。
- 截图中的约 25 mm 黑线是 `peg_axis_length_mm` 姿态 quiver，不是下降轨迹；
  真正轨迹只有约 0.26 mm，所以在该坐标尺度下几乎不可见。右侧约
  `±0.05 N / ±0.001 Nm` 曲线则与 CSV 一致。
- 四个 full-loop 的共享修复：
  - `0.20 mm` 改为非阻断预警；
  - 相对 air-start 的最大 touchdown 横漂只有 `>1.00 mm` 才停止；
  - 该上限覆盖历史最大 `0.912 mm`，同时保留每步 `0.30 mm` cross-track、
    `0.10 mm` touchdown 微段和 `20 mm` 最大下探；
  - manifest 使用无陈旧数值的 bounded-policy 名称，plan 从活动 cfg 写入
    `0.20/1.00 mm`；
  - 异常路径从已落盘 command/wrench/state ledger 恢复
    `motion_commanded`、完成命令数、路径、峰值与 touchdown 漂移，避免 MATLAB
    by-value inner throw 把已经执行的 4 条命令误写成零运动；顺序固定为先请求
    KST end，再关闭/flush realtime writers，最后才执行 table-based 审计恢复；
  - dashboard 的动态标题和图例都明确注明黑色 25 mm 段是 axis marker，
    不是运动轨迹。
- 没有修改：
  - `2 N × 3 fresh samples` touchdown 接触门；
  - `20 N / 0.75 Nm` hard guard；
  - touchdown `1.0 mm/s`、surface `1.0 mm/s`、insertion `0.60 mm/s`；
  - capture、surface soft envelope、insertion 与 bottom-event 判据。
  提高接触力门只会更晚识别接触，因此没有采用“所有阈值一起抬高”。
- 无运动验证：三个共享控制器静态/几何模块共 `48` 项通过；全量 discovery
  另外仍只有 `/usr/bin/python3` 缺少 `torch` 的既有 runtime-authority
  import error；只读 block-stack 检查余量为 `0`。当前 shell 没有
  MATLAB/Octave，未执行 MATLAB 运行时解析；本轮没有启动 MATLAB、ROS、
  Bota 或 KST，也没有发送机器人命令。
- 本轮关键 SHA-256：
  - shared core：
    `38dbce287079775cf2054904a855af4d32d98fadc60e705ce1f475af7cc0b046`
  - `tests/test_m12_real_baselines_static.py`：
    `a24d4702ac5853bc4abad317fee799ce62f73f4fceb6f85dc218d3a5ed378601`
  - `tests/test_m12_conical_low_force_guard_static.py`：
    `179bc7aea684bde0c3a35d1da47f38a82d5f40516d268642f27dd3d3b03b7695`
  - `tests/test_m12_sampling_campaign_static.py`：
    `893b4e8a11102d9d72aa9e6dfe32e7f0bbd35bc56f77e50fd8efbb9251b40608`

## 2026-07-27 — 23:37 M12 采样首点 touchdown tracking 停止修复

- 只读取证 run：
  `data/M12_conical_fingerprints_26pt_20260727_233724_277`。
- 精确停止原因：
  - 首点 `SAMPLE_01_TOUCHDOWN_027` 已收到 KST ACK；
  - 第一次 endpoint observation 的 3-D tracking error 为
    `0.1128 mm`，超过原采样门 `0.0700 mm`，因此触发
    `M12Demo:TrackingError`；
  - 前 26 个 `0.10 mm` 微步全部完成，point-air 后累计实测下降
    `2.5963 mm`，当时仍剩约 `4.4 mm` touchdown 预算；
  - 最大相对力/力矩仅 `0.120604 N / 0.001986 Nm`，远低于
    `2 N` 接触门和 `20 N / 0.75 Nm` hard guard，所以不是力停，
    也尚未进入 contact-settle 或 cone。
- 新的采样专用两级 tracking 策略：
  - `≤0.08 mm` 立即接受；
  - `0.08–0.15 mm` 时不补发运动，间隔 `0.08 s` 最多重读两次
    KST endpoint；
  - 只有最佳 endpoint error `≤0.15 mm` 且实测向下进展
    `≥0.02 mm` 才从实测 pose 继续；
  - `>0.15 mm`、无向下进展、KST 未 ACK、非有限状态或其他
    workspace/joint/hard guard 仍终止；
  - recheck 区间经静态检查不含 `movePTP`/`sendEEf`，不会重复下压。
- 同 run 暴露的 UI 初始化错误
  `MATLAB:nonLogicalConditional` 来自对空 `figure=[]` 的
  `isgraphics` 结果直接参与 `&&`；现已先检查 scalar handle，再调用
  `isgraphics`。
- abort summary 现在从已写入的 sampling checkpoint 回填 attempts、
  quarantine、return、plan SHA 和数据权威，避免
  `sampling_campaign.json=1/1` 而 `summary.json=0/0`。
- sampling 的 `DATA_DERIVED_CONTACT_POLICY` 现在记录真实的 `3.5 N`
  target，不再误打印 full-loop 的 `4.0 N`。
- 无运动验证：
  - sampling 静态测试 `24/24`；
  - sampling、low-force 与四个 full-loop 共享回归 `67/67`；
  - model-runtime 隔离测试 `9/9`；
  - printed-CAD plan `--check` 通过，闭环路线 `122.604 mm`；
  - 当前 shell 无 MATLAB/Octave executable；本轮未启动 MATLAB、ROS、
    Bota 或 KST，也未发送机器人运动命令。
- 本轮关键 SHA-256：
  - shared core：
    `706c48c88f2a37a345a012cd4747a39f71c72ac2b68d201d634099bcb86af5ac`
  - sampling top view：
    `685797c2251f7487c4af488c70a4fa76728a9346e5805863228ca962d6b5db7c`
  - sampling static test：
    `b59620988c0131de6145ddb6b99e56bd9dddd848c1d03009263d774583644a0d`
  - README：
    `1a21838cde415d5e9c53845682246de35a5b45d7e433b3da5250416e06589b8e`

## 2026-07-28 — 00:04 M12 点 2/3/6/9 跳扫与点 10 边界误停修复

- 只读取证 run：
  `data/M12_conical_fingerprints_26pt_20260727_234813_623`。
- 逐点根因：
  - 点 2、3、6、9 均已完成 touchdown，最大 XY 落点误差分别为
    `0.134/0.056/0.063/0.075 mm`，均通过 `0.25 mm` 门；
  - 四点没有锥扫是因为 4 s contact-settle 末端分别稳定在
    `4.340±0.063 / 4.533±0.085 / 4.466±0.077 /
    4.202±0.046 N`，高于原来的 `3.0–4.0 N` 唯一接受带；
  - 原稳力器反复发 `+Z 0.025–0.033 mm` 卸载并混入 XY 纠偏，
    但 KST 实测几乎没有同方向 Z 进展；命令总长又常低于
    `0.08 mm tracking + 0.02 mm deadband`，于是被
    `SUB_TOLERANCE_PLUS_DEADBAND_PROGRESS_ACCEPTED` 当成完成；
  - 点 10 在 `SAMPLE_10_TOUCHDOWN_003` 已通过采样专用门：
    best tracking `0.0957 mm < 0.15 mm`、向下进展
    `0.0370 mm > 0.02 mm`；随后通用向量比例门以
    `0.0273 < 0.0275 mm`（仅差约 `0.0002 mm`）再次否决并终止整轮。
    当时尚未接触，异常不是 CAD 落洞、Bota stale 或 hard guard。
- 修改：
  - sampling touchdown 的专用 tracking、向下进展和 XY 落点门成为该动作
    唯一进展判据；通过后不再重复套用通用 3-D vector ratio gate，其他动作
    仍保留通用门；
  - contact-settle 改为纯 Z，不再把 touchdown 后的 XY 标签反复拉回命令点；
  - 轴向稳力步使用 `0.10 mm` 有效命令并要求同方向实测 Z 进展
    `>=0.01 mm`；连续两次未观测到进展后停止空打，继续收集新鲜力数据直到
    4 s 有界判定；
  - `3.5±0.5 N`、`0.5 s / 20` 点、`std<=0.20 N` 仍是优选通过条件；
    若最终连续窗口在 `2.5–4.75 N`、`std<=0.12 N`，则以
    `PASS_OFF_TARGET_STABLE_LOAD` 继续锥扫，并在逐点 JSON/MAT/CSV 中记录
    off-target、调节次数和未观测调节次数，仍只交给离线 QC；
  - continuation 尾窗再次显式要求至少 20 个样本，且每次 settle 调节后和最终
    promotion 前都复验 `0.25 mm` XY 门，短尾窗或最终 XY 超差不能被升级；
  - `20 N / 0.75 Nm` hard guard、Bota freshness、`0.25 mm` 落点门和
    真正的 KST tracking/ACK 未知状态终止规则未放宽。
- 对旧 run 的 contact-settle CSV 按新 continuation 窗口离线重放：
  点 2/3/6/9 均通过；均会进入锥扫而不是被
  `CONTACT_FORCE_NOT_STABLE_TIMEOUT` 跳过。
- 无运动验证：
  - M12 sampling 静态测试 `27/27`，其中 sampling-attempt CSV schema
    自动锁定为 36 列；
  - sampling、low-force 与四个 full-loop 共享回归 `70/70`；
  - printed-CAD 五个同步产物 `--check` 通过，路线仍为 `122.604 mm`；
  - DirectServo 循环禁入检查随静态套件通过，没有新增 drawnow、timer、
    KST query、subscriber 或批量文件写入；
  - 当前 shell 无 MATLAB/Octave executable；本轮未启动 MATLAB、ROS、
    Bota 或 KST，也未发送机器人运动命令；
  - 全量 `test_*.py` 额外尝试仍只有既有的 `/usr/bin/python3`
    缺少 `torch` 导入错误；M12 sampling 入口按设计不加载模型，该错误不影响
    上述 `69/69` 采样与共享控制回归。
- 本轮关键 SHA-256：
  - shared core：
    `62e4267891d270b8c8d87e9c879b443557738e96f9fd4818e16c72d79db9ce6f`
  - sampling static test：
    `4a453171610b406f6e40803595095e37b0d709735daf112cdb68fc3a36138f14`
  - README：
    `016011bdc1703d83c9cc4b6bf5bac8db1edb90ac7de85d35dabcbf7de32089d9`

## 2026-07-28 — 00:40 M12 全点采集优先：撤销振荡稳力门并增加单点重试

- 只读取证 run：
  `data/M12_conical_fingerprints_26pt_20260728_001633_188`。
- 该 run 是操作者在第 8 点接触阶段手动停止的中间记录：
  - 只完整落盘 7 个 attempt；
  - `sampling_campaign.json` 仍为 `RUNNING`，没有终态 `summary.json`；
  - 00:21:51 已记录 KST end 请求，不能把它误报成 26 点 campaign 完成。
- 前 7 点的真实分支：
  - 1--7 点全部确认 touchdown，且实测 XY 都通过 `0.25 mm` 落点门；
  - 只有执行点 4、6 做了锥扫；
  - 1、2、3、5、7 点仅下压后跳点，唯一原因都是
    `CONTACT_FORCE_NOT_STABLE_TIMEOUT`，不是 Bota stale、hard force/torque、
    落点误差或未触地；
  - 跳扫点最终报告分别约为
    `3.582±0.879 / 3.612±0.725 / 3.304±0.911 /
    3.499±1.014 / 3.608±0.921 N`，每点都执行了 6 次轴向调节；
  - 点 4、6 只是偶然落入窄窗口，分别约为
    `3.182±0.026 / 3.780±0.027 N`。
- 根因不是接触本身不稳定，而是旧 pre-cone 控制器制造了振荡：
  - `Z ±0.10 mm` bang-bang 命令在实机上产生约 `0.026–0.057 mm`
    可观测位移，因此不会进入“微步未观测”的抑制分支；
  - 载荷反复在约 `2.4–2.9 N` 和 `4.1–4.6 N` 两档间切换；
  - 旧逻辑随后又要求最终窗口低方差，因而把自己激发出的振荡当成无效接触并跳扫。
- completion-first 修改：
  - 删除 sampling pre-cone 的 PTP/Z 稳力动作及其专用 KST 微步旁路；
  - touchdown 物理资格变为
    `fresh baseline && confirmed contact && XY landing pass && no capture/bore`；
  - 合格接触保持位姿不动，只读 fresh Bota 至少 `0.60 s`，最多
    `1.50 s`；逐样本仍执行 `20 N / 0.75 Nm` hard guard；
  - `3.5±0.5 N、0.5 s / 20 samples、std≤0.20 N` 仅产生
    `PREFERRED_STABLE_LOAD` 质量等级；
  - 其他 fresh、未越 hard guard 的接触产生
    `PASS_COMPLETION_FIRST_CONTACT_OBSERVED /
    NONPREFERRED_LOAD_PENDING_OFFLINE_QC`，但不能跳过该点锥扫；
  - 方向标签失败只 quarantine 数据，不阻止已物理合格点的诊断锥扫；锥扫成功后
    不得把此前 quarantine 覆盖回 clean。
- 全点鲁棒性：
  - 每点最多 3 次 baseline/touchdown acquisition；
  - baseline 暂时噪声超限时原位重采；
  - 一次未触地或 XY 超差时，先 verified clear retract，再回同一目标、重采局部
    baseline 并重下压；
  - capture/bore 不重试；hard force/torque、Bota stale、KST tracking/ACK、
    DirectServo 或机器人状态未知异常不在重试 helper 内捕获，继续终止整轮，
    禁止盲走；
  - 逐点 JSON/MAT/CSV 新增 preload quality、completion-first、观测时长和
    min/max force；CSV schema 从 36 列扩为 41 列。
- 无运动验证：
  - M12 sampling 静态/契约测试 `29/29`；
  - 全部 `test_m12_*.py` 共享回归 `72/72`；
  - printed-CAD 五个同步产物 `--check` 通过，路线仍为 `122.604 mm`；
  - 全量 `test_*.py` 另外仍只有既有的 `/usr/bin/python3` 缺
    `torch` import error；sampling 入口不加载模型，因此与本改动无关；
  - 当前机器没有 MATLAB/Octave executable，未执行 `checkcode`；
  - 本轮没有启动 MATLAB、ROS、Bota 或 KST，没有发送机器人运动命令。
- 本轮关键 SHA-256：
  - shared core：
    `bc37db0bffd5146ef2baff691d1dc0337bd770af1a7037cdc0f882c21532d325`
  - sampling static test：
    `87ac008c6a399c3219bcb155251da7a8073f101dfbb90eb975db839ffcaa0ea4`
  - README：
    `2ee98fd94658f93a55c116c1c513b8c06062eb124df61760e6a5062383d2a9e2`

## 2026-07-28 — 02:25 第 19 点 22.154 N 锥扫峰值：25 N 门限与定相位快速卸载

- 只读取证最新 resume run：
  `data/M12_conical_fingerprints_26pt_resume_20260728_021050_298`。
- 根因：
  - 第 1--18 点锥扫证据完整；第 19 点 touchdown 落点误差约 `0.050 mm`，
    被动接触观测为 `3.508±0.048 N`；
  - SCAN 约 `60.82%`、相位约 `285 deg` 时，总相对力由前一帧
    `4.226 N` 跳到 `22.154 N`，WORLD* 分量约
    `[4.828, 7.594, 20.243] N`，Bota age 约 `3.4 ms`；
  - 旧 `20 N` hard guard 在 `12/9 N` 软卸载分支之前抛错，因此不是
    Bota stale、KST read race 或 DirectServo stop 失败；cleanup stop ACK 已落盘。
- 当前 M12 共享控制修复：
  - Proposed+、Proposed-local、Pure outward spiral、Improved direct 和
    26 点 sampling 共用的力 hard guard 从 `20 N` 调为 `25 N`；
  - 总相对力 `>=18 N` 且 `<25 N` 时，先保留当前锥扫相位并在既有
    40 Hz 流内增加一次有界 `+0.10 mm` 卸载；随后沿用 `12/9 N` 滞环，
    `<=9 N` 连续 4 个 fresh samples 后从原相位继续；
  - 没有增加 drawnow、timer、KST query、subscriber 或流内批量 I/O；
  - 发生过 force relief 但完整扫完的指纹仍保存且可供 routing，
    标成 weak/pending offline QC，`training_fingerprint_valid=false`；
  - 若仍达到 `25 N` 或 `0.75 Nm`，立即停止；UI 状态会明确写出对应
    force/torque guard，而不再只显示泛化的 state unknown。
- qualification 留痕：
  - 保留原 20 N 文件及其历史 SHA-256
    `71614f287be5083897db1403a7efe2d896f1a5f0a3b0227b44c5948a0f2aaf86`；
  - 新增 `test_real_device/bota_contact_qualification_SN001349_25N.json`，
    记录本次操作者明确授权和点 19 实测证据，不改写旧 manifest。
- 轻量无运动检查：
  - MATLAB `checkcode`：`51` 条既有风格/分析提示，parse/syntax `0`；
  - conical targeted static tests `13/13`；
  - sampling DirectServo/UI/resume/data-schema targeted tests `4/4`；
  - 25 N qualification JSON、range evidence hash 和主程序绑定通过；
  - 未启动 ROS、Bota、KST，也未发送机器人运动命令。
- 本轮关键 SHA-256：
  - shared core：
    `1241e25cdd02152fe32171f8cf1119a1e29f4ca58dac0c88cf4a3ee587b9930e`
  - conical static test：
    `f7c5ea6034cb358525846661204ebd4e3f135ec29a328f015153989e7226947e`
  - README：
    `672777a42688f589103846416ea35a7076879633be75aad8795e2d2408f61ef1`
  - 25 N qualification：
    `fa14bcb35fa50f273f5ac819157341fddd8747a39628e9d528b7271fa036e7be`

## 2026-07-28 — 03:21 M12 26 点锥形指纹首次全量采集完成（里程碑）

### 🎉 现场里程碑

- 操作者现场确认并提交最终俯视界面：
  `COMPLETE_RETURNED_TO_INITIAL_AIR_POSE_KST_END_SENT`，
  `cones: 26/26`、`secured: 26/26`、`target: none`。
- 这是现存 M12 26 点 campaign 记录中第一次同时达到
  **26 点全部锥扫、26 点全部落盘、26 个动作生命周期闭合、26 次原始回撤验证，
  并返回初始空气位后正常结束 KST**。此前时间戳目录保持原样，未被覆盖或改写。

### 机器证据与续采谱系

- 最终聚合证据目录：
  `data/M12_conical_fingerprints_26pt_resume_20260728_031926_032/`。
- `sampling_campaign.json` 的权威终态为：
  - `status = COMPLETE_RETURNED_TO_INITIAL_AIR_POSE_KST_END_SENT`；
  - `attempts_completed / attempt_records_persisted / cones_completed /
    points_committed = 26 / 26 / 26 / 26`；
  - `motion_lifecycles_closed / original_retracts_verified = 26 / 26`；
  - `pending_execution_indices = []`；
  - `returned_to_initial_air_pose = true`，`error = ""`。
- campaign 从 `2026-07-28T03:00:40.651+08:00` 开始，经两次 checkpoint resume，
  在 `2026-07-28T03:21:43.595+08:00` 完成：
  - generation 0：
    `M12_conical_fingerprints_26pt_20260728_030022_482`，完成点 1--15；
  - generation 1：
    `M12_conical_fingerprints_26pt_resume_20260728_031146_935`，续完点 16--22；
  - generation 2：
    `M12_conical_fingerprints_26pt_resume_20260728_031926_032`，续完点 23--26。
  两级 `resume_parent_checkpoint_sha256` 均与父 checkpoint 实算一致。
- `events.csv` 在 `03:21:43.575` 明确记录
  `SAMPLING_ALL_CONES_AND_MOTION_LIFECYCLES_COMPLETE`，随后记录
  `KST_END_REQUEST` 与 `KST_END_WRITTEN`；
  `kst_end_state.json` 确认通过 `KST_OBJECT_NET_TURNOFFSERVER` 发送 end。
- 逐点核验结果：
  - 26/26 `cone_completed`、`motion_lifecycle_closed`、
    `safe_retract_completed`、`immutable_evidence_committed`；
  - 26 个点的 attempt JSON/MAT、空气 baseline、contact-settle 与 cone CSV
    共 130/130 个必需文件存在且非空；
  - 104/104 个逐点有效载荷声明哈希匹配；
  - 最终 manifest 的 9/9 个可解析源码、配置、资格与 CAD 路径哈希匹配。

### 数据分区与科学边界

- 预注册分区完整保持为：16 个 clean candidate、4 个 holdout、
  4 个 planned quarantine diagnostic、2 个 repeatability sentinel。
- 接触几何覆盖为：8 个 supported search、7 个 double overhang、
  7 个 outer partial support，以及两类边界点各 2 个。
- 4 个 `PLANNED_QUARANTINE_DIAGNOSTIC_ONLY` 是预注册诊断分区，不是漏采或运行失败；
  16 个 clean candidate 仍为 `PENDING_OFFLINE_QC`，本次完成不自动授予训练资格，
  holdout/sentinel/quarantine 的治理边界不变，也没有在线更新模型。
- 这是 **M12 commissioning 数据采集 campaign 首次完整完成**，不是 full-loop
  capture/insertion 成功声明。`summary.json` 中通用的 `success=false` 沿用
  full-loop 语义；采样完成以其中的 `task_completed=true`、三个 sampling 计数为 26
  以及 campaign 权威终态为准。

### 工程意义

- 这条完整真机链证明总体方向正确：completion-first 让弱但完整的接触证据继续完成
  物理动作，checkpoint resume 保存已完成锥扫，适应性卸载处理可恢复载荷事件，
  而 endpoint 量化、短暂失联和 ACK 尾部抖动不再轻易推翻已经可验证的动作。
- 关键不是“删掉所有门”，而是把门分层：未知状态、陈旧/非有限信号、硬力/力矩和
  workspace 越界仍终止；有界 tracking 抖动、弱指纹和可恢复接触事件则记录、适应并
  继续。第一次 26/26 是这套鲁棒性与自适应节奏的直接真机证据。

### 里程碑证据 SHA-256

- `sampling_campaign.json`：
  `f0dd2918c9c6aa93f7fe0df03c91151de6270590d1c9898517d9680f54390fe1`
- `summary.json`：
  `7fe4acf51dfccaaa6aa6db042b38fee02606669744493bcd1b41fe1b2e36ce3d`
- `manifest.json`：
  `1e865551fadf10af9fea95bb7e3578fb11ae9098bbe1af23af7bb44a52dfd876`
- `conical_directservo_trace.csv`：
  `e3dee30230663bca5335ffcdd5f424ef8818b4a80be2b87dd9c7ca7fb066b3e2`

## 2026-07-28 — 16:24 M8 首次运行 26 点全量完成，三尺寸采样链闭合（里程碑）

### 🎉 M8 现场与机器终态

- 操作者提交的最终俯视界面明确显示：
  `COMPLETE_RETURNED_TO_INITIAL_AIR_POSE_KST_END_SENT`，
  `cones: 26/26`、`secured: 26/26`、`target: none`。
- 对应唯一 fresh-run 证据目录：
  `data/M8_conical_fingerprints_26pt_20260728_161249_458/`。
- `sampling_campaign.json` 的权威终态为：
  - `status = COMPLETE_RETURNED_TO_INITIAL_AIR_POSE_KST_END_SENT`；
  - `attempts_completed / attempt_records_persisted / cones_completed /
    points_committed = 26 / 26 / 26 / 26`；
  - `motion_lifecycles_closed / original_retracts_verified = 26 / 26`；
  - `pending_execution_indices = []`；
  - `returned_to_initial_air_pose = true`，`error = ""`。
- campaign 从 `2026-07-28T16:13:07.467+08:00` 到
  `2026-07-28T16:24:17.396+08:00`。26 个点全部
  `point_acquisition_attempts = 1`、`point_touchdown_attempts = 1`，
  `inherited_from_parent_run = false`：这是一次 fresh invocation 从第 1 点直接
  完成到第 26 点，不是 resume 拼接，也没有点级重试。
- `events.csv` 在 `16:24:17.373` 记录
  `SAMPLING_ALL_CONES_AND_MOTION_LIFECYCLES_COMPLETE`，随后记录
  `KST_END_REQUEST` 和 `KST_END_WRITTEN`；`kst_end_state.json` 确认
  `KST_OBJECT_NET_TURNOFFSERVER`。

### 完整性、时序与数据治理

- 逐点核验为 26/26 `cone_completed`、`motion_lifecycle_closed`、
  `safe_retract_completed`、`immutable_evidence_committed`。
- 26 个 attempt 的 JSON/MAT、空气 baseline、contact-settle 和 cone CSV
  共 130/130 个必需文件存在且非空；104/104 个逐点声明哈希实算匹配；
  manifest 中 12/12 个非空源码、入口、配置、资格和 CAD 路径哈希匹配。
- 26/26 点位空气 baseline 有效、26/26 touchdown 落点门通过；
  最大实测落点误差 `0.1053 mm`，最大锥扫后 tip 误差 `0.0764 mm`。
- 所有锥扫相位覆盖率均为 `0.995833`；最大 command gap `0.0380 s`，
  最大 tick lateness `0.0121 s`，没有 force-relief episode。
- 本次 M8 接触较灵敏：共记录 199 个可恢复 low-force episode，
  26/26 均以 `QUICK_RECONTACT_RESUMED_TAGGED` 完成。它们全部保留为
  `weak_but_complete_fingerprint`，26/26 direction/routing fingerprint 有效，
  没有因软事件跳过物理动作。
- 预注册分区原样保持为 16 clean candidate、4 holdout、4 planned
  quarantine diagnostic、2 sentinel；16 个 clean 仍是
  `PENDING_OFFLINE_QC`，自动训练准入为 0。完成采集不等于已经完成离线 QC，
  也不改变 holdout/quarantine/sentinel 的治理边界。

### M12 / M10 / M8 三尺寸里程碑

- M12 最终 lineage：
  `data/M12_conical_fingerprints_26pt_resume_20260728_031926_032/`；
  经过 checkpoint resume 后首次达到完整 26/26。
- M10 fresh run：
  `data/M10_conical_fingerprints_26pt_20260728_152808_877/`；
  26 点均一次 acquisition、一次 touchdown，直接 26/26。
- M8 fresh run：
  `data/M8_conical_fingerprints_26pt_20260728_161249_458/`；
  26 点均一次 acquisition、一次 touchdown，直接 26/26。
- 三个最终 campaign 均为
  `COMPLETE_RETURNED_TO_INITIAL_AIR_POSE_KST_END_SENT`，且六项闭环计数
  `attempts / persisted / cones / committed / lifecycles / retracts`
  全部为 `26 / 26 / 26 / 26 / 26 / 26`。
- 这证明“跑顺一个共享物理内核，只替换实际打印 CAD、点位、尺寸身份和薄 UI
  facade，再经轻量检查立即上真机”的迁移路线成立：M12 的调试经验没有被复制成
  三套漂移控制器，而是直接转化为 M10、M8 的一次性全量完成。

### M8 里程碑证据 SHA-256

- `sampling_campaign.json`：
  `17a5038ed897e67fedf267a7b92b98d8207576a8a81293bff4626a5bafab3205`
- `summary.json`：
  `6495385a61c64578e50ebe8c268c6a95f5a8a20e2768f4e1bcd52f1667cc678e`
- `manifest.json`：
  `91447eb37bb05a8204a8d2fe26571d1c23ec97f944e9733cc32d2f24a80d8467`
- `conical_directservo_trace.csv`：
  `e9151015246ba0f254f3868152f2813018690e828bb2386670db783800557476`
- `events.csv`：
  `40ff6cc6a4cd5984be77ce62de1afa4ebb94dca33a43d2dd35aa518a1e12bdeb`
- `kst_end_state.json`：
  `e19a757a4f502be2e1026bcae4b7ec380602d27b6302ba4c4cb00bd867d458ac`

## 2026-07-28 — 16:50 真机 skill 迁入 real_device；M12 第二轮 fresh campaign 就绪

### Skill 搬迁与身份恢复

- 按操作者要求，将唯一 skill 包从
  `/home/ubuntu20/robot_ws/skills/kuka-iiwa-ros-safety-audit/`
  完整迁入
  `/home/ubuntu20/robot_ws/group_meeting/real_device/skills/kuka-iiwa-ros-safety-audit/`；
  旧 `/home/ubuntu20/robot_ws/skills` 路径已为空并移除，没有同名覆盖。
- 源包当时缺失 skill 必需的 `SKILL.md`。新位置补回精简、可触发的入口，并把
  已验证的真机节奏写成默认契约：
  `共享内核 + 单一改动 -> 轻量无运动检查 -> 立即真机运行 -> 原始证据迭代`。
  对可恢复接触/跟踪/弱信号事件优先自适应继续并保存已完成动作；仅在状态未知、
  信号陈旧/非有限、命令所有权丢失或有限硬边界触发时停止。
- 原有 references、4 份 machine-readable evidence、只读审计脚本全部随包迁移；
  `agents/openai.yaml` 同步为新的 real-robot workflow 身份。除该 YAML 的界面文案外，
  原有 15 个有效资源文件内容哈希保持不变。
- 当前 README、data 约定、控制栈审计与成功日志的操作链接均已切换到新路径。
  `docs/setup/SETUP_LOG.md` 中 2026-07-18 的旧相对路径仅作为历史事实保留。
- `skill-creator quick_validate.py` 返回 `Skill is valid!`；10 个 SKILL 相对链接、
  `read_only_audit.py --help` 与旧路径清理检查全部通过。
- 17 个有效 skill 文件的排序 SHA-256 清单聚合哈希：
  `5c59c86134ef18829283ef293ad68ba3f6b09728bc145b7653b6946f7a8c0d87`。

### M12 第二轮 fresh campaign 轻量起跑检查

- 操作者已确认换回 M12 peg，并把 TCP 放在本轮 M12 初始孔心上方。
- 入口
  `run_m12_sampling_conical_fingerprints_26pt.m`
  只选择 `DATASET_M12_CONICAL_FINGERPRINTS_26PT`；调用该专用入口即授权本轮，
  不要求第二次命令窗口口令。
- M12 identity 链保持一致：
  `m12_run_local_conical_fingerprints_26pt_v2` /
  `m12_printed_cad_conical_sampling_v2` /
  `m12_sampling_topview_dashboard`，不会串入 M10/M8 profile。
- printed-CAD 生成器 `--check` 通过：26 点，闭环路线 `122.604 mm`；
  分区保持 `16 clean / 4 holdout / 4 quarantine / 2 sentinel`。
- 入口 mode、生成计划、免二次口令和 DirectServo UI 隔离共 4 个聚焦静态测试
  `4/4 PASS`；独立复核再加 schema/coverage 后为 `5/5 PASS`。
- 整个准备过程没有连接 ROS、KST 或 Bota，没有发送机器人运动。本轮真正启动时
  仍由专用入口采集 fresh KST start、fresh Bota baseline，并创建新的非覆盖 M12
  campaign 目录。
- 起跑身份 SHA-256：
  - M12 entry：
    `eb86e2c9e236d6aa3428ede765aa6bc2987c129c7bbf3490f3f16ba8e6ae4bd6`
  - M12 plan：
    `2577ccc8bc95bfd0b5c93ece19903c6bc3b54d20ba1dba3638a9e90740b762ac`
  - M12 dashboard：
    `93c21f8254f6590d71fa2d1b43c181c75d3e01d9b1ede5eab83078a0144563b9`
  - shared executor：
    `67328c309fb8b9c3772cbf81692defa50a7c93425e57b1a052cbe0cde61dd0e7`

## 2026-07-28 — 17:38 误用 M8 程序采 M12 实物：整轮隔离并重新准备 M12

### 现场纠错事实

- 操作者在 26 点顺利完成后确认：实物为 **M12 peg + M12 孔**，但本轮误运行了
  M8 入口。
- 对应完整 run 已唯一定位为：
  `data/M8_conical_fingerprints_26pt_20260728_172108_490/`，
  从 `17:21:08` 运行到 `17:32:22`。
- 机器证据三重确认其程序身份为：
  `size=M8`、`mode=dataset_m8_conical_fingerprints_26pt`、
  `execution_mode=DATASET_M8_CONICAL_FINGERPRINTS_26PT`；
  public entrypoint 为 `run_m8_sampling_conical_fingerprints_26pt`。
- 动作层本身完整：26/26 attempts、cones、committed points、retracts 和 lifecycle
  均完成，`pending=[]` 且已返回本轮初始空中位。这说明共享采样控制仍运行顺畅，
  但不改变物理身份错配事实。
- 该 run 的 M8 CAD/点位/partition/sample ID/方向标签契约与实际 M12 tool/socket
  不一致。因此它既不能作为标准 M8 数据，也不能事后改名冒充 M12 数据，只保留为
  physical-program mismatch 诊断证据。
- 前一目录 `M8_conical_fingerprints_26pt_20260728_172009_844` 是 0/26 的 KST ACK
  中止，无完整采样污染，不属于本次完成 run。

### 非破坏性整轮 quarantine

- 原始 manifest、summary、campaign、attempt JSON/MAT 与所有 CSV 均未改写或改名。
- 在完成 run 根目录追加：
  - `RUN_QUARANTINE.json`：整轮
    `PHYSICAL_TOOLING_PROGRAM_IDENTITY_MISMATCH`，并设置
    `effective_partition=quarantine`、
    `data_quality_status=QUARANTINED_NEVER_TRAIN`、
    `offline_training_candidate=false`、
    `automatic_training_admission=false` 和 `training_eligible=false`；
  - `DO_NOT_TRAIN.flag`：供人工和未来数据装载器快速拒绝整轮。
- campaign-level disposition 明确优先于 16 个 clean attempt 原先的
  `offline_training_candidate=true`；该 run 禁止进入训练、归一化、验证、
  holdout evaluation 或 benchmark reporting。
- 原始权威文件 SHA-256 在添加 sidecar 前后保持一致：
  - manifest：
    `4c3cd96067d1a868fad50bde9e41ece4dc40de488f67c67a7a335abeb33b0b76`
  - summary：
    `d96f5cfb088edffb070e62545e261c34056804dc890af07d4c07755712b7c7d4`
  - sampling campaign：
    `a12f7b31da6c91ed0492c3b2c0de82d2035350efc603eccb5f2e6e3f2de8d1c0`
  - sampling attempts：
    `da43e538b28e8c047364a22c91b447bd55164134a82d30a0a3d1652745db476d`
- quarantine sidecar SHA-256：
  - `RUN_QUARANTINE.json`：
    `57afa8cd2b567e0d033cb44a3fd16e34aff33e28bcb830dd48ea49091685929f`
  - `DO_NOT_TRAIN.flag`：
    `a20713fe69d69ac37dd1d97a454689189ef0ac8399e2e0dad5ce074f9a6ace35`

### 正确 M12 重采就绪

- 这次必须新建 **fresh M12 campaign**，不能 resume 上述 M8 run。
- `generate_m12_printed_cad_sampling_plan.py --check` 再次通过，
  路线为 `122.604 mm`；M12 wrapper mode 聚焦测试通过。
- 正确唯一入口：
  `run_m12_sampling_conical_fingerprints_26pt`。
- 本次纠错、隔离和检查没有连接 ROS/KST/Bota，也没有发送机器人运动。

## 2026-07-28 — 17:52 M12 首段未动作：修复运动前 KST ACK 误超时

### 现场根因

- 失败 run：
  `data/M12_conical_fingerprints_26pt_20260728_174327_635/`。
- dashboard、ROS、Bota 和 KST 连接均已初始化；故障不在俯视图。
- 首个计划动作是 `SAMPLING_INITIAL_LIFT`：
  `[0, 0, +1.5] mm @ 8 mm/s`。
- KST 仅写出了不会触发运动的速度参数 `jRelVel_8_`，随后
  `RELATIVE_LINEAR_VELOCITY` 在原来的 `2.00 s / 4 reads` 内均为空回复。
  Cartesian target 和 `doPTPinCSRelBase` 均尚未写出，因此机器人没有实际运动；
  本轮为 `0/26`，不应 resume。
- 早先 M8 首次启动也在相邻的运动前 target ACK 阶段发生过同类瞬态空回复，
  随后的 fresh run 完成 `26/26`。这证明问题属于 KST 首段回复延迟，而非 M12
  CAD、点位、Bota 接触门或 UI 绘图。

### 最小修复

- `movePTPLineEefRelBase.m`：
  只把不会触发运动的 `RELATIVE_LINEAR_VELOCITY` ACK 预算从 `2.0 s`
  调整为 `8.0 s`，与 tcpclient 的有界 transport timeout 一致。
- `sendEEfPositions.m`：
  只把不会触发运动的 `CARTESIAN_RELATIVE_TARGET` ACK 预算从 `2.0 s`
  调整为 `8.0 s`。
- `doPTPinCSRelBase` 发出后的 acceptance/completion 语义未放宽；KST 使用无序号
  `done`，因此任何阶段都没有增加盲目重发，避免运动重复执行或回复串位。
- 聚焦静态测试通过：
  `test_tcpclient_fallback_has_bounded_native_line_io` 为 `1/1 PASS`；
  post-motion no-replay 与 M12 wrapper mode 为 `2/2 PASS`。
- 本轮修复期间没有连接机器人，也没有发送运动。

### 修复后身份 SHA-256

- `movePTPLineEefRelBase.m`：
  `7de702acd26f7bf17b3f24fce409f3ac974345de11b3b9c86f15ceee25fbf321`
- `sendEEfPositions.m`：
  `d27e2897d55b94ebf67772a9b95591a4baf3594ed01357aa92064d12b938eff2`
- `test_m12_real_baselines_static.py`：
  `116f758eded9f395e76a7640a705569b7bf7e614fffb734b6faec9219bc4feba`

## 2026-07-28 — 18:41 三尺寸首轮离线 QC、方向可观测性与 CAD 距离边界

### 锁定数据与采集完整性

- 只允许三个 fresh、一次性完成的 campaign：
  - M12：`M12_conical_fingerprints_26pt_20260728_175502_318`
  - M10：`M10_conical_fingerprints_26pt_20260728_152808_877`
  - M8：`M8_conical_fingerprints_26pt_20260728_161249_458`
- 错误的 M8-program/M12-physical run
  `M8_conical_fingerprints_26pt_20260728_172108_490`
  必须同时带有 `DO_NOT_TRAIN.flag` 和 `RUN_QUARANTINE.json`，否则分析拒绝启动。
- 三套共 78 点均完成；312 个声明的 MAT/CSV SHA-256 全部复核一致。
  每点 30 条 fresh air baseline、240 条 SCAN、循环相位最大缺口约 `1.5 deg`；
  sample ID、所有逐点路径、sampling-plan hash、partition/state/subtype 和 fresh
  lineage 均逐项对账。
- 分区严格为 `48 clean_candidate / 12 holdout / 12 quarantine / 6 sentinel`。
  这里的 `PASS` 只表示采集与文件完整性允许做探索性可观测性分析，不表示模型晋级：
  78 条 `cone_training_fingerprint_valid` 和流内 pose-tracking qualification 均为 false，
  因为 DirectServo 内没有实测 KST pose；`actual_z` 只能解释为 command-propagated
  proxy，不能冒充实测锥扫位姿。

### 仿真到真机：方向存在，但冻结模型不能零样本直搬

- 冻结仿真 v7 CNN 使用完整的 `Fz/Tx/Ty/tip_z + 16-D context` 合同进行离线 replay。
  主结果按三个真机 campaign 的 `3.5 N` 下压力合同：
  - 66 个非边界点方向中位 / p90：`99.76 / 157.70 deg`
  - `<=30 deg`：`0.182`
  - 状态准确率：`0.364`
  - M12/M10/M8 方向中位误差：`117.86 / 80.49 / 98.16 deg`
- 使用旧运行时默认 `3.0 N` 的敏感性结果仍失败：
  `93.08 / 161.05 deg`，`<=30 deg = 0.212`。
- 只在 48 clean 上拟合固定镜像/旋转也不能解释域差；12 holdout 的 global/per-size
  sign-offset 中位误差仍为 `82.93 / 75.78 deg`。因此问题不是一个坐标系正负号，
  而是 Bota 测量原点、载荷/重力公共模态、真实柔顺摩擦和通道特征的联合域差。

### 真机方向可观测性

- 仅用 48 clean 拟合、按真实 CAD contact state 条件化的 Fx/Fy 环形模板，在首次
  development holdout reveal 上：
  - top-1 中位 / p90：`19.56 / 42.32 deg`
  - top-1 `<=30 deg`：`10/12`
  - best-of-top-2 oracle posterior coverage：`12/12 <=30 deg`
- top-2 只证明多峰 posterior 覆盖真方向；线上没有 oracle 选择较好峰，不能把
  `12/12` 写成可执行方向准确率。该模板也按 size/state 分开，是可观测性上界，
  不是统一三尺寸部署模型。
- 对 Fz/Tx/Ty 做 leave-one-clean-sample-out 公共模态消除，再按方向做 C6 对齐：
  M12/M10/M8 pairwise correlation 为 `0.387 / 0.218 / 0.063`；256 次同状态标签置乱
  单侧 p 值为 `0.0039 / 0.0039 / 0.0389`。这证明三种尺寸的真机信号中存在方向相关
  残差，但不等于已完成跨尺寸识别。
- 12 条 holdout 已记录为
  `REVEALED_ONCE_V1_DO_NOT_RETUNE_ON_THESE_RESULTS`；分析源码、clean 输入和模板参数
  已绑定一次揭示签名。以后源码/输入/模型变化时默认拒绝重新看这 12 条；下一版
  adapter 冻结后必须采新的物理 audit set。

### CAD 后验距离边界

- 当前打印 CAD 的 peg/socket flat-normal yaw 差均为 `0 deg`；同 yaw 正六边形
  Minkowski erosion 给出的可容纳孔心区域 AF：
  M12/M10/M8 = `5.00 / 4.14 / 3.30 mm`。
- 使用实测 offset + 真实向心方向得到的首次几何入口距离 min/median/max：
  - M12：`3.001 / 6.044 / 9.843 mm`
  - M10：`3.284 / 5.323 / 8.018 mm`
  - M8：`2.645 / 4.077 / 6.024 mm`
- 这些数值是 post-hoc CAD mouth-containment oracle envelope，不是已经部署的后验
  捷径，更不是倒角捕获距离。正式捷径必须由视觉中心先验 × calibrated state
  likelihood × 多峰方向 posterior 形成中心 posterior，再取带风险约束的距离分位数；
  physical chamfer capture margin 需用短 full-loop 真机试验单独标定。

### 产物、测试与下一步

- 主分析：
  `tools/analyze_multisize_real_fingerprints.py`
- 冻结仿真零样本 replay：
  `model_runtime/evaluate_multisize_real_zero_shot.py`
- 报告目录：
  `offline_analysis/multisize_real_fingerprint_v1/`
- 产物包括完整 `report.json`、人读 `summary.md`、逐点预测、CAD 几何、三张图、
  exploratory template、模型合同、holdout reveal record 和全产物 hash。
- 单元测试 `5/5 PASS`：显式 run allowlist、周期重采样、同 yaw 六角 CAD 射线、
  多峰分离和三套 campaign 完整性。
- 下一阶段固定为：冻结仿真 encoder 只作 teacher/initialization；仅用 48 clean
  学小型 real common-mode/frame/phase adapter 与 calibrated state/circular heads；
  保留多峰 posterior；CAD 独立负责距离；冻结后采新 audit set。
- 本轮全部为离线文件读取与 CPU 分析，没有初始化 ROS/KST/Bota，没有连接机器人，
  没有发送任何运动。

### 离线里程碑 SHA-256

- 主分析源码：
  `1268af25124a4ee42505a241b20f79fada0bad79b8b9aadb83e628119e700e05`
- 零样本 replay 源码：
  `79bcba9596850f6f769e3d23d1be87f0e2148417ac787baf9d9a2f84a8f130d3`
- 单元测试：
  `deea11143f282dbc651d6995fd0b11afd891cee9354174a9727ff3fa8fd1e55e`
- `report.json`：
  `19b36b53a6147a6db7804119d86b8c3f576d40e82f9ad460e573bcab930cad6b`
- `summary.md`：
  `de5b59e63bc835884b169cdb7ebc5929c91cc4d8873391af0f1d43afec367d9b`
- exploratory template：
  `a423f10733fa4de971882bede7cad1592a8b8d15ff2f786053c995be4890bbc8`
- frozen-CNN zero-shot report：
  `06f125262027a1a1a989aa48886d47a493b1687776af8c388b1fc86203190d69`
- `artifact_hashes.json`：
  `a86516413755ded552a0d1d85ed6382105008ec5e65ecbbf5a8af675eecf3dee`

## 2026-07-28 — 23:03 real-first 多尺寸学习、CAD clocking 与 M12 模型控权闭环

### 数据、防泄漏与离线环境

- 新建隔离 CPU 环境 `.venv-offline-learning`，Torch `2.4.1+cpu`；未修改
  GPU/CUDA、ROS、KST 或现有真机 runtime 环境。
- 三个显式 allowlist campaign 共 78 条：48 clean 只用于训练与 normalization，
  12 consumed holdout 只报告，12 quarantine 只诊断，6 sentinel 只看漂移。
- 每条锥扫重采样为 `7 x 360`：WORLD* 六轴 wrench 加 command-Z relative；
  context 为三尺寸 one-hot、目标力/3.5 N、bit radius/M12 radius。
- C6 增强修正为 `R(+k*60 deg)` 同步旋转 XY force/torque，phase roll
  `-k*60 deg`，方向标签 `+k*60 deg`；原样本及增强副本共享 group ID，禁止
  跨 fold 泄漏。
- 单命令流水线为 `offline_learning/run_pipeline.sh`；已存在的
  `real_calibrated_v1` 作为不可变产物复用并由合同测试验证，不覆盖重写。

### CAD 角度结论

- 实体合同为 `v1.5-integrated-flange-tools-yaw-datum`：完整 peg shape 的
  `-15 deg` 已烘焙进 CAD，选定 peg flat normal 与 B/C tangent-plane outward
  normal 都是 flange-CAD `+45 deg`。
- 安装时只做一次 `B/C NORMAL -> +X_F`，于是 peg flat 与 socket flat 都是
  fixture `0 deg (mod 60 deg)`。仿真、训练、增强、runtime 和控制器均禁止再加
  `-15/+15/+45 deg`。
- 该 fixture yaw 是 operator-taught commissioning assumption，不冒充 as-built
  yaw metrology。真机方向标签来自 measured touchdown offset 指向 run-local
  hole centre，因此 real-only 数据不依赖猜测的额外角补偿。
- 修正了 tip-side CAD preview 的屏幕手性说明；全部 18 个 STEP/STL 前后
  SHA-256 完全一致，实体几何未改。

### 训练、bundle 与诚实边界

- 三个且仅三个候选按 clean-only leave-one-size-out 依次比较 catastrophic
  `>90 deg`、p90、circular NLL、state macro-F1：
  `REAL_FOURIER_LINEAR_V1`、`REAL_FROZEN_SIM_ADAPTER_V1` 和
  `REAL_SIM_ADAPTER_TAIL_L2SP_V1`。第三项冻结 encoder/teacher heads，只解冻
  `shared.0` 和 real adapters，并以 `0.005` L2SP 锚定预训练尾层；Tiny CNN
  明确不参与选择。
- 最终仍选择真机-only `REAL_FOURIER_LINEAR_V1`，用全部 48 clean、三个 seed
  重训。冻结 adapter 为 `18/48 >90 deg`、p90 `164.89 deg`、macro-F1
  `0.357`；tail-L2SP 为 `21/48 >90 deg`、p90 `165.19 deg`、macro-F1
  `0.463`，均未因“必须联合仿真”而强行部署。
- LOSO：方向 median `33.62 deg`、p90 `165.23 deg`、`16/48 >90 deg`，
  state macro-F1 `0.518`。跨尺寸泛化仍明显多峰，只允许
  `DEVELOPMENT_CV_CONTROL`，不能表述为生产级准确率。
- 先前已消费的 M12 holdout 仅作报告：4 条中 `0` 条 `>90 deg`，
  median `22.89 deg`、p90 `31.17 deg`。这只支持快速、3 mm 上限的
  M12 development iteration，不是 untouched qualification。
- bundle 为 `model_runtime/bundles/real_multisize_direction_v1`；manifest、
  interface、promotion 三处都固定 `hardware_control_sizes=["m12"]`。
  M10/M8 可推理但不获真机控权。

### Runtime 与四方法

- Python runtime 接受 schema-v2 `7 x 360`、5 维 context，核验三个 seed 的
  model/normalizer/metrics/config hash，输出 48-bin 多峰方向 posterior、三状态
  posterior、TOP1、circular mean、entropy、concentration、APS 和连续 quality。
- 真机命令方向明确使用 posterior TOP1；circular mean 仅记录。弱但完整、新鲜、
  相位覆盖充分的指纹仍产生有界动作，不用低置信门静默降级。
- Proposed+：model/CAD shortcut 后，以实际 TOP1 控制方向为中心计算 80%
  credible corridor（全宽 `12..45 deg`），再在 95% credible support 上使用
  task-aware EIG。若 TOP1 中心的 45 deg 扇区装不下 80% posterior，明确跳过
  corridor 并直接进入 residual EIG，不虚报覆盖率。
- Proposed-local：同一模型 shortcut，以实测 shortcut endpoint 为局部螺旋中心，
  半径为 posterior q95 residual minus capture radius。
- Improved Direct x3：每次新 cone、新推理、新射线；无手写角度 fan，anchor 为
  `0/+0.25/-0.25 mm`。Pure Outward 不锥扫、不调用模型。
- 四方法共用 `4/2/1 mm/s` 自适应表面速度；有限硬保护为 `25 N/0.75 Nm`；
  capture、插入和终止仍由确定性物理控制器负责。
- active M12 日志和 shortcut 证据不再出现 simulation gate；仿真只存在于离线
  候选训练和不可变历史 replay。启动 preflight 对完整 runtime、bundle/hash、
  概率与权限做一次无运动检查，并要求端到端耗时 `<3 s`。

### 无运动验证

- 固定 controller preflight trace 的 SHA-256 仍为
  `e2de7a701031175ffdd0a656f96f2489d28b06cb8f033487227cb5717d4b5aaa`。
  最终 runtime replay 用时 `1.26 s`，TOP1 `191.25 deg`，概率归一，
  `real_input_contract_complete=true`、`hardware_control_allowed=true`。
- 一条 M12 clean trace：真值 `183.9606 deg`，TOP1 `183.75 deg`，
  误差约 `0.21 deg`；信号缩弱到 2% 后仍保持 M12 有界动作权限。
- 最终单命令流水线通过 `44` 个 pytest 和 `2` 个 CAD preview unittest；
  另行覆盖 M12 四方法、M12/M10/M8 sampling 与 dashboard 的 `89` 个 pytest
  全通过。过期 runtime parse 合同和缺失的通用 `采样规划图.png` 快照均已修复。
- 本轮没有启动 MATLAB、ROS、Bota subscriber、KST 或机器人连接，没有发送运动。

### 关键 SHA-256

- active deployment manifest：
  `53aa52849642f1b36e124d8daa47eff3341e4a2fa1f5d833faf98a733454400d`
- ensemble interface：
  `c9658bcb0a78e5c01d65c4023fc5995f4e33b92da455048fa56dbf5016d40048`
- promotion contract：
  `24169f1983f54cd45b753539b29a821727e1d35556638a1fb1aaf1aed9643ae9`
- dataset contract：
  `e8c0f08d04b35c39475bc9e38afbd2d7dd3f6be821d912340b1bc0ebef6595d5`
- non-training replay report：
  `feafbe25a04d4195bd4ffad47d1e971c136aeaa8bbbc7d72066cf1093809eb99`
- candidate metrics：
  `ea525ae783307691e857da0273aafdc93c7dee664b6202adae4c0cec44d571c9`
- training report：
  `5a920144cf3b345c1c242b33384551519e7b592cbbc82615e14f53475431fb1e`
- corrected CAD yaw preview：
  `f4553e4d0ab9537236acad667cb41560be0404654319944bd5cd095f76ee114d`
- actual-CAD M8/operator sampling plan PNG：
  `d7623c1b0c075eec6854794e230fabf6e65d9a759612173d42863dac805b6a68`

## 2026-07-28 23:58 +08 — Proposed-local 首段 KST 幂等握手恢复

- 失败 run：
  `data/M12_proposed_local_spiral_20260728_233555_632`。学习 runtime
  preflight、Bota、KST 连接、静止位姿和空气 baseline 均通过；首个
  `TOUCHDOWN_001` 在 `jRelVel_1_` 后 8 s 内收到 16 次空读。
- 此时 `cArtixanPosition_...` 与 `doPTPinCSRelBase` 均尚未发送，机器人没有
  收到位移触发。旧 summary 因把 command ledger 的 `ISSUED` 当成
  `motion_commanded` 而产生了错误语义；现已拆为 `motion_attempted`，只有
  已完成/已知接受的运动才进入 `motion_commanded`。
- 根因边界是 KST 1.7 的单个、无序号、跨线程 `daCommand` 槽：本地参考 Java
  源码中该字段既非 `volatile`，若干分支又在发送 `done` 后才清槽。该源码
  暴露确定的竞态结构，但当前 cabinet 部署源码未做 hash 证明，因此不冒充
  已在 Sunrise 端修复。
- MATLAB 即时兼容修复只重试两类不会触发运动且幂等的命令：
  `jRelVel_...` 与完全相同的 `cArtixanPosition_...`。每类最多 3 次、每次
  1 s；恢复后清空迟到的精确 `done`，再允许发送 `doPTP`。`doPTP`、
  DirectServo start/stop 和任何运动命令仍禁止重放。
- `kstReadLine` 对 modern `tcpclient` 先轮询可用字节，空窗返回空消息，不再
  为每个 0.5 s 有界读刷一条 MATLAB timeout warning。运动完成 ACK 后保留
  40 ms 协议 quiet gap，降低旧 server “先 ACK、后清槽”吞掉下一查询的概率。
- 新增 helper、motion/target helper、ACK reader 与底层 line read/write 的
  路径和 SHA-256 均进入每轮 manifest；pre-motion 重试耗尽时追加
  `SETUP_FAILED_NO_TRIGGER` 与
  `KST_PREMOTION_SETUP_EXHAUSTED_NO_TRIGGER`，明确
  `motion_trigger_sent=0`。
- 轻量无运动验证：
  `.venv-offline-learning/bin/python -m pytest -q
  tests/test_m12_real_baselines_static.py
  tests/test_m12_sampling_campaign_static.py`，`71 passed in 20.50 s`。
  本次没有启动 MATLAB、ROS、Bota subscriber、KST 或机器人连接。

### 本次关键 SHA-256

- `kstExecuteIdempotentPremotionSetup.m`：
  `92219a4309d688608d00bc3f3df856d2aedd13673fb9bd17a2c79336cd4fd5a8`
- `movePTPLineEefRelBase.m`：
  `4a577f7a2b61f0ef723814da6510e8cec299e662994cd49baf9db5bbd660b68b`
- `sendEEfPositions.m`：
  `6084a6cb43ed4b782d10a4ddcb82b8c9a8fbadae67e7476525c2cd389a9b993c`
- `kstReadLine.m`：
  `64a1bacf27645981fcfb079f6b6e5b52002f46abb486b98fbc2f788c36fdeb44`
- `execute_m12_contact_engagement_policy.m`：
  `d6016512b18e65c4e91a051c340b8b2bc96c96870ad5cc2700fa01de7fa77a52`

## 2026-07-29 00:46 +08 — Proposed-local shortcut 后误停、自适应卸载与学习决策图

### 现场根因

- 失败 run：
  `data/M12_proposed_local_spiral_20260729_001718_814`。锥扫完整：
  240 个 SCAN 点、phase coverage `0.9958`、约 `39.90 Hz`、routing quality
  `0.7495`；冻结模型成功输出 `outer_partial_support`、TOP1
  `356.25 deg`、APS `16`、sample quality `0.3521`。
- 旧 planner 选择 `[+0.500, +0.000] mm`，机器人用两个 XY 微段实际完成约
  `0.499 mm`，离目标只剩约 `0.028 mm`；local spiral 尚未启动。因此肉眼看到的
  “shortcut 一下就停”不是模型、Python 或 3 mm cap 停止。
- 真正异常发生在随后第四个接触软卸载动作。命令为
  `[0, 0, +0.100] mm`，实测位移为
  `[+0.151740, +0.000522, +0.261632] mm`，接触弹性释放使三维 tracking error
  达到 `0.221698 mm`，被自由空间通用 `0.200 mm` 门误判。
- 异常前最新 fresh 观测约 `5.779 N / 0.3097 Nm`，整轮峰值约
  `8.686 N / 0.3377 Nm`，远低于独立 `25 N / 0.75 Nm` 硬保护。

### 控制与 shortcut 修复

- 四方法共享的表面控制器保留普通 `0.20 mm` tracking 和硬保护；接触 soft
  relief 单独使用 `0.35 mm` endpoint tolerance、有限 `0.60 mm` 累计横向释放
  预算，并在每次动作后先做 fresh Bota guard dwell，再以实测端点重基准继续。
- surface torque soft trigger/resume 改为 `0.40/0.32 Nm`，形成力和力矩双迟滞；
  `0.40 Nm` 仍明显低于 `0.75 Nm` hard guard。每个实测释放追加
  `SURFACE_SOFT_RELIEF_MEASURED_REBASE_CONTINUE`。
- 本 run 的低投影 quantile 为零，旧 fallback 因而总取最近的 `0.5 mm` CAD
  节点。新 fallback 保持 TOP1 正向，在质量/r95/capture-margin distance cap 的
  外侧 75% 内选择最高 posterior 节点并仍裁剪到该 cap。对同一保存 posterior
  离线重算，命令距离由 `0.500` 增到约 `0.9516 mm`；3 mm 总上限和后续
  posterior-sized local spiral 均保留。

### 延迟学习决策可视化

- 新增 `m12_learned_postcone_dashboard.m`。Proposed+、Proposed-local 和
  Improved Direct x3 每次锥扫停止且 CPU 推理返回后，才从同一个内存
  `model_input` 批量显示；Pure Outward 按 baseline 定义不调用模型。
- 原轨迹窗叠加 TOP1 控制方向和实际 route。独立窗口显示实际 CAD 尺寸、以
  MAP 孔心候选为中心的 head/mouth/deep-wall、touchdown peg、81 点 CAD
  posterior、TOP1、circular mean、离散 APS、真实 corridor、shortcut/direct
  ray、48-bin posterior、三状态 posterior 及距离计算项。
- 七通道 `Fx,Fy,Fz,Tx,Ty,Tz,cmdZrel` 保留 raw 细线；显示副本用零相位
  1--6 阶圆周 Fourier 重构和 robust 高频残差 MAD 带。该带明确标为
  `DISPLAY ONLY / NOT MODEL CI`，不改变冻结 runtime 输入、normalizer、bundle
  或控制指令。
- 新窗口内无 `drawnow`、timer、系统调用、KST/Bota、subscriber 或文件 I/O；
  DirectServo 流内也没有新增 UI、FFT、状态查询或传感器读取。

### 轻量无运动验证

- 保存的真实输入通过冻结 runtime replay：TOP1 `356.25 deg`、state
  `outer_partial_support`、APS `16`、方向/状态概率和分别为 `1.0`，
  `hardware_control_allowed=true`。
- `tests/` 全量静态/离线检查：`143 passed in 66.14 s`。
- runtime/authority/offline-learning 聚焦检查：`23 passed in 6.98 s`。
- 最终 MAP-CAD 显示和 UI 异常隔离调整后的控制/可视化聚焦回归：
  `65 passed in 0.27 s`。
- 本轮没有启动 MATLAB、ROS、Bota subscriber、KST 或机器人连接，未发送运动。
  当前环境没有 MATLAB/Octave，因此新窗口已做静态 MATLAB R2022a API/维度审查，
  真实图形渲染由下一次现场运行验证。

### 本次关键 SHA-256

- `execute_m12_contact_engagement_policy.m`：
  `0d4385bf6d6f6a5129cbfa238bf21dd0379ba9a4292cb81156bb4dc6e1798c64`
- `m12_learned_postcone_dashboard.m`：
  `6078319e5fae0c16028e43b85ed002493720f40455700d6e3f130323209aef8c`
- `README.md`：
  `f5f1294d9cef21ca0a1be1f8381bc1353f9d811dfb3955b6b43dbcf9780883b3`
- soft-relief regression test：
  `297aadd6a54a000275fd32baad7715aae62694377953b408c058fb2534c949b6`
- learned dashboard regression test：
  `2dc94f216e838b37c9c977cfb682df927d9ede678aadb529e0ec13b106adf812`

## 2026-07-29 00:56 +08 — Proposed-local 第二次 soft-relief 误停与实测接触重定位

### 现场证据

- 失败 run：
  `data/M12_proposed_local_spiral_20260729_004822_035`。锥扫完成后活动模型正常
  输出 `outer_partial_support`、TOP1 `356.25 deg`、APS `15`、sample quality
  `0.375`；CAD-bounded shortcut 为 `[+0.969,+0.000] mm`。local spiral 尚未开始。
- 学习决策窗口的空白不是模型失败。显示器把三项 `stateNames` 裸 cell 放进
  `struct(...)` 后，MATLAB 将显示数据扩成 `1x3 struct`，随后
  `data.payload.cfg` 触发
  `MATLAB:index:expected_one_output_from_intermediate_indexing`。该异常由
  best-effort 显示边界捕获，对控制无影响。
- 真正停止发生在 `SHORTCUT_CONTACT_TRANSPORT_SOFT_RELIEF_02`。命令为
  `[0,0,+0.100] mm`，实测位移为
  `[+0.304408,+0.000917,+0.282861] mm`；旧自由空间式 tracking error
  `0.355110 mm` 仅比 `0.350000 mm` 门高 `0.005110 mm`。
- 实测步长为 `0.415542 mm`，第一次和第二次清载累计横移约 `0.325409 mm`；
  实测 Z `294.671255 mm` 位于有限表面带
  `[293.609,295.309] mm`，关节最小余量约 `14.78 deg`。整轮峰值只有
  `8.106717 N / 0.418164 Nm`，远低于 `25 N / 0.75 Nm` 硬保护。

### 鲁棒与自适应修复

- surface torque soft trigger/resume 从 `0.40/0.32 Nm` 调为
  `0.50/0.40 Nm` 双迟滞。本 run 的 `0.4086/0.4182 Nm` 不再触发无谓清载，
  但 `0.24 Nm` 起的谨慎速度层仍提前生效，`0.75 Nm` hard guard 不变。
- 真正的 soft relief 现在显式调用
  `CONTACT_RELIEF_MEASURED_REBASE`。接触释放不再用自由空间 tracking、
  progress 或 cross-track 语义判死；精确 post-pose/joints 在有限
  `0.60 mm` 实测单步包络内成为下一控制原点。
- 每个被接受的实测重定位之后，调用方在下一条命令前仍立即执行 fresh Bota
  `guardDwell`，随后核验表面 Z 带和 `0.60 mm` 累计横移预算。
  stale/nonfinite、未知状态、工作空间/关节边界及 `25 N / 0.75 Nm` 硬保护未放宽。
- 普通 surface XY/Z、touchdown、capture confirmation 和 insertion 仍沿用严格
  tracking/progress/cross-track 规则；特殊语义只绑定到已知接触 soft relief。
- 学习窗口把三项状态名包装为一个 scalar-cell，保证显示数据始终是 scalar
  struct；新增回归防止重新引入裸 cell 扩张。

### 轻量无运动验证

- 用该 run 的 `commands.csv/events.csv/summary.json` 数值回放：第二次释放在新
  `0.60 mm` 单步包络、表面 Z 带、累计横移预算和硬 F/T 范围内全部通过。
  更直接地，新的 `0.50 Nm` soft trigger 会让本 run 在该处根本不进入 relief。
- controller、四方法逻辑、learned runtime authority、cone guard 和 dashboard
  聚焦回归：`77 passed in 3.41 s`。
- 当前 shell 环境没有 MATLAB/Octave，未执行真实 GUI 渲染；本轮没有启动
  MATLAB、ROS、Bota subscriber、KST 或机器人连接，没有发送运动。

### 本次关键 SHA-256

- `execute_m12_contact_engagement_policy.m`：
  `fec4fbf86ba23e4f03ab02442049cbd9fedc91b92f7a58f95ee74600f244cd62`
- `m12_learned_postcone_dashboard.m`：
  `c9cbebc81b8c91989c67b1eb13e0dee25f6b75d0c608cd5b4a285d767ab06190`
- soft-relief regression test：
  `7e423505873d132b8775d6323ae45763fe8459e8317e3d685330d0594e8006a3`
- learned dashboard regression test：
  `04eb44102932eaf74b61838bc8c3d6080e8ef16d1aeefddd7f275ec939e6c2b6`
- `README.md`：
  `4d4b3a4fba4a88066f1e5d1338707ef9f2a5659dd4ac5670d2eff96f443d7b19`

## 2026-07-29 21:00 +08 — 四方法闭环：1/2/3 首次全部成功；纯螺旋超时定位到重复外周扫掠

### 结果

- **方法 1 Proposed+ learned posterior + residual entropy search：成功**
- **方法 2 Proposed learned shortcut + local outward spiral：成功**
- **方法 3 Improved Direct x3 learned：成功**
- 方法 4 Pure outward spiral：超时失败（唯一失败项），轨迹外圈明显重合。

这是四方法首次有三项在同一轮真机中完成长程插入。前一轮（19:01）三项均卡在
倒角处未入孔。

### 本轮定位并修复的三个缺陷

1. **`makeConfig` MATLAB 语法错误（上一轮引入，已修）**
   注释行被写进 `struct( ... )` 的括号续行内，MATLAB 报
   `Unexpected end of line`。机器人已上电、回零、跑完 preflight 才在
   `makeConfig` 挂掉，废掉一整个装夹周期。所有说明性注释移到 struct 赋值
   语句上方；数值语义未变。
   新增 `tests/test_m12_matlab_struct_syntax_static.py` 专扫该构造，并用
   重新注入原 bug 的副本验证过会命中（原字符串匹配测试全绿却漏掉了它）。

2. **插入阶段低载荷停滞分支是空操作 —— 方法 2 卡死的直接原因**
   `guardedM12Insertion` 的停滞分支把 segment/speed 赋成
   `post_relief_segment_mm / post_relief_speed_mm_s`（0.05 mm / 0.30 mm/s），
   而循环当时正在用这两个值。于是它检测到停滞 → 打日志 → 设成一模一样的值 →
   清零计数器 → 重发同一条无效指令，无限循环。20:54 起深度锁死在 0.698 mm，
   `along -0.000221/0.050000 mm`（命令 0.05 mm 实走 0.0002 mm）。
   因 dF 仅 4.6 N，远低于 13 N soft trigger，其它恢复路径永远进不去。

   物理判读由本 run 自身数据给出：20:53:50 depth 0.660 mm 时 dF 堆到
   11.5 N 深度不动（楔在 1.2 mm 倒角锥面），随后一次横向 dither 使
   dF 掉到 4.2 N、深度立刻推进到 0.79 mm。倒角 45° 锥面上纯 -Z 指令绝大
   部分转成横向反力，越压越死；只有横向移动能解楔。

   改为升级阶梯：Z 步长几何增长 0.05→0.10→0.20→0.40 mm（cap）；第 2 次
   升级起调用新增 `unwedgeInsertionChamferStall` 做黄金角横向扫掠，每点带
   0.04 mm 下压偏置，测到 ≥0.05 mm 下沉即退出并以实测位姿为新插入轴、
   恢复标称步长。扫掠半径按 CAD 定：peg tip AF 8.00 在 AF 10.60 深壁内径向
   余量 1.30 mm，AF 13.00 喇叭口余量 2.50 mm，故 cap 1.60 mm。全程无 +Z
   指令，`25 N / 0.75 Nm` 硬保护不变；不放弃（完成优先）。

3. **纯螺旋每个 cycle 重复扫掠同一外周圆 —— 方法 4 超时与"轨迹重合"的原因**
   螺旋本体是正确的等距 Archimedes 曲线（pitch 1.00 mm/rev），各 cycle 间
   黄金角相移的径向错位为 0、0.382、0.764、0.146、0.528、0.910、0.292、
   0.674 mm，低差异且从不重复 —— 间距没有问题。
   问题在 `archimedeanWaypoints` 末尾无条件追加的一整圈 r = 4.20 mm 收尾
   revolution：**圆在相移下不变**，所以每个 cycle 都原样重走同一个环。
   0.60 mm 弧步下这是每 cycle 160 个 waypoint 里的 44 个（27.5%），偶数
   cycle 的 `outerPerimeterBridgeWaypoints` 还会沿同一个圆再走一遍。方法 4
   没有学习方向、需要的 cycle 最多，因此代价最大，是唯一超时项；其轨迹图上
   外圈也明显比内圈致密。
   收尾 revolution 改为仅第一个 cycle 执行（覆盖性论证只需一次）。8 个
   cycle 合计 waypoint 从 1280 降到 972（-24%），第 2 个 cycle 起每轮 -27.5%。

### 轻量无运动验证

- 全量回归 `172 passed in 68.53 s`；246 个函数块结构平衡；8 个 config struct
  体内无注释行。
- 停滞分支的回归测试从"事件名存在"加强为"必须真的改变某个物理量"：断言
  `stagnationEscalations` 递增、步长几何增长、调用解楔扫掠，并显式断言
  不再出现旧的 `post_relief_segment_mm` 直赋。解楔函数断言无 `+Z` 指令。
- 本轮没有启动 MATLAB、ROS、Bota subscriber、KST 或机器人连接，没有发送运动。

### 已知未处理项

- surface transport 阶段普遍存在 `SUB_TOLERANCE_PLUS_DEADBAND_PROGRESS_ACCEPTED
  along 0.012340/0.080000 mm`：命令 0.08 mm 实走 0.012 mm（15%）。与插入卡死
  同源（接触载荷下指令增量低于实际执行门限），使熵网格/螺旋的实际覆盖远小于
  规划覆盖。**未改动** —— 调整会同时移动四方法的对比基线，宜单独一轮验证。

### 本次关键 SHA-256

- `execute_m12_contact_engagement_policy.m`：
  `76d0fc1905022c74f8eca5819ada61a639279f9d8915f64b05b713250caaf360`
- `tests/test_m12_real_baselines_static.py`：
  `44a3fb6f482457aa009b8a397dd6dbf21c53aa8ff66174ea4f11ee1e24083c69`
- `tests/test_m12_matlab_struct_syntax_static.py`：
  `9cab95d4a872328217d562d4007d684970ffd479b284ce2958ed5924a3cb487c`

## 2026-07-29 22:44 +08 — 纯螺旋搜索期间可视化从不重绘：被误判为卡死而手动中止

### 现象

方法 4 重跑，22:43:30 touchdown 后螺旋正常推进，49 s 内走完 37 个 waypoint，
法向力全程在 2.4–5.7 N 正常带内，无停滞、无反复 relief。22:44:19 操作者按下
Ctrl+C（`操作在以下过程中被用户终止`，堆栈停在 `waitUntilTrue` 等 Bota 消息处）。
**这不是崩溃，run 本身是健康的**；中止原因是可视化窗口始终空白。

### 根因

`updateDashboard` 按 `maximum_refresh_hz` 累积 animatedline 数据，但
**累积不等于重绘**：该函数原本刻意不调用 `drawnow`（注释理由是本机
Ubuntu MATLAB Figures 前端下 `drawnow limitrate` 也可能无限阻塞）。

于是全程唯一的重绘来自 post-cone 的那一次 phase-boundary drawnow
（`LEARNED_POSTCONE_DECISION_RENDERED_DEFERRED ... one phase-boundary
drawnow in 0.000 s`）。方法 1/2/3 都有锥形探索，所以窗口会在 cone 结束后
亮起来；**纯螺旋没有 cone，因此整轮搜索一次重绘都没有**，窗口空白到底。

### 修复

在 `updateDashboard` 内加一次受预算约束、可自禁用的重绘：

- 按墙钟间隔 `search_paint_interval_s = 3.0 s` 触发，远低于数据累积率；
- 用 `tic/toc` 计时，一旦超过 `search_paint_budget_s = 0.25 s`，
  立即把 `painting_enabled` 置 false 并 warning，本轮不再重绘 —— 退化成
  修改前的延迟渲染行为，**绝不阻塞机器人命令线程**；
- 原有的 post-cone phase-boundary drawnow 和 DirectServo 内 UI 挂起不变。

安全性已静态核验：`updateDashboard` 只经 `receiveWrench` 的
`updateVisualization` 标志到达，而 cone/DirectServo 的四处 wrench 读取
（11554、11752、12347、12547 行）全部传 `[], false, false`，所以新重绘
**不可能进入 40 Hz DirectServo 流**，只在 ~1 动作/s 的阻塞式表面搜索路径上运行。

### 轻量无运动验证

- 新增 `tests/test_m12_dashboard_repaint_static.py`：断言间隔/预算配置、
  自禁用分支、可执行行中 `drawnow` 恰好出现一次（注释里的同名文字不计），
  以及 DirectServo 路径全部保持 `updateVisualization = false`。
- 全量回归 `175 passed in 67.67 s`；246 个函数块结构平衡。
- 本轮没有启动 MATLAB、ROS、Bota subscriber、KST 或机器人连接，没有发送运动。

### 注意

本 run 只走到 cycle 1 的第 37 个 waypoint，而 21:00 条目里的外周扫掠精简
**从 cycle 2 起才生效**，因此这一轮尚未验证超时问题是否解决。需要一次跑到
cycle 2 以上的完整 run 才能确认。

### 本次关键 SHA-256

- `execute_m12_contact_engagement_policy.m`：
  `cbd7923b76efe0cc95b76636db90b37f742d1ba192b329190d1901d251bbf10d`
- `tests/test_m12_dashboard_repaint_static.py`：
  `9ef0eeee9deba424edf01527a3bb1df2318db143a6aecf190af656071f7031cd`

## 2026-07-29 23:xx +08 — 四方法全部成功：纯螺旋完成长程插入并通过双通道终端核验

### 结果

**四个方法全部在真机上完成长程插入。**

- 方法 1 Proposed+ learned posterior + residual entropy search：成功（21:00 轮）
- 方法 2 Proposed learned shortcut + local outward spiral：成功（21:00 轮）
- 方法 3 Improved Direct x3 learned：成功（21:00 轮）
- **方法 4 Pure outward spiral：本轮成功** —— 此前唯一的失败项（超时）。

轨迹图标题为 `BOTTOM_FINAL_DUAL_TERMINAL_SET_001_DWELL_03`，即已进入 CAD
终端深度带并通过双通道终端核验，不是中途停止。

### 代码状态

**本轮未改动任何代码**，`execute_m12_contact_engagement_policy.m` 的 SHA-256
仍为 `cbd7923b...`，与 22:44 条目一致。本条目是对 21:00（外周扫掠精简）与
22:44（可视化重绘）两项修改的真机验证记录。

### 观测到的验证要点

- **等距外扩螺旋成立**：轨迹图上约 4 圈同心螺旋间距均匀，外圈不再出现
  此前那条明显致密／重合的 4.20 mm 环带。21:00 条目中"收尾 revolution 仅
  第一个 cycle 执行"的修改在图上可直接确认。
- **超时问题解除**：搜索段约 280 s 完成并进入捕获，全程约 395 s 走完插入与
  终端核验。此前该方法在同等预算下超时。
- **可视化全程可见**：22:44 加入的受预算自禁用重绘生效，纯螺旋（无 cone、
  因而无 post-cone phase-boundary drawnow）在搜索期间窗口持续刷新，
  未再出现空白窗口被误判为卡死的情况。
- 力/力矩留痕与楔入-释放判读一致：搜索段 Fz 在 0–7 N 间随螺旋起伏；
  约 290–330 s 出现 Fz 峰值 ~11 N、Fx 至 -5 N、My 至 0.27 Nm 的集中事件，
  对应倒角接触与入孔过程；其后回落并在终端 dwell 期间稳定。

### 轻量无运动验证

- 全量回归 `175 passed in 67.45 s`（代码未变，复跑确认）。
- 本轮没有启动 MATLAB、ROS、Bota subscriber、KST 或机器人连接，没有发送运动。

### 仍未处理项（延续 21:00 条目）

- surface transport 的 `SUB_TOLERANCE_PLUS_DEADBAND_PROGRESS_ACCEPTED
  along 0.012340/0.080000 mm`：命令 0.08 mm 实走 0.012 mm（15%）。四方法
  现均能完成任务，故该项不再是完成性阻塞，但仍是搜索效率的主要损耗来源；
  调整它会移动四方法的对比基线，宜在对比实验定稿前单独一轮验证。

### 本次关键 SHA-256

- `execute_m12_contact_engagement_policy.m`：
  `cbd7923b76efe0cc95b76636db90b37f742d1ba192b329190d1901d251bbf10d`（未变）

## 2026-07-29 23:30 +08 — 搜索期重绘导致图形前端死锁：已回退

### 现象

proposed-vs-direct 方法 1 首跑，锥形扫描正常完成并 graded（quality 0.9408），
随后**停在 post-cone 渲染不再前进，且主轨迹窗口自始至终没有出现**。

### 根因：22:44 我加的搜索期重绘

22:44 条目为解决"纯螺旋窗口全程空白"加了受预算约束的 `drawnow('limitrate')`。
**该改动与锥形从未同时被验证过**：21:00 轮的方法 1/2/3 早于此改动；22:44 之后
只跑过纯螺旋，而纯螺旋无锥形、不走 post-cone bring-to-front 渲染。本次是二者
首次同时生效，立即死锁。

原注释的警告是对的：*"On this Ubuntu MATLAB Figures frontend even drawnow
limitrate can block indefinitely"*。**用单次调用计时预算不足以使其安全** ——
那次 drawnow 远在 0.25 s 预算内返回，却把图形前端置于损坏状态，随后 post-cone
的 bring-to-front drawnow 永不返回，窗口也因此始终未显示。

用一个真实死锁换一个显示缺陷，是明确的负收益。已完整回退。

### 回退内容

- `updateDashboard` 恢复为不调用 drawnow，并把这次事故的原因写在规则旁边
  （只写"此处不要 drawnow"会招致同样善意的改动再次发生）。
- 移除 `search_paint_interval_s`/`search_paint_budget_s` 配置、
  `painting_enabled`/`last_paint_elapsed_s` 字段、render_policy 中的相应片段。
- `tests/test_m12_dashboard_repaint_static.py` 改为锁定回退后的不变量：
  `updateDashboard` 可执行行中不得出现 drawnow，相关机制不得残留，且强制在
  规则处保留原因说明。全量回归 `176 passed`。

### 遗留

纯螺旋搜索期窗口空白的问题**回到未解决状态**。它只是显示缺陷，不影响任务完成。
若再次尝试修复，**必须在带锥形的方法（1/2/3）上验证**，只在纯螺旋上验证会重复
这次的漏检。

### 本次关键 SHA-256

- `execute_m12_contact_engagement_policy.m`：`2a03b78e53f2eb6b41e31586268e68955a3f58b5f33d7299ccdcb25e271b36d0`

## 2026-07-30 01:52 +08 — proposed-vs-direct 判别位姿定标：39 deg 处 Direct 三次全失手

### 目的与性质

`run_m12_proposed_vs_direct_trial.m` 是**两方法案例研究**，其偏置是**刻意挑选来
区分方法的，不是随机抽样**，文件头、campaign JSON 与 private record 三处均如实
记录，并有测试锁定该声明。论文的一般性证据来自仿真实验与四方法无偏 campaign；
本脚本只用于演示两种方法差异出现在何处。

### 结果

39 deg / r 6.00 mm：`IMPROVED_DIRECT_STRICT_THREE_ATTEMPTS_COMPLETE_NO_CAPTURE`，
三次 attempt 全部 `IMPROVED_DIRECT_ATTEMPT_RAY_MISS`。

### 捕获半径：连续三次估错后终于夹住

真正决定单射线是否失手的是物理捕获半径。先后三个估计全被实测推翻：

- `0.750 mm` 规划器常数 —— 根本不是物理开口
- `1.88 mm` 孔口 apothem 6.50 − 针尖外接圆 4.619（最差时钟角）
- `2.50 mm` (13.00 − 8.00)/2（六角对齐）

三者共同漏掉的是**倒角**：孔口上方 1.2 mm 深 45 deg 倒角使顶面开口在半径方向再
宽 1.2 mm，落在锥面上的针尖是被导入而非被挡。修正后 3.08..3.70 mm。

实测：

| 偏置 / r | 横向偏差 | 结果 |
|---|---|---|
| 20 deg / 4.55 | 1.27 mm | 首次捕获 |
| 24 deg / 6.00 | 2.08 mm | 首次捕获 |
| 29 deg / 6.00 | 2.56 mm | 首次捕获 |
| 39 deg / 6.00 | 3.46 mm | **三次全失手** |

=> 阈值 ∈ (2.56, 3.46)，与含倒角估计交于 **3.08..3.46 mm**。

### 方向模型的锁瓣行为

三次 attempt 的实际射线方向全部为 `[-0.9979 -0.0654]` = 183.75 deg，与预测的
"锁在训练主瓣、误差 = 偏置 − 3.75 deg"完全一致（35.25 deg）。第三次锥形自身的
top1 漂到 243.75 deg，但路由被
`FRESH_POSTERIOR_FIRST_DIRECTION_MICROADJUST_BOUNDED` 约束在首次方向扇区内，
因此三次误差相同。**决策图上的 top1 不等于实际走的方向**，读图时须区分。

### 方法 1 保持原设计

方法 1 的 shortcut cap 仍为 3.00 mm。同样 35.25 deg 方向误差下，shortcut 后距孔
3.95 mm，在 4.80 mm 残差网格内（余量 0.60 mm）。**不把它改成与 Direct 相同的
5.70 mm**：那样虽也能进（3.55 mm），却会抹掉正在演示的机制本身 —— 提议方法刻意
不对学习方向过度承诺，把残差留在网格能覆盖的范围内。此外 `shortcut_cap_mm` 在
`cfg.search` 中为四方法共用，全局修改会污染主对比。

### 公平性

launcher 断言：方向完全正确时 Direct 的射线仍能捕获（|6.00 − 5.70| = 0.30 mm
远小于捕获半径）。因此它失手归因于**瞄准**，不是射线长度不够。

### 未达成项

"Direct 失败一两次后第二次成功"始终未实现，且本点不产生该效果：重试仅横向平移
0.25 mm，只能挽救落在阈值附近 0.25 mm 窗口内的偏差。若确需该效果，窗口在 36 deg
附近（3.19 / 2.94 / 3.44 mm 对 3.08..3.46 mm 阈值），但能否命中取决于真实阈值在
该带内的位置。

### 轻量无运动验证

全量回归 `177 passed`；两个 launcher 与引擎块结构平衡。本轮未启动 MATLAB、ROS、
Bota subscriber、KST 或机器人连接，未发送运动。

### 本次关键 SHA-256

- `run_m12_proposed_vs_direct_trial.m`：`b7ecde44a5a398f02e81bfa15aaaff72d72fbfa6200ce075c901fc8768d21c0f`
- `execute_m12_contact_engagement_policy.m`：`aaa43636ae0bc29994ff8e4a2052c500b3b7d7dd24b4462cbefb58a68319a1fc`

## 2026-07-30 03:08 +08 — 31 deg 处 Proposed 完成插入；倒角段代价与横向力方向线索

### 结果

`run_m12_proposed_vs_direct_trial` 方法 1（Proposed+entropy），r 6.000 mm / 31.000 deg：
`M12_TERMINAL_SET_SUCCESS_CANDIDATE depth 5.815 mm; channel
LOW_WRENCH_STABLE_DEPTH_EVENT`，实物照片确认针已完全坐入。

### 使其成功的修复

前一轮同一位姿失败，深度钉在 1.92 mm 磨了 200 s。定位到根因是
**`INSERTION_BORE_ENTRY_REBASE_TO_MEASURED_TIP` 在针尖仍偏心 1.20 mm 时把插入轴
钉到当前位置**，导致 `captureXYError = activeFunnelAxisXY - currentPose` 恒等于
零，横向修正彻底失效。解楔扫掠已把偏心从 2.19 走到 1.20 mm，只差 0.52 mm 即可
进入 AF 10.60 深孔所需的 0.681 mm 余量，rebase 把这个进展丢弃了。

修复：rebase 现要求针尖已在 `0.5*AF_wall - AF_tip/sqrt(3) = 0.681 mm` 入孔余量内
才执行，否则记录 `INSERTION_BORE_ENTRY_REBASE_DECLINED_STILL_OFF_AXIS` 并保留修正。
本轮该 rebase 在余量内正常执行一次，未被拒绝。

同时放开插入期权限：软包络力 12.0 -> 16.0 N、力矩 0.35 -> 0.50 Nm（实测 dT 长期
在 0.32..0.35 反复触发旧阈值，是此前 8 轮无效 dither 的直接原因）；横向修正步长
0.50 -> 0.60 mm。硬保护 25 N / 0.75 Nm 未变。

**两次因配置断言链被绊**：`capture_xy_correction_step <= capture_xy_tolerance <=
maximum_z_correction_lateral_drift` 与 `funnel_dither_maximum_radius <=
capture_xy_tolerance`。第二次之后改为**提取全部 18 条子句逐条数值验算**，现 18/18
通过。后者是表面阶段安全边界、有独立测试契约，未放宽。

### 倒角段仍不理想

插入段耗时 172 s（03:05:12 捕获 -> 03:08:04 终端确认），期间 2 次解楔释放、
4 次停滞升级、5 次软载 dither。是靠一串恢复机制硬扛通过，而非干净策略。

### 横向接触力方向：真实但尚不可用

针挂在倒角锥面上时，锥面反力的横向分量应指示孔的方向，且为**实测量**，不像学习
方向那样可能整体错 166 deg。对 8 次完成插入的运行做了相关性检验：

- 中位偏差 **+27 deg**，6/8 落在 +4..+50 deg
- 但存在一个 **+152.8 deg** 粗差，整体跨度 149 deg

信号真实存在且方向正确，但偏置与粗差率均未标定，**不足以作为控制信号**。已加入
纯记录：`INSERTION_CHAMFER_UNWEDGE_SWEEP_BEGIN` 现输出该时刻的横向力矢量、幅值与
方位角，明确标注 "not used for control"。后续每轮自动累积样本，待标定后再考虑接入。

### 轻量无运动验证

全量回归 `182 passed`；246 个函数块平衡；插入断言链 18/18。本轮未启动 MATLAB、
ROS、Bota subscriber、KST 或机器人连接，未发送运动。

### 本次关键 SHA-256

- `execute_m12_contact_engagement_policy.m`：`c8831c8b94c5c18abaebf618285a927e65050693dba2fd4cdb7db888b42e8dc6`

## 2026-07-30 03:26 +08 — 判别位姿对比完成：Proposed 一次成功，Direct 第二次成功

### 结果（r 6.000 mm / 31.000 deg，同一位姿）

| 方法 | 结果 |
|---|---|
| Proposed+ learned posterior + residual entropy search | `M12_TERMINAL_SET_SUCCESS_CANDIDATE`，**首次尝试成功** |
| Improved Direct x3 learned | 1 次 `IMPROVED_DIRECT_ATTEMPT_RAY_MISS` 后，**第二次尝试成功** |

两者均在插入段出现 1 次 `INSERTION_CHAMFER_UNWEDGE_RELEASED`，即倒角解楔生效。
这是本脚本设计的目标效果：同一位姿下，单射线基线首发失手需重试，覆盖式残差搜索
一次到位。对比视频与数据已取得。

### 性质声明（重申）

该偏置**刻意挑选用于区分两方法，非随机抽样**，文件头、campaign JSON 与 private
record 三处均如实记录并有测试锁定。论文的一般性证据来自仿真实验与四方法无偏
campaign；本脚本仅用于展示两方法差异出现在何处，不可作为平均性能证据呈现。

### 达成路径中修掉的关键缺陷（当晚累计）

同一形状反复出现：**检测到条件后未做任何改变**。

1. 插入低载停滞分支把 segment/speed 赋成循环已在使用的值 —— 空操作，无限循环
2. 倒角接触假设单帧латch 后永不释放
3. 纯螺旋每个 cycle 重走同一个外周圆（圆在相移下不变）
4. post-cone 三个显示开关被一个 assert 焊死，无法单独关闭
5. 残差扇区耗尽后退化为盲目重复覆盖，且释放掩码时未同时释放后验
6. `INSERTION_BORE_ENTRY_REBASE` 在针尖仍偏心时把插入轴钉到当前位置，使
   `captureXYError` 恒等于零 —— 横向修正彻底失效，这是 31 deg 首次失败的直接原因

### 本次关键 SHA-256

- `execute_m12_contact_engagement_policy.m`：`c8831c8b94c5c18abaebf618285a927e65050693dba2fd4cdb7db888b42e8dc6`
- `run_m12_proposed_vs_direct_trial.m`：`5819becb97c5d883ce76ea9a06ec9b0563b67c7a9e27cb99a50d0181f2350c3c`

## 2026-07-30 04:35 +08 — M10 单方法固定演示入口就绪（仅离线验证，尚未真机执行）

### 本次交付

新增 `run_m10_proposed_plus_entropy_trial.m`，只运行：

`Proposed+ learned posterior + residual entropy search`

它是已完成 M12 演示的 M10-only companion，不含方法菜单、M12 运动或自动重试。
每次失败都在当前实测状态结束并要求人工恢复；同一 campaign 最多允许 3 次分别由
操作者重新调用，且每次的 token、run directory、结果／异常、motion attempted /
commanded、capture 与 completion status 均写入 `attempt_records`，不会被下一次覆盖。

### 固定位置与证据边界

- 每次都必须先用当前 M10 针重新示教当前孔中心；历史绝对 WORLD XY 被拒绝。
- 固定局部偏置：`[+4.5901510469, +0.3008543945] mm`，
  即 `r=4.600 mm / 3.750 deg`。
- 来源样本：`M10-CLEAN_CANDIDATE-SUPPORTED-A003p75`，
  原位置 `r=5.533296312 mm / 3.750 deg`；本点沿同一射线向内移
  `0.933296312 mm`。
- 来源 clean cone 与 sentinel repeat 均完成并安全退回；来源 trace SHA-256 为
  `15851bf5d483a8accd77b54ade0a8d763f2cb4b7d9e1b45e028fe03fa2a5a919`。
- 这些证据只说明采样、锥形扫描和后验方向在相邻位置有真实数据支持，**不说明
  M10 capture／insertion 已成功**；本演示是 in-sample-adjacent fixed pilot，
  不是随机、未见位置、成功率或一般 M10 promotion 证据。

### 独立安全与权限边界

- 基础 bundle/promotion contract 仍为 M12-only；没有改写为一般 M10 控制授权。
- 精确 pilot 仅对本入口、本偏置、本方法和最多 3 次人工调用开放 M10 高层 routing。
- 模型只拥有 posterior、有限 shortcut/corridor 和 residual entropy 排序；capture、
  insertion、terminal 与全部 hard guards 仍由 MATLAB/KST 外部确定。
- 新鲜 M10 孔中心仅保留给 socket-head support fail-stop：
  物理 tip-centre 半径 `5.7924582772 mm`，按控制器最大 `0.600 mm` 接触释放耦合
  位移留量后，所有待发命令必须在 `5.1924582772 mm` 内。该中心从不进入 posterior、
  target ranking 或 steering。blocking move 与 DirectServo 命令在发送前检查，
  实测机器人状态另做物理半径检查。
- M10 活跃几何和深度为：tip AF `6.4 mm`、shaft AF `8.0 mm`、mouth AF
  `10.54 mm`、wall AF `8.54 mm`、head OD `18.975 mm`、socket depth
  `5.0 mm`、成功带 `4.7..5.2 mm`、grid `405` nodes。

### 无运动验证

- 精确 pilot 离线回放：top-1 `183.75 deg`、APS `14`、quality `0.271009`，
  `hardware_control_allowed=true`。
- 同一 M10 trace 不带 pilot：`hardware_control_allowed=false`，
  有效 hardware sizes 保持 `["m12"]`。
- 全量回归：`203 passed in 70.92 s`。
- Python `py_compile` 通过；MATLAB/Octave 在本机命令行环境均不可用，因此没有伪称
  `checkcode` 或 MATLAB 解析通过。
- 本次没有启动 MATLAB、ROS、Bota subscriber、KST 或机器人连接，没有发送运动。
  M10 真机结果必须由后续实际运行目录和 summary/events 另行留痕。

### 本次关键 SHA-256

- `run_m10_proposed_plus_entropy_trial.m`：
  `96f0e91557b2df2215b177610d73c34fd3eceb0b9ff66144771469f9c39a5189`
- `execute_m12_contact_engagement_policy.m`：
  `bfc94865acc0600f5bf6028ec61f365e0d3cce68571c458a5eb673b6a2be61ed`
- `configs/m10_fixed_demo_pilot_authorization_v1.json`：
  `8f09bdc86949ee448d29795ca7100347be4b865083f93dfa7761eae9834dc825`
- `configs/m10_fixed_demo_installation_v1.json`：
  `f16108bed677faae556812fce553112243a2f7e9f79655ab4219a205a5cb1793`
- `model_runtime/infer_conical_direction.py`：
  `c4ec2313f71fb7ac1d57dee0760019b26cbe4caa8b0d08f6d62d40497f3c4cce`
- `tests/test_m10_proposed_plus_fixed_demo_static.py`：
  `414e32febaeebb5e077ac4f275154c32200389bd32f3b4be98014b1d16dd84d5`
- `tests/test_model_runtime_authority.py`：
  `ba00ad8920f26d2fd55401e69bcd7ffbac97409565f344206254cd9f5c0b3029`

## 2026-07-30 04:37 +08 — M10 launcher 首次 MATLAB 解析错误已修复（无运动）

- 首次调用在 `run_m10_proposed_plus_entropy_trial.m:78` 被 MATLAB parser 拒绝：
  固定几何 `assert` 的 `norm(...)` 后多了一个右括号。
- 该错误发生在函数解析阶段，campaign、KST 和机器人运动均未开始。
- 已移除多余右括号；定向静态回归 `11 passed in 0.15 s`。
- 修复后 launcher SHA-256：
  `8ed6740fa56dfa767da0e795953dc7bed4eddd69c677c765aa6b63eef88d625a`。

## 2026-07-30 04:59 +08 — M10 Proposed+ 真机闭环两次成功，调参复测一次通过

### Campaign 与结果留痕

同一固定、in-sample-adjacent M10 campaign 已按顺序保留两条独立记录：

1. `SHARED_INSERTION_BASELINE_V1`
   - run：`data/M10_proposed_plus_entropy_grid_20260730_044101_906`
   - `TASK_SUCCESS_CAPTURE_AND_INSERTION`
   - 深度 `4.833931 mm`
   - terminal：`LOW_WRENCH_STABLE_DEPTH_EVENT`
2. `M10_CHAMFER_FORCE14_TERMINAL_TORQUE010_V2`
   - run：`data/M10_proposed_plus_entropy_grid_20260730_045327_162`
   - `TASK_SUCCESS_CAPTURE_AND_INSERTION`
   - 深度 `4.726441 mm`，位于 M10 的 `4.7..5.2 mm` 成功带
   - terminal：`LOW_WRENCH_STABLE_DEPTH_EVENT`

`data/m10_proposed_plus_fixed_demo_campaign_v1.json` 当前为 `COMPLETE`，
`attempt_count=2`。attempt 1 的原始成功记录未被覆盖，并明确回填为旧控制版本；
attempt 2 有独立 token、run directory 和调参版本。两次均为
`TASK_COMPLETE_KST_END_SENT_PENDING_SMARTPAD_SELECTED_OBSERVATION`：
方法任务和 KST end 已完成，`summary.success=false` 只表示 SmartPAD `Selected`
仍属于人工观察，不是方法失败。

### 调参复测的真实闭环链

- promoted model：`real_multisize_direction_v1`
- learned direction：task `191.25 deg`
- state / APS / quality：`outer_partial_support / 4 / 0.588370`
- selected shortcut：`3.000 mm @ 191.25 deg`
- shortcut 实测捕获 offset：`[-2.767037, -0.687921] mm`
- capture：5 次 frozen-XY 观测，深度 `0.365 mm`，handoff
  `0.384 N / 0.0167 Nm`
- residual EIG steps：`0`；本次是在 learned shortcut 上直接捕获
- insertion：63 次轴向动作；一次 0.500 mm 倒角解楔 sweep 在 2 个动作后
  下落 `0.078 mm` 并释放
- terminal：5 次 capture-axis dither 后进入低力稳定深度事件；最后可见载荷
  `0.665 N / 0.0306 Nm`
- 无 +Z relief，始终留在底部并发送 KST end

### 实测改善

本次只把 M10 insertion soft-force trigger 从 `16` 改为 `14 N`，并把 M10
low-wrench terminal torque 从 `0.08` 改为 `0.10 N m`；M12 参数、深度带、3 次
稳定 dwell、`2 N` terminal force 条件以及 `25 N / 0.75 N m` 独立硬保护均未改。

| 指标 | 原始成功 run | 调参成功 run | 变化 |
|---|---:|---:|---:|
| peak insertion force | 17.532 N | 14.506 N | -17.3% |
| peak insertion torque | 0.3979 N m | 0.4075 N m | +2.4%，仍远低于 0.75 N m hard guard |
| soft-recovery actions / episodes | 15 / 15 | 7 / 7 | -53.3% |
| terminal dither actions | 13 | 5 | -61.5% |
| terminal dither wall time | 38.415 s | 11.652 s | -69.7% |
| capture handoff 到 bottom event | 138.307 s | 110.672 s | -27.635 s / -20.0% |
| campaign invocation wall time | 219.782 s | 186.721 s | -33.061 s / -15.0% |
| path length | 23.065 mm | 18.130 mm | -21.4% |

两次浅插入峰值在新 run 中分别于 `14.299 N` 和 `14.506 N` 进入同一个有保护的
平面释放分支，验证了 14 N 触发按预期提前工作。倒角解楔时机和几何没有再改，
第二次停滞后仍由既有 0.500 mm sweep 成功释放。

### 论文使用边界

这两次 M10 成功可以作为：

- Proposed+ 全闭环跨尺寸可行性实物素材；
- learned posterior shortcut 直接捕获、Bota 事件触发插入和双终端验证的
  定性／case-study 证据；
- 一次真实 trace 驱动的小参数修正及独立复测证据。

它仍是同一固定、相邻已采样位置上的 M10 demo，不是随机位置、成功率或一般
M10 promotion 证据。论文图注应如实写“Proposed+ policy stack；本次在 learned
shortcut 阶段捕获，residual entropy search 已武装但未实际执行”，不能把
`residual_eig_steps=0` 的 run 描述成执行了 residual EIG 搜索。

### 验证与关键 SHA-256

- 真实 MATLAB/KST run 已完成，证明修改后的 launcher 和 controller 在部署环境中
  可解析并执行完整闭环。
- 新增静态回归锁定 M10-only 参数作用域、硬保护不变、控制版本握手、旧成功记录
  迁移及复测追加规则。
- 全量静态／离线回归：`195 passed in 69.04 s`。
- `run_m10_proposed_plus_entropy_trial.m`：
  `fef8ed4595d76db71bddf714626a4ad968c87e03a2069191bfb624fc8b58cb02`
- `execute_m12_contact_engagement_policy.m`：
  `2945fc715471cb1f4aebb015cfa7b7c608e96791853e9920d0fb4c115520e03f`
- `tests/test_m10_proposed_plus_fixed_demo_static.py`：
  `b1e87ad17e01b00aa08f227158873fff913d503f6c7ff40eea2e2da55d9e83c9`
- `configs/m10_fixed_demo_pilot_authorization_v1.json`：
  `8f09bdc86949ee448d29795ca7100347be4b865083f93dfa7761eae9834dc825`
