# v1.5 真机工装 STEP 打印清单（2026-07-24）

唯一几何记录：`generated/manifest.json`  
推荐真机配置：一体式法兰工具 + `hardware_easy_v2` 双边倒角底座。

## 一、最终要打印的 4 个 STEP

| 顺序 | STEP | 用途 | STEP SHA-256 |
|---:|---|---|---|
| 1 | `generated/integrated_flange_peg_m8.step` | M8 完整法兰—peg 一体工具 | `22303873354eeb02d82c2492cb3a5d6b8ba73860b31a67804ca501347692f24b` |
| 2 | `generated/integrated_flange_peg_m10.step` | M10 完整法兰—peg 一体工具 | `17381968212daa2b7758dcf3795df32b3a4e687b42c482d3550111e03ae8daa2` |
| 3 | `generated/integrated_flange_peg_m12.step` | M12 完整法兰—peg 一体工具 | `362018e71e4bb80ea07640d6fb7bc0ddbe6178c25046f0283741e5a49e52a4fe` |
| 4 | `generated/socket_base_hardware_easy_v2.step` | 推荐的三 socket 公共底座 | `f3b26582f409547e63296f7b8afb6f30d52d64f4f97f0dd5f74b4d1ffafab93f` |

这次必须重打的是前三个一体工具。底座的实体几何、孔朝向、间隙和倒角
没有改变；若现有 `hardware_easy_v2` 底座实测合格，无需因 STEP 重新序列化
导致的 SHA 变化而重打。

不再打印 `modified_blue_flange_receiver`、三个 `peg_cartridge`、
`tool_free_retaining_clip`、`quick_change_fit_coupon` 或 `assembly_*`。
它们已从 active 目录移出并完整归档到
`archive/v1_3_quick_change_2026-07-21/`。

## 二、建议先打印的 3 个 socket 小样

正式打印大底座前，先打印：

- `generated/socket_fit_coupon_m8.step`
- `generated/socket_fit_coupon_m10.step`
- `generated/socket_fit_coupon_m12.step`

每个小样包含相对 hardware-easy-v2 深壁目标 AF 的
`-0.10/0/+0.10/+0.20/+0.30 mm` 五档总 AF 偏置。用对应一体式 peg
试插，选择“低轴向力可进入、无明显卡边、插入后横向晃动仍可接受”的档位。
若 0 档已合适，直接使用当前最终底座；不要把未经测量的打印补偿偷偷写成
nominal 几何。

## 三、三个一体工具的打印方式

- 传感器侧法兰平面贴光滑打印板，peg 垂直向上。
- 不使用 raft；默认不需要 support。根部截面随打印高度单调缩小。
- 推荐 0.12--0.16 mm 层高、至少 6 圈壁；中心承力区和 peg 用 modifier
  设为 100% infill。最终参数以材料和机器实测为准。
- 对 M8 尤其限制横向载荷；42 mm 力臂并未因一体化而缩短。
- 六角 seam 固定在一个角附近，打印后去除 tip、四条/六条棱和法兰孔毛刺。
- 检查传感器安装平面的平面度；如需研磨，只做轻微整面研磨并重新测量
  sensor-to-tip 距离。

硬件 peg 倒角：

| 规格 | 主杆 AF | 45° 倒角宽/深 | tip AF | 完整 AF 直段 |
|---|---:|---:|---:|---:|
| M8 | 6.00 mm | 0.60 mm | 4.80 mm | 12.40 mm |
| M10 | 8.00 mm | 0.80 mm | 6.40 mm | 12.20 mm |
| M12 | 10.00 mm | 1.00 mm | 8.00 mm | 12.00 mm |

验收：三个物理 tip 到法兰基准面的高度应一致；当前固定 TCP 合同要求
`42.00 ± 0.25 mm`，tip 轴相对法兰基准轴的径向跳动要求 `<=0.20 mm`。
`0.20 mm` 小于最小的 M8 单边深壁间隙 `0.25 mm`，是保守的首轮 FDM
工程门限，不是假装经过统计标定的制造公差。超出高度带时不能继续沿用
42 mm TCP，应返修/重打，或把实测 TCP 写回并重新资格确认。记录每件实测
高度、跳动、质量、装上传感器后的 tare 和低载荷静挠度。

这里的 `tip_axis_runout_mm` 指 tip 中心相对拟合法兰基准轴的最大径向偏心；
若用百分表得到的是峰—峰 TIR，应除以 2 后与 `0.20 mm` 比较。tip 高度指
粉色 CAD 配合平面到物理 tip 端面中心的垂直距离。

### peg—法兰角度验收

- CAD 中完整 peg（tip 倒角、13 mm 接触段、AF14 过渡段）统一绕法兰
  `+Z` 旋转 `-15 deg`。
- 选定 peg 平边的外法向为法兰 CAD `+45 deg`，应与中央 R36 外圆 A/C
  （B/C）位置的**切平面外法向**平行；对准的是法向，不是切线本身，也不要
  误用四个法兰臂的直侧面。
- 用 `generated/preview_peg_flange_yaw_alignment.png` 对照打印件；目标物理
  平行误差 `<=1 deg`。同时验收 `42.00 ± 0.25 mm` tip 高度和
  `<=0.20 mm` 径向跳动。三个尺寸必须使用同一个基准和同一个相位。
- 预览严格采用“人眼位于 peg tip、沿 `+Z` 看向 flange”的底视图：屏幕向右
  是 `+X`，屏幕向下是 `+Y`，`+Z` 指向纸内，因此绕 `+Z` 的右手正 yaw
  在图上是顺时针。若把 `+Y` 画向上，那是相反的 flange-side 视角，不能拿来
  判断打印件手性。
- 旧 v1.4 三件工具和旧工作姿态不允许与本清单混用。装机后重新教导，使
  法兰 B/C 切平面的**外法向**映射到 fixture `+X_F`。这一次映射会同时让
  已打印的 peg 选定平边法向从 flange-CAD `+45 deg` 变为 fixture `0 deg`，
  与 socket 的 `0 deg` 平边法向重合。CAD 的 `-15 deg` 已经做进实体；
  装机、仿真、训练和在线变换都不得再补一次 `-15/+15/+45 deg`，随后只记录
  实测 peg/socket 残余 yaw。

## 四、hardware-easy-v2 底座

| 规格 | 深壁 AF | 单边深壁间隙 | socket 45° 倒角 | 入口 AF | 外圆直径 |
|---|---:|---:|---:|---:|---:|
| M8 | 6.50 | 0.25 | 0.80 | 8.10 | 14.95 mm |
| M10 | 8.54 | 0.27 | 1.00 | 10.54 | 18.975 mm |
| M12 | 10.60 | 0.30 | 1.20 | 13.00 | 21.275 mm |

- 大底板底面平放打印，三个 socket 向上；默认无需 support。
- `M8/M10/M12` 为 7 mm 粗体、0.8 mm 深凹刻；`+X` 为 6.5 mm。
- 三个中心孔为 0.5 mm 真通孔。FDM 后若不通，只允许用细针沿既有轴线清理，
  不要扩大成新的定位间隙。
- 检查入口 45° 面连续、深壁无 elephant-foot/毛刺、外圈条纹不进入顶部
  2 mm 光滑接触保护带。

## 五、物理边界和首次上机步骤

双边倒角的对齐平边近似导向余量为 M8/M10/M12 的
`1.65/2.07/2.50 mm`，约为原仿真组合几何的 `5.3--5.5 倍`。
它不是保证捕获半径；倾斜、yaw、层纹、摩擦和多点接触仍可卡住。

深壁加宽还带来约 `±9.75°/±7.59°/±6.63°` 的理想居中 yaw 空程。
因此第一次上机按以下顺序：

1. 测量三个 socket 实际深壁 AF、入口 AF、深度和中心坐标；测量三个工具
   的 tip 高度、质量/COM。
2. 已知 socket 六角朝向下先做 yaw 对齐：将法兰 B/C 切平面的外法向
   对齐 fixture `+X_F`；不要对齐切线，也不要对已内置的 `-15 deg` 再补偿。
   记录实际 peg/socket 相对 yaw 后，再以低速、低 Fz 下压。
3. 启用 XY/倾角柔顺和横向力矩限幅；禁止刚性 Z 压到底。
4. 先做无旋转插入与退出，检查回零；再做小角度 bounded seating twist。
5. 终态使用插入深度 + 低横向力/矩 + 有限时间；需要传扭矩时再加小角度
   torque gate，不能只看 Fz 或 Z。
6. 用 `hardware_easy_v2` peg/socket 几何重跑 MuJoCo，重新标定触觉指纹、
   终态门限和时间预算后，才统计可与真机对齐的成功率。
7. 正式比较的初始偏差/yaw 分布必须包含 easy-v2 导向区之外的样本，并对
   所有方法固定相同分布与时间预算；分别记录直接捕获、搜索后捕获、超时、
   安全中止和终态成功，防止机械漏斗让所有方法成功率饱和。
8. 只有当三个打印件的实际相对 yaw 落在现有仿真训练覆盖
   `[-2,+2] deg` 内，且 wrench/锥形相位/posterior/XY 指令统一变换到训练
   canonical frame，才可先复用现有模型做 shadow inference；否则按实测 yaw
   中心重采 M8/M10/M12 并重新 fine-tune/audit。

所有方法共用同一 easy 工装时，方法间比较公平；论文必须披露这是一种工程
捕获偏置，不能称严格 nominal sim-to-real，也不能把成功率提升说成方向估计器
本身变准。
