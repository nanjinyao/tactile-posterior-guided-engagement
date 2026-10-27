# Posterior-risk-bounded shortcut：定义、性质与证据边界

## 1. 这项创新到底是什么

这里的“捷径”不是网络回归出的中心坐标，也不是“置信度高就直接进洞”。它把一次四通道锥形接触指纹、视觉先验和软 CAD 状态似然组合成中心 posterior，再只沿估计方向消费其中一段保守的一维距离；剩余二维不确定性仍由 residual search 处理。

因此准确表述是：

> a posterior-risk-bounded shortcut followed by residual search

不应表述为 `APS-gated guaranteed insertion`、`center regression` 或 `certified funnel`。

## 2. 四通道指纹与圆周 posterior

固定低力锥形动作给出

$$
y(\phi)=
\begin{bmatrix}
F_z(\phi)&\tau_x(\phi)&\tau_y(\phi)&z_{tip}(\phi)
\end{bmatrix}^{\!\top},
\qquad \phi\in[0,2\pi).
$$

信号重采样为 $4\times360$，三个独立 seed 的温度标定方向概率取均值：

$$
p_k=\frac{1}{M}\sum_{m=1}^{M}
\operatorname{softmax}\!\left(\frac{g_m(y)}{T_m}\right)_k,
\qquad M=3,quad k=0,\ldots,47.
$$

48 个方向 bin 的中心为 $\theta_k$。控制方向与圆周 concentration 为

$$
\mu=\arg\sum_kp_ke^{i\theta_k},
\qquad
\kappa=\left|\sum_kp_ke^{i\theta_k}\right|.
$$

令三个 seed 的 top-1 方向最大圆周间距为 $\Delta$。APS 集合只用于描述/旧结构门：

$$
A_\tau=\min\left\{A:\sum_{k\in A}p_k\ge\tau\right\},
\qquad \tau=0.9605978727.
$$

当前连续距离公式并不直接把 $|A_\tau|$ 当输入；旧 binary structured gate 仍被保留为 telemetry，但不是非零捷径的执行开关。

## 3. 从方向 posterior 到二维中心 posterior

设首次探测位置为 $x_0$，视觉残差候选为

$$
\delta_j\sim\mathcal N(0,\sigma_v^2I),
\qquad \|\delta_j\|\le R_v,
\qquad C_j=x_0+\delta_j.
$$

M8 paired contract 使用 $\sigma_v=2\,\mathrm{mm}$、$R_v=\sqrt{5.991}\sigma_v=4.895\,\mathrm{mm}$。候选按方向 posterior 与软 CAD 状态似然加权：

$$
w_j\propto
p_{b(\arg\delta_j)}
\left[0.02+0.98\sum_s p(s)L_s(\|\delta_j\|)\right].
$$

其中 $p(s)$ 是同一指纹的 support-state posterior；$L_s$ 由 head 外缘、socket 上口和 $0.35\,\mathrm{mm}$ 平滑过渡构造。最终重采样 $N=4096$ 个中心假设。

## 4. 捷径长度的实际实现

令

$$
\hat d=
\begin{bmatrix}\cos\mu&\sin\mu\end{bmatrix}^{\!\top},
\qquad
z_j=(C_j-x_0)^\top\hat d,
\qquad
p_-=\frac1N\sum_j\mathbf1[z_j\le0].
$$

先把 concentration、seed agreement 和反向 posterior mass 压成连续质量因子：

$$
q=\operatorname{clip}\!\left(
\kappa[\cos(\Delta/2)]_+(1-2p_-),0,1
\right).
$$

随后定义

$$
\alpha(q)=0.10+0.15q,
\qquad
\lambda(q)=0.35+0.30q.
$$

实际执行距离为

$$
L^*=\begin{cases}
0,&p_-\ge0.20,\\[2mm]
\min\!\left\{
\left[Q_{\alpha}(z)-0.25r_{cap}\right]_+,
\lambda Q_{0.99}(\|C-x_0\|),
L_{hardware}
\right\},&p_-<0.20,
\end{cases}
$$

其中未设置硬件 cap 时省略 $L_{hardware}$。命令目标为

$$
x_s=x_0+L^*\hat d.
$$

这个规则可解释为“有约束的一维最大前进”：低质量 posterior 使用更低的投影 quantile 和更小的径向消费比例；反向质量达到 $0.20$ 时退化为零捷径。

## 5. 能证明什么，不能证明什么

在固定经验 posterior 上，当 $p_-<0.20$ 时，代码结构给出

$$
\widehat P\!\left[z\ge L^*+0.25r_{cap}\right]\ge1-\alpha,
\qquad
L^*\le\lambda Q_{0.99}(\|C-x_0\|).
$$

这约束了沿 $\hat d$ 方向的 posterior 过冲风险和单次消费比例，但不是二维 capture 保证。移动后的 residual 半径定义为

$$
R_{res}=\left[
Q_{0.99}(\|C-x_s\|)-r_{cap}
\right]_+,
$$

所以

$$
\widehat P[\|C-x_s\|\le R_{res}+r_{cap}]\ge0.99.
$$

局部 outward spiral 的螺距为

$$
h=0.8(2r_{cap}).
$$

但有限 spiral 没有外边界 padding，且当前命令执行没有独立 endpoint tracking tolerance，因此它不是完整域覆盖证书。

对 M8，$r_{cap}=0.25\,\mathrm{mm}$，当前投影余量只有

$$
0.25r_{cap}=0.0625\,\mathrm{mm}.
$$

已记录案例中 contact-limited endpoint tracking error 可明显超过该值。因此当前最强诚实结论是 `posterior-risk-bounded command`，不是 `tracking-certified shortcut`。

## 6. 三个冻结数值例子

| case | APS | $\kappa$ | $\Delta$ | $p_-$ | $q$ | $L^*$ | $R_{res}$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| clear `sobol_000` | 3 | 0.997 | 0° | 0 | 0.997 | 2.873 mm | 1.732 mm |
| fuzzy `sobol_016` | 8 | 0.835 | 37.5° | 0.059 | 0.698 | 0.903 mm | 3.402 mm |
| reverse-mass `sobol_002` | 10 | — | 127.5° | 0.257 | 0 | 0 | fallback |

这三个例子说明距离随 posterior 质量连续变化，而不是按 APS 大小做二值开关。

## 7. Proposed 与 Proposed+ 的语义合同

Proposed 与 Proposed+ 在同一首次 raw fingerprint、视觉/CAD prior 和随机 seed 下共享完全相同的首次 posterior shortcut。分叉发生在捷径之后：

```text
shared first fingerprint and first shortcut
├─ Proposed  -> residual outward search -> conditional full outward fallback
└─ Proposed+ -> measured-path Bayesian updates -> task-aware grid
               -> conditional outward fallback
```

旧 `transport_distance_mm` 对 Proposed+ 表示序列化时的 latest planned distance，可能被后续 belief replan 覆盖，且该 replan 未必真正执行。新 telemetry 分开记录：

- `first_shortcut_*`：首次共享计划；
- `latest_planned_*`：最后一次重规划；
- `latest_plan_executed`：该计划是否已下发控制器；
- `executed=true` 仍不等于 measured tip 到达命令 endpoint。

例如 `sobol_016` 的首次捷径是 $0.903266\,\mathrm{mm}$；旧结果里的 $0.542486\,\mathrm{mm}$ 是第 4 次 belief update 的后续 replan，第 5 条 grid segment 已捕获，因此它不是第二条实际执行的捷径。

## 8. 论文可证伪实验

核心捷径应至少与以下共享 residual/physical funnel 的 paired ablation 对比：

1. $L=0$：完全保留 residual policy，只去掉捷径；
2. fixed-length shortcut：隔离自适应长度的作用；
3. legacy binary gate：隔离连续质量映射的作用；
4. continuous proposed：当前公式。

报告 paired strict success、capture time、XY/contact path、hard stop、endpoint tracking error 与 fallback 次数。只有这些实验完成后，捷径才能从“清晰的算法创新”升级为“被独立消融支持的性能贡献”。

## 9. English claim sentence

> We convert a calibrated circular tactile posterior and a soft vision/CAD center posterior into a quality-adaptive lower-quantile transport command. The command spends only a bounded fraction of posterior radial support, while residual search remains responsible for capture. It is a posterior-risk-bounded shortcut, not a tracking or insertion certificate.

