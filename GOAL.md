# Route 2 Goal

> **主线硬性精度目标：在预注册、冻结且同口径的目标 benchmark 上，溶剂化自由能 MAE 必须不高于 `1.5 kcal/mol`；未达到该阈值，不能判定本 Goal 完成。**

全局 Goal 保留两个互不冒充的产品 profile：

1. **单 MACE-POLAR + continuum**；
2. **MACE-MDP permanent source + MACE-POLAR induced response + continuum**。

从 2026-08-16 起采用并行 workspace 分工：纯 MACE-POLAR 的 505-development
验收由独立线路执行；本 worktree 只运行和修改 MDP+POLAR hybrid。两条路线
最终都计入全局 Goal，但 checkpoint、scalar/profile、benchmark 和 evidence
不得跨线路拼接。本主线继续使用模型无关的
source/field/continuum/solver/derivative 契约，为后续其他可接收外场的
MLIP 保留模块化接入面。AIMNet2 的 geometry-mediated 自洽仍是独立支线。

## 标量与导数目标

每个 profile 必须先冻结一个可复算、内容寻址的唯一 operational scalar：

\[
E(\mathbf R)=E_\mathrm{vac}(\mathbf R)+G_\mathrm{solv}(\mathbf R,z^*(\mathbf R)),
\]

其中状态方程、选根规则、能量 ledger、continuum、cavity、source/receiver 表示和 checkpoint 都属于 profile identity。随后从**同一标量**取得：

- **E**：总势能与分项明确的溶剂化能；
- **F**：完整保守力，包含 model/source、continuum、moving cavity 和隐式状态响应；
- **V**：明确原点与符号约定的分子 virial；
- **H/HVP**：同一保守力的 Hessian/Hessian-vector product，带误差、对称性和拓扑检查；
- 通过各自 admission 后再供 OPT、FREQ、TS/IRC 与 MD/NVE 使用。

Operational conservative PES 与严格统一变分 Tier V 必须分开声明。原 checkpoint 若不满足 energy-source 共轭，仍可定义并求导一个明确的 operational scalar，但不得冒充共同变分泛函。

## 化学精度硬目标

在**预注册、冻结且同口径**的目标 benchmark 上：

\[
\boxed{\mathrm{MAE} \le 1.5\ \mathrm{kcal\,mol^{-1}}}
\]

该阈值是主线 accuracy admission 的硬目标，必须绑定：

- 数据集、成员清单、几何与构象策略；
- 溶剂、continuum 方程与 cavity；
- permanent/induced source 和 receiver 表示；
- energy ledger 与是否包含 CDS/nonpolar；
- MACE-MDP/MACE-POLAR checkpoint、long-range evaluator 与运行时；
- 代码、配置和 evidence 哈希。

不得用以下方式冒充达标：

- 用 electrostatic component 与实验总溶剂化自由能混比；
- 用 CDS/nonpolar 或其他分项的误差抵消掩盖错误的 electrostatic component；
- 混用不同 profile/backend/checkpoint 的证据；
- 在看到测试集结果后更换 ledger、符号、系数、cavity 或数据子集；
- 用 E 的 MAE 推断 F/V/H 已正确。

最大绝对误差、分子级误差和分项误差必须同时报告；“1.5”专指预注册目标 panel 的 **MAE**。

## 当前进度（不等于最终 admission）

### hybrid 窄域 E/F data-only admission

在 clean commit `7561edfa849732f522093d4eb33dcdace539dec9` 上，新的
rich-v2 合同完成了两次独立 CUDA 冷重放。两次运行的 141-event
measurement SHA-256 均为
`4e60a2597bc4724116857497e38faa56762d9d8b40fa6b7a2abe4711009c38a9`，
全部 root、Richardson、方向、平移、旋转与闭环门通过。机械聚合后发布的
独立 data-only overlay SHA-256 为
`32b05deaaa596cf5872ff1986e54e142a94c49dd0db6f2ac1870ad287beba863`。

该 overlay 只承认 exact local electrostatic profile 的实验性 `E/F`；静态
registry 和 safe default 仍关闭，`H/V/M`、解析力、完整溶剂化、化学精度、
FREQ/OPT/MD 与 Tier V 均未由这份证据开放。

hybrid accuracy 下一步分为两层：已观察的 MNSol-10 只作冻结 known-panel
regression；最终 leakage-safe 主门使用完整 148-row confirmation partition，
并分别披露 102 个无 prior-pilot geometry overlap 与 46 个 overlap 记录。

完整 hybrid 标量现在具有独立、仍关闭的 `Phi0_hyb + G_SMD-CDS` identity。
MNSol-10 的 pre-result 工具链已按 label-free inputs、source-backed seal、单次
prediction terminal、model-free scorer、aggregate-only publisher 分层。
该链先后经过 verified Pro 的两轮 `STOP` 修复审计；semantic-stage claim、
`fsync(file)->linkat->fsync(parent)` 与 boot-safe orphan recovery 完成后，第三轮
最终 verdict 为 `APPROVE`。跟踪审计制品 SHA-256 为
`3fafcae2587f6bdc558d588ccd57a251cd41231297c78648dab1e3897214f58a`；该批准只解除
MNSol-10 执行架构门，不是 accuracy admission。

该 exact known-panel stage 随后在 clean commit
`b0de0aab81e94e4c305122149dfe7ac4498cc08a` 上执行一次并永久关闭：

- `attempt_slot_id`：`dd120f1feb82d5df0325e54ee929e4d4ac5b0cb0d082a49eba3b6c319af5ebb8`；
- `execution_id`：`6418623ebdf43aa370acf21a7e3fb448dd4db9938003e7597da1dbf92ae57687`；
- prediction terminal file SHA-256：
  `4d70342dd626286b56ba71863280d95a777a9156142fbce9c58b27ed024574d5`；
- public aggregate-only failure SHA-256：
  `2b42f99df83069889107f79008b1f5d4b261d5ad31649fe58c2344ac855b57eb`；
- 第一条记录在入账前触发 `solution total minus vacuum does not equal
  predicted DeltaG_solv`，因此 `validated_record_count=0`、未打开实验标签、未进行
  scorer、没有 MAE/RMSE/误差结论。

这是一份工程/数值闭合 terminal negative，不是 hybrid accuracy 的正或负结论；
MNSol-10 不得重跑，完整溶剂化能力仍关闭。执行代码的静态归因表明该门比较
`fl(fl(fl(V+H)+C)-V)` 与 `fl(H+C)`，固定 `1e-12 eV` 容差不能区分大真空能量
消去产生的 binary64 roundoff 与真实 ledger 错误。未来 148-row stage 只能在前瞻性
注册新的 scale-aware float64 closure 后执行，不能用这次失败反向调参。
verified Pro 的后续数值审计批准前瞻性合同：以 `math.fsum(H,C)` 直接构造
`DeltaG`、以 `math.fsum(V,H,C)` 独立构造 `Phi`，用 exact-dyadic shadow 做位级
校验，只让冗余 `Phi-V=DeltaG` 接受精确有理 ULP envelope。跟踪 failure-analysis
SHA-256 为 `d58f67d9df08c3e5f30fae4f7c912b34a41711f9a7537001e2ea13779d6d7c1a`；
`maple/solvation/release/float64_ledger.py` 的 14 个对抗测试通过，但该合同只供新的
148-row preregistration，不能追认 MNSol-10。

### 固定 zero-field source 的结构等变 continuum 精度门

新的、独立 profile

```text
route2-pure-macepolar-point-l1-frozen-
smoothpartition-harmonic-ddcosmo-smd-v1
```

已经用预注册参数完成冻结 MNSol-10/10-solvent 面板：

- MAE：`0.9194258754 kcal/mol`，通过 `<= 1.5 kcal/mol` 硬门；
- RMSE：`1.0261979389 kcal/mol`；
- 最大绝对误差：`1.6479417114 kcal/mol`；
- 对同一 point-source pyddx ddPCM 基线，逐条 polarization component 的
  平均绝对差为 `0.1921734816 kcal/mol`，最大为 `0.6284443996 kcal/mol`；
- 完整埋藏的 1-propanol O-H hydrogen chart 保留固定 48 维 Schwarz 系统，
  rank 为 `48/48`，不再出现 weighted-shell 的 `48 -> 44` 降秩；
- 单球解析值、真实 pyddx ddCOSMO 收敛、SO(3) 旋转、平移、source
  JVP/VJP、坐标有限差分和同标量 HVP 均已通过候选级测试。

该结果证明：在不拟合 source/cavity/CDS、且不使用 laboratory-fixed
surface grid 的条件下，结构等变 harmonic continuum 可以达到主线精度目标。
它仍是 **frozen zero-field source + ddCOSMO dielectric scaling + SMD-CDS**
的候选证据，不等于 MACE-POLAR mutual self-consistency、finite-dielectric
ddPCM、公开保守力、Tier V、Hessian/FREQ 或 MD admission。

权威证据：

- 预注册：
  `docs/implicit-solvation/benchmarks/route2-mnsol10-harmonic-ddcosmo-preregistration-v1.json`；
- 公开 aggregate：
  `docs/route2/evidence/mace-polar-point-l1-harmonic-ddcosmo-mnsol10-accuracy-v1.json`。

### finite-dielectric harmonic ddPCM 精度门

预注册的独立 finite-dielectric profile

```text
route2-pure-macepolar-point-l1-frozen-
smoothpartition-harmonic-ddpcm-smd-v1
```

已经在相同冻结 MNSol-10/10-solvent 面板上完成一次性运行：

- MAE：`0.8834415187 kcal/mol`，通过 `<= 1.5 kcal/mol` 硬门；
- RMSE：`1.0050963427 kcal/mol`；
- 最大绝对误差：`1.6482383685 kcal/mol`；
- 使用 published finite-dielectric 两阶段方程，而不是 uniform COSMO
  dielectric scaling；
- 单球解析值、真实 pyddx ddPCM 收敛、内外球切触、SO(3)、平移、source
  JVP/VJP、坐标有限差分和 HVP 双线性对称均已通过候选级测试。

该结果把 finite-dielectric、结构等变、平滑 continuum 的冻结精度门闭合，
但仍是 zero-field source 证据。下一主线是 checkpoint-native 八通道 receiver、
mutual MACE-POLAR SCF、MACE-MDP permanent + MACE-POLAR induced 的 separated
耦合，以及同一 operational scalar 的完整隐式 E/F/V/H/HVP。

权威证据：

- 预注册：
  `docs/implicit-solvation/benchmarks/route2-mnsol10-harmonic-ddpcm-preregistration-v1.json`；
- 公开 aggregate：
  `docs/route2/evidence/mace-polar-point-l1-harmonic-ddpcm-mnsol10-accuracy-v1.json`。

### 既有 separated ddPCM 进度

当前 `MACE-MDP point permanent + MACE-POLAR Gaussian induced + ddPCM` 实验候选在四分子、固定几何、electrostatic-only 的跨-backend诊断中得到：

- MAE：`0.869089 kcal/mol`；
- 最大绝对误差：`1.763092 kcal/mol`；
- 四个自洽根残差均小于 `1.5e-11 eV`。
- 真实双 checkpoint 水分子解析 block-adjoint 力已与同一 scalar 的
  Richardson 单坐标结果符合到 `5.50e-11 eV/Å`；
- 独立全坐标随机方向检验在 `h=2e-4,1e-4,5e-5 Å` 下的误差依次为
  `5.31e-7, 1.35e-7, 3.01e-8 eV/Å`，呈二阶收敛；自洽根残差为
  `4.35e-13 eV`，adjoint 残差为 `2.04e-11 eV`。

这说明候选值得继续推进，并在该小型诊断上达到数值目标；它**尚未完成最终 accuracy admission**，因为比较仍是 ML/ddPCM 对 QM/PCMSolver 的跨-backend electrostatic component，且只有四个固定几何，没有同-backend matched decomposition、完整溶剂化自由能或畸变构型覆盖。

### 修正后的 heterogeneous harmonic ddPCM 主线

此前 harmonic hybrid 将 permanent/induced source 误压成同一种表面 source
语义。当前 general-source v2 已改为 ddPCM 的完整 primal/adjoint 结构：

```text
permanent: MACE-MDP exterior point q/p
induced:   MACE-POLAR 1.5-A Gaussian delta-q/delta-p
drive:     complete phi-side adjoint -> native 1.5/3.0-A U8 field
state:     stored primal X plus adjoint xi
```

在**看结果前冻结**的四分子、固定几何、electrostatic-only 诊断中，
`surface_lmax=3` 的新 harmonic backend 得到：

- MAE：`0.8466964716 kcal/mol`，通过 `<= 1.5 kcal/mol` 目标；
- RMSE：`1.1417493529 kcal/mol`；
- 最大绝对误差：`1.7191664851 kcal/mol`（benzene）；
- 四个体系的 cold/wide roots 全部通过预注册的 source/field/energy 一致性门。

这比相同电子侧、pyddx/ddPCM 跨-backend诊断的
`0.8690891909 kcal/mol` MAE 略好，说明先前约 `2.58 kcal/mol` 的 harmonic
退化主要来自 source 语义错误，而不是 spherical-harmonic ddPCM 架构本身。
该比较仍然是跨-backend electrostatic component，不是最终完整
`Delta G_solv` admission。

同一个 general-source v2 profile 的真实双 checkpoint 水分子解析力也已
完成同一标量检验。对非刚体单位方向，中心差分步长
`2e-4, 1e-4, 5e-5 A` 的绝对误差依次为：

```text
1.6923373e-6
4.2245531e-7
1.0640590e-7 eV/A
```

误差按二阶中心差分规律下降；adjoint 真残差为
`1.598e-13`，平移梯度和为浮点零，cold/wide root 差为
`8.79e-12`。当前点的 residual Jacobian 最小奇异值为 `0.96362`，但这只
证明局部 branch，不证明整个化学域全局单根。

hybrid profile 现在通过模型无关的 `OperationalImplicitPES` 门面取得
same-scalar 导数，而不是为 MDP/POLAR 写一套专用求导代码：

```text
MACE-MDP + MACE-POLAR:
  point permanent source + 1.5-A Gaussian induced source
  + native 1.5/3.0-A receiver
```

该门面从各自冻结的 equation/ledger/root 取得同一标量的内部
`E/F/molecular-virial/HVP/H`；`H` 由相同 Richardson force-HVP 后端组装，
不是另一套能量或力实现。真实水分子 HVP canary 使用
`1.25e-4 A` 与其逐次二分步长，结果为：

| profile | root residual | adjoint residual | max HVP Richardson error (`eV/A^2`) | projected bilinear asymmetry (`eV/A^2`) |
| --- | ---: | ---: | ---: | ---: |
| MACE-MDP point + MACE-POLAR induced v2 | `8.28e-13` | `5.27e-16` | `1.26e-3` | `6.90e-8` |

该 hybrid 记录通过固定的 `5e-3 eV/A^2` HVP 误差门和
`2e-4 eV/A^2` 双线性对称门。该结果证明当前水构型上的同标量二阶
方向导数链已闭合；它不替代畸变 PES、完整 Hessian/FREQ 或多溶剂完整
自由能验证。共同 artifact 中另含一个 pure MACE-POLAR 诊断记录，但该记录
属于另一线路，不作为本 Goal 的证据。

权威候选证据：

- 预注册：
  `docs/implicit-solvation/benchmarks/route2-mace-mdp-polar-harmonic-ddpcm-general-source-four-prereg-v1.json`；
- 四分子精度：
  `docs/route2/evidence/mace-mdp-polar-harmonic-ddpcm-general-source-four-4cf8db40.json`；
- 真实解析力：
  `docs/route2/evidence/mace-mdp-polar-harmonic-ddpcm-general-source-water-force-4cf8db40.json`。
- hybrid HVP/virial（共同 artifact 中只采用 hybrid record）：
  `docs/route2/evidence/operational-harmonic-ddpcm-water-hvp-canary-4cf8db40.json`
  （artifact SHA-256
  `eaaa9fc49d5c8efb13cb0993929c76b3f2201d109f5501e81986a420a066f7cc`）。

性能证据也说明当前不应先改 MACE-MDP 的科学身份：在这次七几何
三步长力重放中，MDP CPU/float64 模型加载约 `6.03 s`，全部 MDP+POLAR
anchor 约 `13.17 s`；相对地 continuum 组装约 `24.88 s`，全部 root
solves 约 `147.09 s`。MACE-POLAR 已在 CUDA 上且位于每轮 root 中；
MACE-MDP 只在每个新几何生成一次 permanent source。后续可以增加独立、
内容寻址并通过 CPU/CUDA parity 的 MDP-CUDA v2，但它不是当前主瓶颈，
也不能通过静默修改已审计 CPU profile 来实现。

## 工程与科学约束

- 优先复用官方 checkpoint、pyddx/ddX、PCMSolver、已有 SMD/CDS 与 provenance 组件；
- 模型、source/receiver、continuum、ledger、solver、导数和 benchmark 分层；
- permanent point source 与 induced Gaussian source 不得被一个统一 Gaussian kernel 偷换；
- 内部 experimental callable 与 release admission 分离，所有未通过能力继续 fail closed；
- 不以拟合、校准、误差抵消、benchmark gaming 或局部补丁替代物理定义；
- 保持原框架的模块性、逻辑性、科学边界、整洁性、效率与可维护性。


## 高难数学推理的 Pro 模型交叉咨询

当隐式微分、伴随方程、变分/共轭关系、算子谱与稳定性、Hessian 或其他
高难数学推理对科学正确性产生实质影响时，应主动使用 **Chrome 向 Pro 模型
发起独立、只读的交叉咨询**，而不是仅依赖单一路径推理。咨询必须遵守：

- 只提交抽象方程、已公开资料和经过最小化/脱敏的问题；不得泄露冻结 benchmark
  的私有逐记录身份或标签、sealed confirmation、私有 checkpoint、绝对路径或凭据；
- Pro 模型的回答仅作为第二意见和反例来源，不替代本地推导、官方/上游文献、
  数值残差、有限差分、JVP/VJP 转置检验、测试或冻结 evidence；
- 对采用或拒绝的关键建议记录问题、假设、结论及本地核验结果；不能据此在看过
  benchmark 结果后改预注册口径、拟合残差、调参或进行 benchmark gaming；
- 若 Pro 建议与冻结 profile、来源代码或可复现实验证据冲突，以可审计证据为准，
  并把冲突保留为显式 unknown，而不是静默改写科学声明。
