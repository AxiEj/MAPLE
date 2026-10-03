# Gaussian-CHA 精确单球色散内核 v3：阶段结果

**已完成：独立解析色散内核的冻结数值验证。尚未接入完整 Gaussian-CHA profile，也未开放 FREQ/TS。**

```text
精确色散内核 + 独立验证  已完成
            ↓
新 profile 与现有框架耦合  待独立审查/实现
            ↓
完整 E/F 与 15 组 OPT 重验  待执行
            ↓
FREQ / PRFO / Dimer 重验   待执行
```

## 源码变化

仅新增两个生产模块和两个回归测试文件，未改写恢复的 17 个旧文件：

- `torch_continuum_dispersion_domain.py`：实时单覆盖 SAS 域认证；使用 dispersion 半径和严格尺度保护；不满足条件即拒绝。
- `torch_single_sas_dispersion_v3.py`：同一 sigma 分段标量的精确积分；E/F/H/HVP 保持解析/自动微分，无运行时有限差分或旧积分 fallback。
- 对应两个 `tests/solvation/test_torch_*dispersion*.py` 新测试：域边界、零距离、分段切触、最差浮点相消案例及导数回归。

原网格积分存在方向相关二阶导数误差。新内核去掉坐标网格。开发中另发现“较短球冠”算法会把约百万量级的数相减；现固定采用有条件数界的 all-inner 基准和上球冠修正。保留原失败数据，没有放宽门槛。

## 正式验证

冻结 A5 协议和 191 个候选源文件后，两个独立进程完成同一矩阵；raw、verdict、execution 三类文件逐字节一致。独立代码和架构复核均为 CLEAR。

- 15 个固定三点水样例；25 个合成边界样例（含一个应拒绝的特殊点）。
- 15 个域认证正/负 fixture；另加 1 个最差稳定性回归。
- 每个水样例：72 个独立能量采样用于力校验，90 个解析力采样用于 Hessian 校验，18 个正式 HVP 方向加 2 个辅助方向。
- 每轮观察到 3,675 次公共标量图计算；300 个 HVP 请求各自绑定一个不同的新图。
- 运行时 FD、旧积分及已声明的坐标框架构造禁止入口均无调用；有限差分只在验证器中使用。

| 检查 | 最大误差 | 单位 |
|---|---:|---|
| 能量 vs 独立角积分 | 2.22e-16 | kcal/mol |
| 力 vs 独立能量差分 | 5.07e-11 | kcal/mol/Å |
| Hessian vs 解析力五点差分 | 2.94e-12 | kcal/mol/Å² |
| 直接 HVP vs 稠密 H 乘积 | 2.14e-16 | kcal/mol/Å² |
| 有限旋转 Hessian 协变 | 1.35e-16 | kcal/mol/Å² |

这些是该内核、该冻结矩阵的数值一致性结果，不是实验频率或化学精度。Hessian 的力差分校验也不是独立完整 Hessian oracle；独立能量/力、80 位参考点、解析恒等式和协变检查提供互补证据。

## 质量和保留

- 新内核测试 25 passed；根进程加入旧色散回归后 35 passed。
- 验证器对抗测试 44 passed；Black/Pyflakes/Pyright 通过。
- 保留并重新核验 1,521 个历史证据成员，零变化；191 个候选文件零变化。
- 根工作区、旧 v1 与 OPT-v2 工作区仍干净。没有 commit 或 push。
- 开发期验证器出现的重复案例、图绑定、参考精度和证据身份缺口已修复；旧 scratch 结果仍保持未认证，未搬用作正式结果。

## 明确边界

`standalone_kernel_validated=true`；`qualification_claim/profile_integrated/freq_ts_qualified/scientific_accuracy_claim=false`。

没有声明完整 Gaussian-CHA、OPT/FREQ/TS、真实反应过渡态、普通 `.inp`、任意分子、GPU/MD/IRC 或物理精度支持。后续先做 profile 耦合审查，再执行全模型 E/F 与 OPT 重验，最后验证 FREQ/PRFO/Dimer。

## 可恢复证据

- 主结果：`.omx/benchmarks/route1-cha-single-sas-dispersion-v3-20261003/FINAL_STANDALONE_SUMMARY.json`
- 候选：`FREEZE_B_candidate_001.json`，SHA256 `c70801c6274406735236cd53d0019f13033a3e1dcab98b7ddc0e7ae687ce2a0d`
- 源码恢复包：`CORE_IMPLEMENTATION_SOURCE_v1.tar.gz`，SHA256 `49e2995e6b59eaf712aad0d6a68a42efaf84078dc941c24a2d191efe07bd314b`，基于 commit `90486bd85334395fa16bc40bc8a76b8c4df647fb`。
- 工作区：`.omx/worktrees/cha-dispersion-v3-20261003`。
