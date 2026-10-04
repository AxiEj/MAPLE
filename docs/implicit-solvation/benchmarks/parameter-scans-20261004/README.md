# 参数扫描结果：可直接用于云端分析

本目录发布已完成的扫描结果，不包含模型权重、原始几何、运行缓存或整个 `.omx`。
**先读取 `configuration-summary.csv`**：83 行数据集/配置汇总，适合直接绘制
MAE–lmax、MAE–Lebedev 网格曲线以及检查失败数。完整精度保留在 CSV/JSON 中。

## 文件与范围

| 文件/`study` | 配置 | 实现与用途 |
|---|---|---|
| `fp64-lmax-high.json` | lmax=7/9/11/13/15，1202 点 | 已有 native-FP64，完整 FreeSolv 642 / MNSol 653 |
| `fp64-lmax-low.json` | lmax=1/2/3/4/5，1202 点 | 已有 native-FP64，同一参考与输入；没有补跑 lmax6 |
| `fp64-grid.json` | lmax15，110/302/590/770/974/1202 点 | 已有 native-FP64 网格扫描；1202 参考复用 |
| `fp32-historical-comparison.json` | lmax15，1202 点 | **历史私有研究 FP32 后端**与 native-FP64 对比，不是本次原 Torch 路径 |
| `fp32-original-dense.json` | 194/302 点 × lmax1–7 | **本次原 `TorchDDPCM` FP32、完整稠密矩阵、两次直接求解** |
| `configuration-summary.csv` | 上述 5 组研究共 83 行汇总 | FreeSolv、MNSol；已有 combined 汇总也保留 |
| `freesolv-records.csv` | 19,902 行，31 配置 × 642 个输入 | 5 组研究的逐分子 FreeSolv 结果，失败项未删除 |
| `provenance.json` / `manifest.json` | 原始证据哈希 / 发布文件哈希 | 可追溯与完整性检查，不依赖本机绝对路径 |
| `verify_results.py` | Python 标准库 | 云端校验文件哈希、覆盖范围及 FreeSolv 重新聚合 |

**MNSol 只发布配置汇总，不发布逐行结果。** 这是仓库现有
[`route2-mnsol-protocol-v1.json`](../route2-mnsol-protocol-v1.json) 与
[benchmark 数据约定](../README.md) 的边界。不能把本目录称为完整 MNSol 逐行数据包；
若需按分子/溶剂任意重新分组，须在合适的私有环境提供获授权的逐行数据。
FreeSolv 逐行文件来自 FreeSolv v0.52 基准上的计算结果，沿用项目
[`route2-protocol.json`](../route2-protocol.json) 的数据来源说明和已有冻结输入；
本次没有重新获取数据。

## 云端读取

不需要 Torch、MACE、PySCF、pyddx 或模型权重即可分析结果：

```bash
python docs/implicit-solvation/benchmarks/parameter-scans-20261004/verify_results.py
```

在已有 pandas 的 notebook 中：

```python
import pandas as pd
from pathlib import Path
p = Path("docs/implicit-solvation/benchmarks/parameter-scans-20261004")
s = pd.read_csv(p / "configuration-summary.csv")
a = s[(s.study == "fp32-original-dense") & (s.dataset != "combined")]
print(a.pivot(index="lmax", columns=["dataset", "n_lebedev"], values="mae_kcal_mol"))
a.pivot(index="lmax", columns=["dataset", "n_lebedev"], values="mae_kcal_mol").plot(marker="o")
rows = pd.read_csv(p / "freesolv-records.csv")
```

`study` 必须参与分组：同一个分子或参考配置可能出现在多个研究中，不能跨 study
直接把行数相加当作独立样本，也不能把历史私有后端当作原 Torch 的结果。

## 列定义与解释边界

- `mae_kcal_mol` / `rmse_kcal_mol`：对实验值的误差，单位 kcal/mol。
- `reference_mae_kcal_mol`：**同一个统计面板**上的 native-FP64 lmax15/1202 参考 MAE。
- `mae_change_kcal_mol`：当前 MAE 减匹配参考 MAE；历史 FP32 比较这一列由已验证的两项子集 MAE 相减，不是全数据集 MAE 增量。
- `mean_abs_energy_delta_kcal_mol` / `max_abs_energy_delta_kcal_mol`：逐分子计算能量相对参考的偏差，**不等于对实验 MAE**。
- `metric_panel=full`：全数据集有限；`fixed-cds-finite`：预先固定 CDS 有限面板；`matched-finite-subset`：历史配对有限子集；`unavailable-full`：不提供失败配置的成功子集 MAE。
- CSV 空格值/JSON `null` 表示不可提供或未评估，不是 0；`metric_panel_n=0` 表示该配置没有可发布的规定面板指标，**不代表没有有限单行**，有限数看 `finite_records`。
- `full_panel_mae_kcal_mol`：只有完整数据集有限才有值。不能用上述子集 MAE 填补。
- FreeSolv 逐行 `raw_finite` 仅表示有限能量；`common_gate_pass` 空值为未在该研究评估，不可解释为通过。`reference_prediction_kcal_mol` 在失败行仍保留，便于查对已有参考。

## 关键结论和失败项

最新原 Torch 扫描：1295 个输入 × 14 配置 = 18,130 项全部有记录；
17,010 次原 Torch 调用均得到有限原始能量，另外 1120 项被既有的 80 个 FP32 CDS
失败阻断，未运行 continuum。没有新增 continuum 执行失败。

每个 FP32 配置的有限面板为 FreeSolv **641**、MNSol **574**、combined **1215**；
这些值都不是完整 642/653/1295 面板 MAE。保留的 `1e-12` 残差门槛仍未通过，
common-gate 通过数为 0，**所有本目录结果均不声明科学准入或生产可替换**。
最新比较是 FP32/source/CDS + 降网格 + 降 lmax 的联合偏差，而非单独 dtype 偏差。
本次未新增数值差分计算；代码的导数验证为 analytic/autograd，扫描文件本身只报告能量。

302点/lmax7：FreeSolv MAE **1.5111787535312624**（+0.006981196475984053）；
MNSol MAE **1.316681176966824**（+0.012803961417354515）kcal/mol。
194 点实验 MAE 略低不能推出积分更准确，存在误差抵消。

FP64 网格 110 点有 1189/1295 个 native 求解失败，因此全数据集 MAE 不可提供；
不是把剩余 106 个有限结果当作全量精度。用户目标点数映射在 `fp64-grid.json`：
100→110，300→302，500→590，700→770，900→974，1100/1200→1202。

最大 46 原子、302点/lmax7 独立 FP32 能量探针峰值 RSS 为 **1,012,648 KiB**。
同规格一阶保守预算 FP32 **2,539,141,120 bytes**、FP64 **5,078,282,240 bytes**；
50% 是预算下降，不是实测进程 RSS 减半。没有受控速度对照，不发布加速比。

## 可复核范围

FreeSolv 逐行数据可在云端独立重算本包全部 FreeSolv MAE/RMSE/能量偏差；
MNSol/combined 只可核查公开汇总及其来源哈希，不能在缺少私有逐行数据时独立重算。
`verify_results.py` 不声称重新运行原科学求解器，也不重新验证 MNSol 原始记录。
输入/模型/源代码的详细执行证据仍保留在本地；历史 FP64 源身份已通过修改前归档验证，
不是假称旧源文件在当前 commit 下未变。当前原 Torch 实现提交为 `c0dc21fd`。
