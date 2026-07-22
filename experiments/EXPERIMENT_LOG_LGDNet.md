# LGDNet Experiment Log

This file records repository-level LGDNet changes, releases, training runs, test runs, threshold searches, dataset splits, and ablations. It is intentionally concise for the public LCA-EQ release.

## GitHub Paper Consistency Audit and Fixes (2026-07-21)

实验名：
GitHub Paper Consistency Audit and Fixes

配置文件：
`configs/lgdnet_lca_eq.yaml`

模型文件：
`src/lgdnet/models/lsgm.py`; `src/lgdnet/models/lgdnet.py`

数据 split：
唯一论文事件级划分 `data/lca_eq/splits/paper_scene_split/{train,val,test}.csv`

核心改动：
对照论文检查 GitHub 仓库。按论文数据发布边界，从 Git 跟踪中删除 `data/lca_eq/*/images/*.png`，保留 land masks、damage masks、events.csv 和 split CSV；`.gitignore` 新增 `/data/lca_eq/*/images/` 防止误传原始 RGB 影像。README 和 data README 明确说明原始 Google Earth RGB pixels 不在 GitHub 中重新分发，用户需要在合法访问源影像后按 split 相对路径恢复 1024 x 1024 patches。LSGM 实现更新为 `ZD=phi(FD)`、L2-normalized prototype affinity、`Fout = FD * (1 + lambdaS * AS - lambdaU * AU)`。

启动命令：
未启动训练；仅执行论文一致性检查、数据发布边界修正和 LSGM 公式修正。

GPU：
未使用 GPU。

训练轮数：
0

monitor：
未训练。

best checkpoint：
不随 Git 仓库发布；建议通过 GitHub Release asset、Git LFS 或外部归档存储发布。

best val 指标：
未重新训练；历史开发记录中的 best `val_combined_F1=0.8286` 仅作为内部参考，不作为本次 release 新实验结果。

test 指标：
未重新测试。

阈值搜索结果：
未重新搜索；历史推荐 damage threshold 为 `0.75`。

结论：
仓库现在与论文在数据集命名、7 类地物标签、二分类损毁标签、事件级 split、post-event-only 输入、support/non-support 分组和 LSGM 残差调制公式上保持一致。当前 `src` 按 official PyTorch release implementation 维护，README 记录论文训练设置、数据发布内容和 Zenodo 归档链接。

下一步：
Zenodo draft publish 后，应将最终 Dataset DOI 和 Model DOI 回填 README。

## GitHub Minimal Repository Cleanup (2026-07-21)

实验名：
GitHub Minimal Repository Cleanup

配置文件：
`configs/lgdnet_lca_eq.yaml`

模型文件：
`src/lgdnet/models/lgdnet.py`; `src/lgdnet/models/lsgm.py`

数据 split：
仅保留 `data/lca_eq/splits/paper_scene_split/{train,val,test}.csv`

核心改动：
按用户要求将 GitHub 仓库压缩为最关键、最基本内容；删除协作模板、额外文档、figure tiles、大型参考代码和额外画图脚本。README 改为最小论文 release 说明，并保留训练配置、模型代码、数据标签、事件元数据、split 和基础测试。

启动命令：
未启动训练；仅执行仓库清理和文档更新。

GPU：
未使用 GPU。

训练轮数：
0

monitor：
未训练。

best checkpoint：
不随 Git 仓库发布；建议通过 GitHub Release asset、Git LFS 或外部归档存储发布。

best val 指标：
未重新训练；历史开发记录中的 best `val_combined_F1=0.8286` 仅作为内部参考，不作为本次 release 新实验结果。

test 指标：
未重新测试。

阈值搜索结果：
未重新搜索；历史推荐 damage threshold 为 `0.75`。

结论：
GitHub 仓库已最小化为代码、配置、LCA-EQ release assets、测试、许可证和必要实验记录。因项目操作要求，`experiments/EXPERIMENT_LOG_LGDNet.md` 保留。

下一步：
如需进一步减小仓库体积，应将 `data/lca_eq` 标签资产迁移到 Release/HuggingFace/Zenodo，仓库仅保留下载脚本和 split manifest。

## GitHub Paper-Consistency Blocking Fix (2026-07-22)

实验名：
GitHub Paper-Consistency Blocking Fix

配置文件：
`configs/lgdnet_lca_eq.yaml`

模型文件：
`src/lgdnet/models/lgdnet.py`; `src/lgdnet/models/lsgm.py`

数据 split：
唯一论文事件级划分 `data/lca_eq/splits/paper_scene_split/{train,val,test}.csv`

核心改动：
修复公开 GitHub 仓库与论文描述的投稿阻断矛盾：README 改为 official PyTorch implementation；环境固定到 PyTorch 2.7.1 / torchvision 0.22.1；默认训练设置改为 30 epochs 和 384 x 384 crop；默认 LGDNet loss 改为二任务目标 `L = lambda_land * L_land + lambda_damage * L_damage`，移除默认 auxiliary damage loss；新增 `scripts/preprocess_lca_eq.py`、`scripts/evaluate_lgdnet.py` 和 `data/lca_eq/aoi/` metadata；README 增加 Zenodo dataset/model draft 链接。

启动命令：
未启动训练；执行仓库一致性修复、静态检查和单元测试。

GPU：
未使用 GPU。

训练轮数：
0；本次不训练。

monitor：
`val_combined_F1` 保持为论文默认 monitor。

best checkpoint：
Zenodo model draft 对应 `LGDNet-urbanssf-l-final8-alltrain-paperlsgm-bhost-os4-c05-gpu4-e30-best-epoch=28-val_combined_F1=0.8286.ckpt`

best val 指标：
沿用已记录最优 `val_combined_F1=0.8286`；本次未重新训练。

test 指标：
本次未重新测试。

阈值搜索结果：
推荐 damage threshold `0.75`。

结论：
公开仓库已消除 PyTorch 版本、epoch、crop size、额外损失项、数据发布内容和仓库定位表述上的主要论文矛盾。

下一步：
Zenodo draft publish 后将最终 Dataset DOI 和 Model DOI 回填 README；Zenodo draft publish 后，应继续将最终 DOI 回填 README 和论文数据可用性声明。


## Zenodo DOI Publication and README Update (2026-07-22)

实验名：
Zenodo DOI Publication and README Update

配置文件：
`configs/lgdnet_lca_eq.yaml`

模型文件：
Zenodo Model DOI `https://doi.org/10.5281/zenodo.21479452`；best checkpoint SHA256 `754a9d0c7a259c37e4b832dacafae2df72b72693f97707a2c3a101c779530ea9`

数据 split：
论文事件级划分 `data/lca_eq/splits/paper_scene_split/{train,val,test}.csv`；Zenodo Dataset DOI `https://doi.org/10.5281/zenodo.21472884`

核心改动：
确认 Zenodo dataset/model draft 已 publish，并将 GitHub README 中的 draft deposit links 更新为正式 DOI links。

启动命令：
未启动训练；仅执行 Zenodo publish 状态确认和 README DOI 回填。

GPU：
未使用 GPU。

训练轮数：
0

monitor：
不涉及；历史 best monitor 为 `val_combined_F1`。

best checkpoint：
`https://doi.org/10.5281/zenodo.21479452` 中的 LGDNet best model package。

best val 指标：
沿用已记录最优 `val_combined_F1=0.8286`。

test 指标：
本次未重新测试。

阈值搜索结果：
推荐 damage threshold `0.75`。

结论：
GitHub README 已引用正式 Zenodo DOI，数据集和模型归档链路完成。

下一步：
将 Dataset DOI 和 Model DOI 同步写入论文 Data Availability / Code Availability 部分。
