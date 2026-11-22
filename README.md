# AIDR-Net

Appearance–Intent Disentanglement and Relational Reasoning Network：面向 URL、静态 HTML 与网页截图的三模态钓鱼检测研究实现。

工程实现了 M1–M6、可选辅助监督、逐阶段消融、训练、独立评估与带证据预测。默认轻量编码器从零初始化，便于离线验证；研究配置接入 ModernBERT 与固定分辨率 SigLIP 2。尚未使用真实数据训练，不能将工程测试或合成数据结果视为有效性证明。

## 目录与模块职责

```text
ABCDPhish/
├── configs/
│   ├── smoke.json                    # 小尺寸端到端检查
│   ├── lightweight.json              # 从零训练的轻量完整网络
│   └── pretrained.json               # ModernBERT + SigLIP 2 研究配置
├── src/aidr_net/
│   ├── config.py                     # 配置定义、校验与读写
│   ├── model.py                      # 六模块编排与原始特征旁路
│   ├── data/
│   │   ├── static_evidence.py         # M1：URL、DOM、表单目标与来源
│   │   ├── preprocessing.py           # 分词、结构/角色特征、图像/区域映射
│   │   └── dataset.py                 # JSONL 数据集、批处理、设备搬移
│   ├── modules/
│   │   ├── trimodal_encoders.py       # M2：三模态细粒度编码
│   │   ├── semantic_disentanglement.py# M3：角色引导双查询注意力
│   │   ├── relational_reasoning.py    # M4：四类关系与 Relation Transformer
│   │   ├── evidence_fusion.py         # M5：证据感知门控融合
│   │   ├── prediction_heads.py        # M6：主分类与辅助预测头
│   │   └── common.py                  # 掩码安全的注意力、池化、Transformer
│   ├── training/
│   │   ├── losses.py                  # M6：缺失标签掩码与多任务损失
│   │   ├── metrics.py                 # F1、FPR、AUROC、AP 等指标
│   │   └── train.py                   # 训练、验证、日志、最优权重保存
│   ├── evaluation.py                  # 独立数据集评估入口
│   └── inference.py                   # 批量预测与证据/注意力导出
├── scripts/
│   ├── create_demo_data.py            # 生成本地合成数据
│   └── enrich_ocr.py                  # 可选 Tesseract OCR 适配
├── tests/
│   ├── test_network.py                # 梯度、缺失证据、损失、消融和恢复测试
│   └── test_pretrained_adapters.py    # 本地微型 HF 模型接口测试
└── docs/
    ├── architecture.md               # 数据流、张量形状与建模边界
    ├── data_format.md                # 数据与辅助标签协议
    └── verification.md               # 已执行验证及其适用范围
```

## 安装

在工程根目录执行，建议使用独立虚拟环境。PyTorch 的 CPU/CUDA 版本请按机器环境安装。

```powershell
python -m pip install -e .
```

当前验证环境：Python 3.10.10、PyTorch 2.5.1+cu121、Transformers 5.15.0。依赖范围中其他版本未逐一测试。Windows 入口推荐使用下述 `python -m ...`，无需将用户 Scripts 目录加入 PATH。

## 快速跑通完整链路

```powershell
python scripts/create_demo_data.py --output data/demo
python -m aidr_net.training.train --config configs/smoke.json --train data/demo/train.jsonl --val data/demo/val.jsonl --output runs/smoke --epochs 2 --batch-size 2 --device cpu
python -m aidr_net.evaluation --checkpoint runs/smoke/best.pt --manifest data/demo/val.jsonl --output runs/smoke/evaluation.json --device cpu
python -m aidr_net.inference --checkpoint runs/smoke/best.pt --manifest data/demo/predict.jsonl --output runs/smoke/predictions.jsonl --attention --device cpu
python -m unittest discover -s tests -v
```

合成页面仅用于程序检查，样本数很少、模式非常简单。其分数不具有研究结论意义。

训练产物包含 `config.json`、`run_arguments.json`、`metrics.jsonl`、`best.pt` 与 `last.pt`。最优模型按验证集总损失选择。预训练模式还会保存 `assets/html/` 分词器、`assets/visual/` 图像处理器；checkpoint 包含完整模型参数及骨干配置。迁移推理时一起复制权重和 `assets/`，无需重新下载骨干权重。当前 checkpoint 用于恢复推理，不包含优化器或断点续训状态。

## 使用真实数据

先按 [数据协议](docs/data_format.md) 准备独立的训练、验证和测试 JSONL。路径相对于 JSONL 所在目录解析。正式实验应按时间、页面来源及域名/站点簇设计切分，处理近重复页面；程序只自动拒绝训练/验证中完全相同的 URL，并未替代数据泄漏审计。

```powershell
python -m aidr_net.training.train --config configs/pretrained.json --train data/train.jsonl --val data/val.jsonl --output runs/pretrained --epochs 10 --batch-size 2 --device cuda
python -m aidr_net.evaluation --checkpoint runs/pretrained/best.pt --manifest data/test.jsonl --output runs/pretrained/test_metrics.json --device cuda
```

预训练配置首次运行需要下载权重，或将两个 `*_model_name` 改为本地目录并设置 `local_files_only=true`。默认冻结预训练骨干，训练投影层、DOM 结构编码及 M3–M6；将 `freeze_backbones` 改为 `false` 可进行全量微调。显存消耗由批大小、HTML 长度和是否微调共同决定。

仅支持固定分辨率 `google/siglip2-base-patch16-224` 这一类 SigLIP 2 视觉接口；不支持 NaFlex。`image_size`、`patch_size` 必须匹配权重。HTML 和图像接口分别依据 [ModernBERT 官方文档](https://huggingface.co/docs/transformers/model_doc/modernbert) 与 [SigLIP 2 官方文档](https://huggingface.co/docs/transformers/model_doc/siglip2)。

## 辅助任务与消融

默认只要求钓鱼二分类标签。缺失的品牌、意图、一致性与关系标签均为 `-100`，对应损失不参与训练。品牌任务需配置 `num_brands` 和训练集确定的品牌 ID 表；HTML 与截图标签分别提供，允许不一致。不会将节点规则、跨域提交或二分类标签自动转换为辅助真值。

修改配置的 `stage` 可逐步验证：

| stage | 模型 |
|---|---|
| `baseline` | M2 三模态特征 → 掩码注意力池化 → MLP |
| `disentangled` | 原始特征与 M3 表示等权融合 → MLP |
| `relations` | 原始特征、M3 和 M4 表示等权融合 → MLP |
| `full` | 三类特征经 M5 证据感知门控 → MLP |

这些阶段是工程中提供的增量对照。正式论文仍需增加拼接、普通跨注意力等控制模型，并验证角色分工与证据解释的可靠性。

## OCR 与证据输出

无 OCR 时，视觉编码器与角色查询仍正常工作，视觉角色先验为零，表示未提供角色证据。可以直接提供 `visual_regions`，也可以先用本地 Tesseract 生成 OCR 区域：

```powershell
python -m pip install -e ".[ocr]"
python scripts/enrich_ocr.py --manifest data/train.jsonl --output data/train_ocr.jsonl --lang eng
```

另需安装 Tesseract 可执行程序；若不在 PATH，传入 `--tesseract-cmd`。中文可用 `--lang eng+chi_sim`，需要相应语言包。OCR 词框仅生成弱角色先验，不生成品牌真值。

预测 JSONL 包含原始证据、目标地址及提取来源、模态/关系可用性、融合权重与目标池化权重。`--attention` 增加 HTML Token、截图 Patch 的双角色查询注意力。未接受有效关系标签监督的关系预测头不输出评分。权重、启发式证据质量与注意力均不是经过校准的可信度，也不构成因果解释。
