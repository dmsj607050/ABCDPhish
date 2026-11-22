# 工程验证记录

验证日期：2026-11-22。环境：Windows、Python 3.10.10、PyTorch 2.5.1+cu121、Transformers 5.15.0，CUDA 可用。

## 自动测试

`python -m unittest discover -s tests -v`：11 项全部通过，包含：

- M2–M6 全链路前向、反向及主要参数非零有限梯度。
- 全部七种非空 URL/HTML/截图组合；全掩码安全处理与门控归一化。
- 缺失模态的占位内容不影响预测。
- 辅助标签缺失或对应模态缺失时不计入损失。
- 可选关系监督参与梯度，缺失关系标签/证据时被屏蔽。
- baseline、disentangled、relations、full 四个增量阶段。
- CUDA 前向与反向。
- 本地随机微型 ModernBERT 与 SigLIP 的 Token/Patch 接口及冻结参数行为。
- 保存/恢复前后预测一致；预训练模式保存本地处理器和骨干配置后恢复。
- 相对 action、base、空 action 与公共/私有后缀域名边界。
- AUROC/AP 的同分数处理及单类别情况。

另外执行了 `python -m compileall -q src scripts tests`。

## 完整命令链

README 中的合成数据生成、两轮 CPU 训练、独立评估、权重恢复与两条带注意力预测导出均执行成功。smoke 配置参数量为 153,648。相同种子重复运行得到相同训练损失记录。

产物位于 `runs/smoke/`，包括最优/末轮权重、配置、训练参数、逐轮日志、评估 JSON 与预测 JSONL。`data/demo/` 与 `runs/` 是本地生成产物，已在 `.gitignore` 排除。

## 验证范围

未下载或执行正式大小 ModernBERT/SigLIP 2 权重的训练；HF 适配测试使用本地随机小模型，验证调用接口与恢复机制。未使用真实钓鱼数据训练、未完成真实消融/跨时间测试、未验证语义解耦和解释忠实度。OCR 脚本需额外 Tesseract 环境，本次未执行实际 OCR。合成数据上的高分不代表真实检测能力。
