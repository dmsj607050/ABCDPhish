# 网络实现与张量接口

记批大小 B、公共维度 D、HTML 序列长 Lh、视觉 Patch 数 Ls、URL 长 Lu、目标数量 T、每角色查询数 Q。所有可学习模块使用 PyTorch。

## M1：静态证据提取

`extract_evidence` 输出 URL、主机名、可注册域、DOM 节点、节点标签/深度/角色先验及表单提交目标。标题、标题节点和图片属性提供弱表象先验，输入框、表单与按钮提供弱意图先验。二者不是完整的语义或可见性解析器。

`EvidencePreprocessor` 保留来源并转换为张量。URL 采用 UTF-8 字节词表，0 为 padding。轻量 HTML 采用字节序列，预训练 HTML 采用快速分词器及字符偏移匹配 DOM。视觉角色来自可选区域/OCR，未提供时为零。

## M2：三模态细粒度编码

| 编码 | 输出 | 实现 |
|---|---|---|
| URL | `[B,Lu,D]` | Byte embedding + Char-CNN + 位置编码 + Transformer |
| HTML | `[B,Lh,D]` | 字节/ModernBERT 特征 + DOM tag/depth embedding + 结构 Transformer |
| Screenshot | `[B,Ls,D]` | Patch Transformer / SigLIP 2 Patch 特征投影 |

`Ls=(image_size/patch_size)^2`。所有分支保留序列而非仅 CLS。URL 原始特征包含整个地址；身份关系另对 `host=... domain=...` 编码，提交目标使用同样身份表示。身份投影与 URL 编码参数共享。URL 风险由可学习原始 URL 特征保留，不产生未经监督的风险概率。

## M3：角色引导的双查询注意力

HTML 与截图分别有独立查询模块，每个模块包含 `[2,Q,D]` 个可学习查询。两类角色是 A=appearance、I=intent。

```text
attention = masked_softmax(Q_role K^T / sqrt(d_head) + beta * role_prior)
Z_role = mean_queries(LayerNorm(Wo(attention V) + role_query))
```

输出 `roles['html']` 与 `roles['visual']` 均为 `[B,2,D]`，即四组表示 `Z_H^A, Z_H^I, Z_S^A, Z_S^I`。注意力为 `[B,2,Q,L]`，支持追踪关注的 Token/Patch。

弱先验只影响注意力 logits，不是硬掩码，也不是伪标签。不同角色允许共享证据，不强制正交。在没有角色辅助标签的情况下，代码只提供结构归纳偏置，不能据此证明实现了语义解耦。

## M4：四类关系与联合推理

每个成对关系使用 `[x,y,|x-y|,x*y]` 经 MLP 编码。

1. HTML 表象与截图表象。
2. HTML 静态交互意图与截图可见交互意图。
3. URL 域名身份与页面表象；页面表象由可用的 HTML/截图表示聚合。
4. 访问域名与各提交目标；额外输入同主机/同可注册域/目标有效性三个事实，逐目标编码后注意力池化。

四个 `[B,4,D]` 关系 Token 加入类型编码，与 summary Token 一起进入 Relation Transformer。summary 输出关系分支表示；逐关系输出可用于可选监督。缺失关系在注意力中屏蔽；summary Token 保证全部关系缺失时不产生全掩码 NaN，关系分支随后标为不可用。

模态存在与语义确定性分开记录。能计算身份关系不等于已知品牌授权；角色证据不足的关系仍可学习，但其启发式质量较低。M4 主任务输出是关系嵌入，不将相似度或跨域事实直接定义为恶意。

## M5：证据感知自适应融合

原始分支先在 URL/HTML/视觉全局特征间进行掩码注意力池化；解耦分支对四组角色表示池化；关系分支取关系 summary。三分支分别投影到公共 D 维。

```text
gate_input = [all_projected_features, branch_availability, branch_quality]
alpha = masked_softmax(MLP(gate_input))
Z_F = sum_k alpha_k * projected_feature_k
```

门控输入的质量来自模态覆盖度、节点/区域角色先验和目标提取可用性，均是启发式特征，不是校准后的可靠性估计。不可用分支权重严格为零。原始编码特征直接进入 M5，避免 M3/M4 信息压缩导致完全失去原始证据。

## M6：预测及多任务监督

主分类输出 `[B]` 二分类 logit，使用 BCEWithLogits。可选品牌头共享于两个模态的表象表示；意图头共享于两个模态的意图表示；两种跨模态一致性由独立头预测。每项均同时检查标签是否存在及对应模态是否可用。

```text
L = L_cls + lambda_app*L_app + lambda_intent*L_intent
    + lambda_align*L_align + lambda_relation*L_relation
```

未标注项只返回可微零损失。仅有二分类标签时，模型仍可端到端训练，但辅助头不会凭空获得监督。关系评分只有在该类型实际接受过有效关系标签训练时才导出；默认不显示未训练头的随机分数。

解释性输出包含提取证据、来源、缺失掩码、注意力与融合权重；要进一步支持“忠实解释”的论文主张，还需干预、遮挡或反事实评估。对真实效果、角色可分性、校准与合法第三方服务误报，目前没有实验结论。
