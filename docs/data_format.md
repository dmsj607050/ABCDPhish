# 数据与标签协议

## JSONL 样本

每行是一个 JSON 对象。以下为便于阅读的展开示例，保存时应合并成一行：

```json
{
  "id": "sample-001",
  "url": "https://example.test/login",
  "html_path": "html/sample-001.html",
  "screenshot_path": "screenshots/sample-001.png",
  "label": 1,
  "visual_regions": [
    {"box": [0.05, 0.05, 0.55, 0.2], "text": "Example", "role": "appearance", "confidence": 0.9},
    {"box": [0.2, 0.4, 0.8, 0.8], "text": "Sign in", "role": "intent", "confidence": 0.8}
  ],
  "appearance_labels": [2, 2],
  "intent_labels": [[1, 1, 0, 0], [1, -100, 0, 0]],
  "alignment_labels": [1, -100],
  "relation_labels": [-100, -100, -100, -100]
}
```

上例品牌 ID 为 2，使用时需设置 `num_brands>=3`。最小训练样本可以仅有 `url` 和 `label`；完整三模态样本建议提供 URL、HTML 与截图。所有模态均缺失会报错。只含空标签的 HTML 不视为有效文本特征。只有截图时也可推理。

| 字段 | 含义 |
|---|---|
| `id` | 可选样本标识，用于结果追溯 |
| `url` | 原始访问 URL，可缺失 |
| `html_path` / `html` | 本地 HTML 路径或内联字符串；路径优先，可缺失 |
| `screenshot_path` | 本地图片路径，可缺失 |
| `label` | 0=Benign、1=Phishing；训练/评估必需，推理可省略 |
| `visual_regions` | 可选 OCR/人工区域；省略时不虚构视觉角色先验 |
| `appearance_labels` | `[HTML品牌ID, 截图品牌ID]`；未知为 -100 |
| `intent_labels` | `[HTML标签列表, 截图标签列表]`；每项 0/1/-100 |
| `alignment_labels` | `[表象一致性, 交互意图一致性]`；0=不一致、1=一致、-100=未知 |
| `relation_labels` | 四类关系的兼容性标签；0=不兼容、1=兼容、-100=未知 |

默认意图列表顺序为 `login, credentials, payment, download`，可通过 `intent_names` 修改。0 表示有依据的否定，未标注或不可判断应使用 -100。截图中的密码框/按钮只能支持可见交互类型，不能作为实际提交或窃取行为的真值。不同模态标签分别标注，避免把 HTML 隐藏行为强加给截图。

关系顺序固定为：`appearance_consistency`、`intent_consistency`、`brand_domain_identity`、`visit_submission_target`。品牌—域名及提交目标兼容性须来自可信标注或独立证据；不能仅从域名相同/不同生成。默认 `lambda_relation=0`，未提供可靠标签时不训练关系辅助预测头，M4 表示仍接受主分类损失训练。

## 区域坐标与角色

`box` 是相对于原始截图宽高归一化的 `[x0,y0,x1,y1]`，满足 `0<=x0<x1<=1`、`0<=y0<y1<=1`。图像处理采用固定尺寸拉伸，区域按 Patch 重叠面积映射到视觉角色先验。

`role` 支持 `appearance`、`intent`、`both` 或省略；`confidence` 在 [0,1]。省略角色时仅对已知登录/支付/下载词生成弱意图先验，不自动识别品牌。框和文本保留在输出来源中。角色先验是查询偏置，不是标签，也不阻止两个角色关注相同区域。

## 路径和数据采集约定

所有相对文件路径以 JSONL 的父目录为基准，也支持绝对路径。缺失文件会抛出包含记录位置的错误，不会被静默当作缺失模态。HTML 解析只读取静态内容，不运行 JavaScript，不访问页面、表单目标或授权查询服务。建议采集时让 HTML、URL 与截图来自同一次页面快照。

只提取静态表单 `action` 与提交控件 `formaction`；空 action 指向当前访问 URL，相对目标按 `<base>` 或文档 URL 解析。支持关联外部表单的 `formaction`，不分析 JavaScript 动态目标。域名采用本地公共后缀表及私有后缀，域名事实不等于授权判断。

M1 的 DOM 节点上限、Token 上限和目标数量上限会导致截断，角色节点优先保留但不能保证所有证据进入编码器。输出 `targets_truncated` 标记目标截断；关键任务需结合实际长度分布调整配置。训练前应统一品牌映射、确认标签来源并设计独立测试集。
