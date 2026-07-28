# Kimi K3模型简介

Kimi K3是Moonshot AI（Kimi）目前能力最强的开源旗舰模型，也是首个达到 **2.8T（约3T级）参数规模** 的开源模型。模型基于 **Kimi Delta Attention（KDA）** 与 **Attention Residuals（AttnRes）** 构建，采用 **Stable LatentMoE**（896 专家中激活 16 个），原生支持视觉（MoonViT-V2），上下文窗口达 **1M tokens**，面向长程编程、知识工作与推理等前沿智能场景。官方称相较 Kimi K2，整体 scaling 效率约提升 **2.5×**。

模型始终开启 thinking，并通过 `reasoning_effort`（`low` / `high` / `max`，默认 `max`）调节推理强度；多轮对话需回传完整 `reasoning_content`（preserved thinking history）。权重与代码仓库均以 [Kimi K3 License](https://huggingface.co/moonshotai/Kimi-K3) 开源。

## 整体架构

<p style="text-align: center;">
  <img src="kimi_k_3_architecture.jpg" alt="Kimi K3架构图" />
</p>

## 模块说明

主要参数如下（来源：[官方模型卡片](https://huggingface.co/moonshotai/Kimi-K3) / [技术报告](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)）：

| **架构**                     | 混合专家模型（MoE） |
| ---------------------------- | ------------------- |
| **总参数量**                 | 2.8T                |
| **激活参数量**               | 104B                |
| **层数**（含稠密层）         | 93                  |
| **稠密层数量**               | 1                   |
| **注意力层构成**             | 69 KDA + 24 Gated MLA |
| **注意力隐藏维度**           | 7168                |
| **Latent MoE 维度**          | 3584                |
| **MoE隐藏维度**（每个专家） | 3072                |
| **注意力头数**               | 96                  |
| **专家数量**                 | 896                 |
| **每token选择的专家数**    | 16                  |
| **共享专家数量**             | 2                   |
| **词表大小**                 | 160K                |
| **上下文长度**               | 1M（1048576）       |
| **注意力机制**               | KDA & Gated MLA     |
| **残差 / 跨层连接**          | AttnRes             |
| **MoE框架**                  | Stable LatentMoE    |
| **激活函数**                 | SiTU-GLU            |
| **视觉编码器**               | MoonViT-V2          |
| **视觉编码器参数量**         | 401M                |
| **量化**                     | MXFP4 weights / MXFP8 activations（自 SFT 起 QAT） |
| **模态**                     | Text, Image         |

### 语言模块

相对 Kimi K2 / K2.5（MLA + MoE），LLM 部分的主要差异如下：

- 总参数 2.8T / 激活 104B（K2.5：1T / 32B）
- 层数 93，其中仅 1 层稠密层；注意力层按 **69 KDA + 24 Gated MLA** 混合（约 3:1）
- 注意力头数 96（K2.5：64），隐藏维度仍为 7168
- MoE：896 专家、每 token 选 16、共享专家 2；专家 FFN 隐层 3072，并引入 Latent MoE 维度 3584（Stable LatentMoE）
- 激活函数改为 **SiTU-GLU**（K2.5：SwiGLU）
- 跨层引入 **AttnRes**，在深度方向选择性聚合历史层表示
- 路由侧配合 **Quantile Balancing**；训练侧提及 **Per-Head Muon**；自 SFT 起做 MXFP4/MXFP8 量化感知训练

部署上官方推荐 vLLM / SGLang / TokenSpeed，并建议在较大高带宽通信域（如 ≥64 加速器的 supernode）上服务。

### 视觉模块

视觉模块采用 **MoonViT-V2**（约 401M 参数），相对 K2.5 的 MoonViT（约 400M）为同系列升级。模型卡片模态写明 Text / Image；产品与评测侧亦覆盖视频理解（如 Video-MME）。

**视觉与语言融合的关键步骤**（与 Kimi VL / K2.5 路线一致，细节以技术报告与权重实现为准）：

1. **预处理**：由 ImageProcessor 完成抽帧（视频）、尺寸调整、padding，再切分为 **Patch**，得到模型可用的视觉张量。
2. **Patch嵌入**：对 patch 做卷积式嵌入并叠加位置编码（含时间与空间）。
3. **编码**：经多层 Encoder 堆叠，提取高层视觉特征。
4. **Patch池化与merge**：通过 `merge_kernel` 等在空间（及配置下的时间）维度聚合，压缩视觉 token 数量。
5. **MM Projector**：将视觉 hidden 维度映射到与文本一致的 `text_hidden_size`（如 MLP / PatchMerger）。
6. **序列拼接**：按占位符将视觉特征写入文本 embedding 序列，得到最终送入 LLM 的 `inputs_embeds` 与 `attention_mask`。

视觉–语言数据处理流程可参考：[VLM视觉–语言融合流程解析（Kimi K2.5/VL）](https://zhuanlan.zhihu.com/p/2018404307385500510)

## 相关资料

- [技术报告（PDF）](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)
- [官方仓库（MoonshotAI/Kimi-K3）](https://github.com/MoonshotAI/Kimi-K3)
- [模型卡片与权重（Hugging Face）](https://huggingface.co/moonshotai/Kimi-K3)
- [模型卡片与权重（ModelScope）](https://www.modelscope.cn/models/moonshotai/Kimi-K3)
- [整体介绍（官方博客）](https://www.kimi.com/zh-cn/blog/kimi-k3)
- [API 快速开始](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart)
- [Kimi Delta Attention / Kimi Linear](https://github.com/MoonshotAI/Kimi-Linear)
- [Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals)
- [FlashKDA 算子](https://github.com/MoonshotAI/FlashKDA)
