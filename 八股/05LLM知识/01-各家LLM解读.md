# 大模型发展史

## InstructGPT（2022.01.27）
> https://arxiv.org/abs/2203.02155  
> https://zhuanlan.zhihu.com/p/637419868 (通俗的PPO讲解)

InstructGPT的核心目的：**遵循人类的意图**，因为大语言模型无监督训练方法是 "从互联网上的大量语料库中学习根据上文来预测下一个词"，它做的是个生成任务。而不是 "根据人类的指令生成有帮助，无害的对应答案"。

### RLHF
![rlhf](rlhf.png)

1. **构建一个人类标注的指令遵循数据集，进行SFT**  
作者们聘请了 40 个人的团队来标数据，主要是针对用户提交给 OpenAI API 的问题 (Prompts) 标了一些期望的回复 (human-written answer)，并使用这个标注的数据微调 GPT-3，这一步是个有监督学习 (SFT)。但是值得注意的是，这个 SFT 的数据集是一种对话形式的数据集，带有 Prompt，这种数据集的采集成本很大，所以数据的量级不大。

2. **构建一个奖励模型数据集，训练一个RM**  
给定一个prompt，传给第一步训练的模型，生成多个结果，让人类进行排序。RM学习排序，这个模型的作用是预测人类更喜欢模型的哪个输出（学习人类偏好）。

3. **使用这个RM作为奖励函数，并使用PPO强化学习算法微调第1步训练的模型以最大化这个奖励**  
将第1步训练的 LLM 模型视为策略 (Policy)，第2步训练的 Reward 模型视为环境 (Environment)，采用 PPO 的 RL 方法训练 LLM 模型，得到最终的模型。

## ChatGPT（2022.11.30）
同InstructGPT，可能数据量更大

## LLama（2023.02）
> https://arxiv.org/pdf/2302.13971  

模型结构的改进：
1. RMSNorm  

原Norm公式：  
$\bar{a}_{i}=\frac{a_{i}-\mu}{\sigma}g_{i}， \mu=\frac{1}{n} \sum_{i=1}^{n} a_{i}, \quad \sigma=\sqrt{\frac{1}{n} \sum_{i=1}^{n}\left(a_{i}-\mu\right)^{2}} .$  

RMSNorm公式（**优点**：省去了减均值的操作，减少计算量）：  
$\bar{a}_{i}=\frac{a_{i}}{\operatorname{RMS}(\mathbf{a})} g_{i}, \quad \text { where } \operatorname{RMS}(\mathbf{a})=\sqrt{\frac{1}{n} \sum_{i=1}^{n} a_~{i}^{2}}$

2. SwiGLU激活函数  

SwiGLU是门控线性单元（GLU）的一种，GLU其实不算是一种激活函数，而是一种**神经网络层**。它是一个线性变换后面接门控机制的结构。其中门控机制是一个sigmoid函数用来控制信息能够通过多少。  

$\operatorname{GLU}(x, W, V, b, c)=\sigma(x W+b) \otimes(x V+c)$  
$\operatorname{SwiGLU}(x, W, V, b, c)=\operatorname{Swish}_{1}(x W+b) \otimes(x V+c)$  
$\operatorname{Swish}_{1}(x)=x \sigma(1 x)$  
$\operatorname{ReLU}(x, W, V, b, c)=\max(0, x W+b)\otimes(x V+c)$  

**优点**：SwiGLU相比于ReLU在Transformer架构下能降低约1-2%的困惑度，对ReLU更平滑

3. RoPE(Rotary Position Embedding)旋转位置编码  
> https://zhuanlan.zhihu.com/p/647109286  （旋转位置编码公式的推导）

RoPE不同于绝对位置编码的相加运算，RoPE是将位置编码和query（或key）进行相乘得到的。
![rope](rope.png)
$Q_{i}=X_{i} W^{Q} R_{i}$   
$e_{i, j}=\frac{Q_{i} K_{j}^{T}}{\sqrt{H}}$(原Attention公式)  -> $e_{i, j}=\frac{Q_{i} R_{i-j} K_{j}^{T}}{\sqrt{H}}$(旋转位置编码公式)  
其中左侧的大矩阵$R_m$就是位置m的位置编码，与右侧query向量相乘得到增加了位置信息的query。因此，这种编码不是作用在embedding的输入层，而是作用在与Attention的计算中。

**优点**：1. 有很好的外推性【可以通过旋转矩阵$R$来生成超过预期训练长度的位置编码，提高了模型的泛化能力和鲁棒性】 2. 解决了绝对位置编码（每个位置的位置编码向量都是不一样的）无法实现的：任何位置之间的相对距离在不同长度的句子中应该是一致的【有效地保持位置信息的相对关系】
> 什么是大模型的外推性？  
> 外推性是指大模型在**训练时和预测时的输入长度不一致**，导致模型的泛化能力下降的问题。例如，如果一个模型在训练时只使用了512个 token 的文本，那么在预测时如果输入超过512个 token，模型可能无法正确处理。这就限制了大模型在处理长文本或多轮对话等任务时的效果。

## GPT-4（2023.03.15）
OpenAI没有给出技术细节报告

## LLama2（2023.07.18）
> https://arxiv.org/pdf/2307.09288

模型结构的改进：
1. GQA（分组查询注意力）
![](gqa.jpg)
标注的MHA见图中最左边的示例，每个query head都有对应的key head和value head；而GQA将query head分成了多个组，每个组共享一个key head和value head。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# MHA
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        # 计算每个头的维度大小
        self.head_dim = embed_dim // num_heads

        # 定义Q K V的线性变换层
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        # 最终输出变换
        self.output_linear = nn.Linear(embed_dim, embed_dim)  

    def forward(self, x, mask=None):
        # x.size: batch_size, seq_len, embed_dim
        batch_size, seq_len, embed_dim = x.size()

        # 线性变换得到q k v 然后将embed_dim维度分割成多个头
        # 形状变换: (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, num_heads, head_dim)
        # 转置维度: (batch_size, num_heads, seq_len, head_dim)
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算缩放点积注意力（Scaled Dot-Product Attention），每个头都有独立的注意力分数
        # 注意力分数: (batch_size, num_heads, seq_len, seq_len)
        multihead_attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))

        # 应用掩码（如果需要）
        if mask is not None:
            multihead_attention_scores = multihead_attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # 计算注意力权重
        multihead_attention_weights = F.softmax(multihead_attention_scores, dim=-1)

        # 计算上下文向量
        # (batch_size, num_heads, seq_len, seq_len) x (batch_size, num_heads, seq_len, head_dim) = (batch_size, num_heads, seq_len, head_dim)
        context = torch.matmul(multihead_attention_weights, v)

        # 合并多个头
        # 转置维度: (batch_size, seq_len, num_heads, head_dim)
        context = context.transpose(1, 2).contiguous()
        # 重塑形状: (batch_size, seq_len, embed_dim)
        context = context.view(batch_size, seq_len, embed_dim)

        output = self.output_linear(context)
        return output

x = torch.randn(2, 4, 512)          # batch_size:2 seq_len:4 dim:512
mha = MultiHeadAttention(512, 8)    # 8个头
output = mha(x)

# GQA
class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_groups):
        super(GroupedQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_groups = num_groups

        # 定义线性变换层
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, self.head_dim * num_groups)
        self.v_linear = nn.Linear(embed_dim, self.head_dim * num_groups)
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # k v 重塑形状，得到的是组数，而不是头数
        k = self.k_linear(x).view(batch_size, seq_len, self.num_groups, self.head_dim)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_groups, self.head_dim)

        # 将K和V扩展到与查询头数匹配
        # (batch, seq_len, num_heads(copy了 self.num_heads // self.groups 份), head_dim)
        k = k.repeat_interleave(self.num_heads // self.num_groups, dim=2)  
        v = v.repeat_interleave(self.num_heads // self.num_groups, dim=2)

        # ==========================后面就和正常的MHA一样了===============================#
        # 调整维度顺序以便矩阵乘法
        # (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)  
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算注意力分数
        # (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))
        
        attn_weights = F.softmax(scores, dim=-1)

        # (batch_size, num_heads, seq_len, head_dim)
        context = torch.matmul(attn_weights, v) 

        # 合并多个头
        # 转置维度: (batch_size, seq_len, num_heads, head_dim)
        context = context.transpose(1, 2).contiguous()
        # 重塑形状: (batch_size, seq_len, embed_dim)
        context = context.view(batch_size, seq_len, embed_dim)
        
        output = self.output_linear(context)

        return output
```

GQA的优点：1. 模型性能和MHA几乎相同的同时，**节约 KV-Cache(推理阶段，保存前序token计算过的KV向量) 显存空间，减少了计算量，提升速度**。
> 引申MQA，G=1的GQA注意力机制：大幅降低计算量，但损失性能

## Qwen（2023.08.03）
> https://github.com/QwenLM  
> https://arxiv.org/pdf/2309.16609 (官方技术报告)

### 模型改变
1. Embedding和Output Projection不再共享  
原因：基于初步的实验结果，使用独立的权重矩阵（untied weights）可以**获得更好的性能**。这可能是因为输入嵌入和输出投影在功能上有不同的需求，使用独立的权重可以更好地满足各自的需求。代价是增大了模型的参数

2. RoPE  
矩阵使用的是高精度FP32，提升模型对序列中相对位置关系的理解，提升处理位置敏感任务的性能

3. 在大多数层中移除Bias，但在QKV上保留以提升模型的外推能力  
提升模型的稳定性、减少参数、降低过拟合风险；但在注意力层保持Bias可以增强模型的外推能力，更好地捕获数据特征，提升模型在处理未知数据的表现

4. 使用pre-norm（基本上就是一个共识）  
提升模型的稳定性，缓解梯度消失和梯度爆炸。加快模型的收敛速度

5. RMSNorm  
见LLama

6. SwiGLU  
见LLama，隐藏层大小从4x 变成 8/3x

### 上下文长度扩展
1. NTK 感知插值 & 动态NTK
> https://blog.csdn.net/v_JULY_v/article/details/135072211 （通俗易懂的外推讲解）

仅在推理阶段使用的免训练技术，以扩展模型的上下文长度。

2. LogN-Scaling  
LogN Scaling通过一个取决于上下文长度与训练长度之比的因子重新缩放注意力中Q和V的点积，确保注意力值的熵随着上下文长度的增长而保持稳定。
$\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{\log _{m} n}{\sqrt{d}} Q K^{\top}\right) V$  
其中m是训练长度，n是预测长度。这种技术有助于维持模型在处理长序列时的注意力机制的有效性。

3. window attention  
a. 窗口注意力：将注意力限制在一个上下文窗口内，防止模型关注到太远的内容  
b. 分层注意力：低层注意力窗口更小，高层注意力窗口更大（研究团队观察到Qwen模型在处理长上下文时在不同层次上的建模能力存在差异，较低的层次相对于较高的层次更加敏感于上下文长度的扩展）

## Mixtral（2023.12.11）
@TODO

## Qwen1.5（2024.02.05）
@TODO

## LLama3（2024.04.18）
> https://ai.meta.com/blog/meta-llama-3/  

没有特别多的技术改进

## GPT-4o（2024.05.14）

## Qwen2（2024.06.06）

## o1（2024.09.12）

## Qwen2.5（2024.09.19）

## 混元（2024.11）

## minimax（2025.01）

## deepseek-r1（2025.01.20）

## o3（2025.01.31）

## QwQ（2025.03.06）

# 翻译了各家大模型的技术文章
https://zhuanlan.zhihu.com/p/670574382?utm_psn=1884598268316586416

# 