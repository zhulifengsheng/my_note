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

**优点**：SwiGLU相比于ReLU在Transformer架构下能降低约1-2%的困惑度

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

## LLama2（2023.07.18）

## Qwen（2023.08.03）
> https://github.com/QwenLM

## Qwen1.5（2024.02.05）

## LLama3（2024.04.18）

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