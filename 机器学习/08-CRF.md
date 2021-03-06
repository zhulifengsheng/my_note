[TOC]

# 概率图模型

在监督学习下，模型可以分为判别式模型与生成式模型

## 判别式模型

模型根据样本的输入feature和目标label，直接去学习一个复杂的映射，直接建模$P(Y|X)$【LR会直接输出分类的概率】

## 生成式模型

这类模型会先从训练数据中，将所有的数据分布情况摸透，然后最终确定一个分布$P(X,Y)$来作为所有样本的分布。给了一个新样本X后，通过条件概率计算出Y：$P(Y|X)=\frac{P(X,Y)}{P(X)}$

生成式模型在训练的过程中，只是对$P(X,Y)$建模，我们需要确定得到这个联合概率分布的所有参数。

## 隐马尔可夫模型（生成式模型）

![](hmm.png)

首先，明确HMM的5要素：【HMM做了一个观测独立假设，**认为$o_i$之间是没有关系的**】

1. 隐状态$i$，我们的隐状态节点不能随意取，只能从限定的**隐藏状态集**$\left\{s_1,...,s_N \right\}$中取值，共N个
2. 观测状态$o$，我们的观测状态节点不能随意取，也只能从限定的**观测状态集**$\left\{g_1,...,g_M \right\}$中取值，共M个
3. 矩阵A，状态转移矩阵$A=[a_{ij}]_{N \times N}$，矩阵中每个元素都是一个概率$P(s_j|s_i)$，表示若当前时刻为$s_i$下一时刻为$s_j$的概率
4. 矩阵B，观测概率矩阵$B=[b_{ij}]_{N\times M}$，矩阵中每个元素表示根据当前状态获得各个观测值的概率$P(g_j|s_i)$
5. 初始状态概率，模型在初始时刻下各状态出现的概率，通常记为$\pi = \left(\pi_1, \pi_2,...,\pi_N\right)$，其中$\pi$表示模型的初始状态$i_1$为$s_i$的概率

HMM是一个有向无环图，在实际的任务中，隐状态可以是文字$i$，观测状态是语音$o$。模型训练完成后，我们可以根据语音推测文字。
$$
\begin{aligned}
\lambda &= (\pi, A, B)\\
P(o,i|\lambda) &= \prod_{t=1}^{T}P(o_t,i_t|\lambda) \\
&= \prod_{t=1}^{T}P(o_t|i_t,\lambda)p(i_t|i_t-1,\lambda) \\
P(i|o) &= \frac{P(i, o)}{P(o)}
\end{aligned}
$$

## Conditional Random Field（判别式模型）

![](crf1.png)

与HMM的区别：无向图（打破观测独立假设） + 判别式模型

序列问题主要关注于链式条件随机场，在链式条件随机场中，相邻的每两个节点组成一个最大团，用一个$F$函数做能量函数。$F$函数可以进一步被拆分为最大团的转移函数$t$和当前时刻的状态函数$s$
$$
\begin{aligned}
P(Y|X)&=\frac{1}{Z}\exp\left(\sum_{t=1}^{T}F(y_{t-1}, y_t, X)\right) \\
&=\frac{1}{Z}\exp\left(\sum_j\sum_{t=1}^{T}\lambda_jt_j(y_{t-1}, y_t, X) + \sum_k\sum_{t=1}^{T}\mu_ks_k(y_t,X)\right)
\end{aligned}
$$
举例一个词性标注问题：

LSTM会得到每个词预测为每个词性的概率，CRF会通过转移函数和状态函数计算出整个句子（将每个词的标注综合判断）概率最大的那个结果。

LSTM+CRF的模型就是在学习LSTM和CRF的转移函数与状态函数。