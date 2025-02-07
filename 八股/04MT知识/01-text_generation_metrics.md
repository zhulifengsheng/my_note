[TOC]

# BLEU

BLEU的核心是对翻译结果查准率（precision）的度量，它采用的方式是比较并统计在机器译文和参考译文中共同出现的n元词的个数

eg:

- 机器译文：the the the the
- 参考译文：the cat is standing on the ground

以1-gram为例，当前机器译文一共有4个单词，计算每个单词是否出现在参考译文中。因为上述例子中4个单词都是the，所以这4个单词都出现在参考译文中，故BLEU1=4/4=100%。但这显然不合适，因为参考译文中最多只有两个the，所以我要按照下面的公式修正一下，BLEU1=2/4=50%：
$$
Count_{clip} = min(count, max\_ref\_count) \\
p_n = \frac{\sum_{ngram \in 机器译文}Count_{clip}(ngram)}{\sum_{ngram' \in 参考译文}Count(ngram')}
$$
同理，我们会扩展到BLEU2 BLEU3 BLEU4上，将BLEU1-4的查准率取log，然后进行加权求和。
$$
BLEU = BP \times \exp(\sum_{n=1}^Nw_n\log p_n)
$$
现在，还有一个问题是机器译文过短导致的异常高分的BLEU，我们通过BP进行惩罚。

eg:

- 机器译文：of the
- 参考译文：It is the guiding principle which guarantees the military forces always being under the command **of the** Party

如果没有对译文长度的额外判断，BLEU会得到100%的高分。因此BLEU加入了一个乘性的长度惩罚因子，记参考译文长度为r，机器译文长度为c
$$
BP =\left\{
\begin{aligned}
1 && if\space c>r \\
e^{1-\frac{r}{c}} && elsewhere
\end{aligned}
\right.
$$


# METEOR

METEOR认为BLEU有四大欠缺：

1. BLEU没有考虑召回率
2. 高阶的n-gram方法并不足以衡量翻译结果的通顺度，需要更多的衡量方法
3. 直接将翻译结果和参考译文进行匹配过于hard
4. 几何平均值的BLEU没有意义，BLEU1-4有一项为0，BLEU值整体就是0了

第一步：利用外部模块【完全相同、去词根相同、同义词，这三个是有顺序关系的，假如某个单词已经通过之前的匹配机制成功了，后续的匹配机制就不需要匹配了】，找出机器译文和参考译文的单词之间一一对应的关系【一个单词只能匹配一个单词】，如果存在多个匹配关系，取交叉数最少的。

![](meteor1.webp)

![](meteor2.webp)

第二步：计算$F_{mean}$，定义m为机器译文和参考译文匹配到的总对数，机器译文的长度为t，参考译文的长度为r，那么查准率的计算方式为$P=\frac{m}{t}$；召回率的计算方式为$R=\frac{m}{r}$，加权结果表明侧重$R$。
$$
F_{mean} = \frac{10PR}{R+9P}
$$
第三步：块匹配，上述的计算只考虑了1-gram的匹配结果，现在我们加入一个块惩罚。“块”是句子中的连续单词序列，组成块的单词需要满足两个条件：1. 机器译文中每个单词都有匹配 2. 这些单词的顺序和参考译文中的顺序相同。显然，块数越少越好。定义块数为$c$
$$
\begin{aligned}
Penalty &= 0.5 \times (\frac{c}{m})^3 \\
Score &= (1 - Penalty) \times F_{mean}
\end{aligned}
$$

# ChrF

$$
chrF = (1+\beta^2)\frac{chrP \times chrR}{\beta^2chrP + chrR}
$$

chrP是查准率，看机器译文中有多大比例的**字符级**n-gram在参考答案中出现

chrR是查全率，看参考答案中有多大比例的**字符级**n-gram在机器译文中出现

