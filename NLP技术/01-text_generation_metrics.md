[TOC]

# BLEU

BLEU的核心是对翻译结果查准率（precision）的度量，它采用的方式是比较并统计共同出现的n元词的个数

eg:

- 机器译文： the the the the
- 参考译文：the cat is standing on the ground

1-gram：机器译文一共有4个单词，计算每个单词是否出现在参考译文中，因为上述例子中4个单词都是the，所以这4个单词都出现再来参考译文中，BLEU1=4/4=100%。但这显然不合适，因为参考译文中最多只有两个the，所以我要按照下面的公式修正一下，BLEU1=2/4=50%：
$$
Count_{clip} = min(count, max\_ref\_count) \\
p_n = \frac{\sum_{ngram \in C}Count_{clip}(ngram)}{\sum_{ngram' \in C'}Count(ngram')}
$$
同理，我们会扩展到BLEU2 BLEU3 BLEU4上，将BLEU1-4的查准率取log，然后进行加权求和。
$$
BLEU = BP \times \exp(\sum_{n=1}^Nw_n\log p_n)
$$
还有一个问题是机器译文过短导致的异常高分的BLEU

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
3. 没有直接将翻译结果和参考译文进行匹配，n-gram会将一些常用语（the and）也匹配上，这些常用词没什么意义
4. 几何平均值的BLEU没有意义，BLEU1-4有一项为0，BLEU值整体就是0了
