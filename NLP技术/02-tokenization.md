[TOC]

# BPE

learn bpe：学习得到单词合并表 codec文件 和 字典文件

第一步：给文本中的单词结尾加上</w>，然后拆分为一个一个的字符

第二步：统计字符对中出现次数最大的字符对，将该字符对加入单词合并表【可以将合并后的子词添加到词表中，被合并词如果出现频率为0则从词表中摘除】，并将文本中所有该字符对合并

第三步：重复第二步，至设定的合并次数或没有频率大于1的字符对了



apply_bpe：先将单词结尾加上</w>并拆分成一个一个的字符，然后根据单词合并表的排序，进行字符合并。

eg：有单词合并表 [u, g] [h, ug]，给一个单词mhug，得到 [UNK, hug]

# WordPiece

learn wordpiece：首先将单词切分为字符，不同于bpe在后面添加</w>，word piece在每个非首字符前面添加##

不同于bpe， wordpiece将$\frac{freq-of-pair}{freq-of-first-element \times freq-of-second-element}$最高分的两个字符合并，即WordPiece选择能够提升语言模型概率最大的相邻子词加入词表。
$$
句子S的LM似然值为 \log P(S) = \sum_{i=1}^{n} \log P(t_i) \\
\log P(t_{xy}) - (\log P(t_x) + \log P(t_y)) = \log \frac{P(t_{xy})}{P(t_x)P(t_y)}
$$

```
1. 
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
2. 
("h" "##u" "##g", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("h" "##u" "##g" "##s", 5)
3. 合并(##g, ##s) -> ##gs
Vocabulary: ["b", "h", "p", "##g", "##n", "##s", "##u", "##gs"]
Corpus: ("h" "##u" "##g", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("h" "##u" "##gs", 5)
4. 合并("h", "##u") -> hu
Vocabulary: ["b", "h", "p", "##g", "##n", "##s", "##u", "##gs", "hu"]
Corpus: ("hu" "##g", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("hu" "##gs", 5)
5. 不断循环至设定的词表大小
```

与BPE不同的是，wordpiece记录的是词表，而不是单词合并表



apply wordpiece：WordPiece finds the longest subword that is in the vocabulary, then splits on it

eg：hugs -> hu + ##gs	hu是词表中最长的子词



如果出现了不存在于词表中的子词，则将整个单词标记为UNK

eg：bum -> b + ##um -> b + ##u + ##m(不存在) -> UNK

# Unigram

与BPE 和 WordPiece不同，Unigram先从一个大词表开始不断减少词表大小至设定的值。

第一步：建立基础的大词表，可以使用语料中的所有字符加上常见的子字符串，也可以使用BPE的词表（此时，频率为0的词不需要清除）

第二步：根据当前的词表计算LOSS，然后计算去掉词表中每一个词，LOSS会增大的数值，选择增大数值最少的子词并去掉它们（去掉百分之p的子词数量，p是自定义的超参数）【注意不要去掉基础字符，以确保每个单词都可以被子词切分】



计算LOSS的方式：

```
语料库：
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
初始词表：
["h", "u", "g", "hu", "ug", "p", "pu", "n", "un", "b", "bu", "s", "hug", "gs", "ugs"]

Unigram language model：
ug的概率为20【ug的数量】/210【子词数量总和】，原因如下：
("h", 15) ("u", 36) ("g", 20) ("hu", 15) ("ug", 20) ("p", 17) ("pu", 17) ("n", 16)
("un", 16) ("b", 4) ("bu", 4) ("s", 5) ("hug", 15) ("gs", 5) ("ugs", 5)
现在，去切分一个给定的单词"pug"，找到所有的子词切分结果
1. P(pug) = p(pu) x p(g) = 5/210 x 20/210 = 0.0022676
2, p(pug) = p(p) x p(u) x p(g) = 5/210 x 36/210 x 20/210 = 0.000389
显然1的分词方法比2的分词方法得到的概率更大，一般地，分的子词越少概率越高，这也很符合我们的直觉，希望一个词被分成的子词越少越好。

回到训练：
每个单词的分词结果如下：
“hug": ["hug"] p(hug) = 15/210
"pug": ["pu", "g"] p(pug) = 17/210 x 20/210
"pun": ["pu", "n"] p(pun) = 17/210 x 16/210
"bun": ["bu", "n"] p(bun) = 4/210 x 16/210
"hugs": ["hug", "s"] p(hugs) = 15/210 x 5/210

LOSS = sum{-log(P(word))}
     = 10[hug单词个数] x （-log(15/210)[负对数似然] + ... + ）
现在我们计算去掉token后对LOSS影响最小的子词，不断循环
```

