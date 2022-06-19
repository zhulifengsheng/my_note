[TOC]

# 任务

1. 有监督学习
2. 无监督学习：
   1. 生成式学习：自编码器（Auto-Encoder），主要用于数据的降维或者特征的抽取
   2. 对比式学习：对比学习
3. 半监督学习：有一些数据有标记，给没有标记的数据指明了方向

# 无监督学习

## 自编码器

结构：1. encoder：它可以把原先的图像压缩成更低维度的向量。 2. decoder：它可以把压缩后的向量还原成图像，通常它们使用的都是神经网络。

![在这里插入图片描述](01-autoencoder1.png)

**将vector用于下游任务**

应用：

1. Text retrieval（文字检索）：将一个document从bag of word特征降维到2维
2. Image retrieval（图片检索）

模型发展历史：

1. de-nosing auto-encoder

![image-20220619160512551](01-autoencoder2.png)

2. feature disentangle

   encoder学习（压缩）得到的特征会包含很多的信息，比如对一段声音进行压缩就会包含语音内容和发言者音色、音调等信息。那有没有一种方法可以将这些信息区分开呢？比如，下图将content和speaker的信息区分开。

   如果可以成功地将声音信号encode的向量中内容和音色区分开，那个给两段声音，就可以做到变声的效果。

   ![image-20220619160912561](01-autoencoder3.png)



## 对比学习

