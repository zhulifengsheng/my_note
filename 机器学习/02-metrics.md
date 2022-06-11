[TOC]

# 模型能力评估指标

## K折验证

将数据集切分为K分，做K次训练，每次训练留出其中不同的一份做验证集。将K次训练得到的结果平均化即为最终的结果。

## ACC

$$
Accuracy = 预测正确 / 样本总数
$$

## Precision Recall F1

|           | 预测1              | 预测0              |
| --------- | ------------------ | ------------------ |
| **真实1** | True Positive(TP)  | False Negative(FN) |
| **真实0** | False Positive(FP) | True Negative(TN)  |

$$
\begin{aligned}
Precision &= \frac{TP}{TP+FP}，预测为正的样本中实际为正的比例 \\
Recall &= \frac{TP}{TP+FN}，真实为正的样本中预测为正的比例 \\
F_\beta &= \frac{(1+\beta^2)\times P \times R}{\beta^2 \times P+R}，\beta > 1时Recall更重要，反之Precision更重要
\end{aligned}
$$

## P-R ROC AUC

对于一个二分类任务，我们根据所有样本的预测结果进行排序，按照这个顺序逐个将样本预测为正例。每次预测正例数目加一，每次都可以计算出TP TN FP TN四个值，以Precision为纵轴、Recall为横轴则是P-R曲线；以$TPR=\frac{TP}{TP+FN}$为纵轴，$FPR=\frac{FP}{TP+FN}$为横轴则是ROC曲线。AUC是ROC曲线 与坐标轴之间的面积

![img](https://upload-images.jianshu.io/upload_images/18391151-c428d6ae7af5a784.png?imageMogr2/auto-orient/strip|imageView2/2/w/367/format/webp)

![img](https://upload-images.jianshu.io/upload_images/18391151-3b0021b47e5d0028.png?imageMogr2/auto-orient/strip|imageView2/2/w/633/format/webp)

## 偏差与方差

给定一堆来自同一分布的训练集，他们的样本数量是一样的。假设训练集有5份，我们使用这5份训练集$D_i$训练5个模型。

对于一个测试集中的样本，5个模型预测的结果算均值得到$\bar f(x)$。

方差定义：不同训练集训练的模型预测结果$-\bar f(x)$ 的均值，**方差反映了不同数据对训练结果的干扰情况**
$$
var(x) = \frac{\sum_{i=1}^{5}(f(x;D_i) - \bar f(x))^2}{5}
$$
偏差定义：$\bar f(x)$与真实结果之间的差异，**偏差反映了学习算法的期望预测与真实结果的偏离程度**
$$
bias(x) = \bar f(x) - y
$$
泛化误差 = 偏差^2 + 方差 + 误差（公式推导过程省略） 

![img](https://pic1.zhimg.com/80/v2-7f56516f55463656e81d55edc5c069e8_1440w.jpg)

欠拟合阶段：模型拟合能力差，偏差大；模型训练不充分，训练数据不同而导致的差异没有被模型捕捉，方差小

过拟合阶段：模型拟合能力强，偏差小；但模型过分训练，捕捉到了不同训练数据其各自的局部规律，导致每个模型的预测结果差距大，方差大

例如样本真实值为4.4：模型前期预测结果为[3.1, 3.2, 3.3, 3.4, 3.5]  -> 中期为 [4, 4.2, 4.4, 4.6, 4.7] -> 后期为[2, 3, 4.4, 5.6, 7]
