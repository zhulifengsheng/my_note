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

对于一个二分类任务，我们根据所有样本的预测结果进行排序，按照这个顺序逐个将样本预测为正例，每次都可以计算出TP TN FP TN四个值，以Precision为纵轴、Recall为横轴则是P-R曲线；以$TPR=\frac{TP}{TP+FN}$为纵轴，$FPR=\frac{FP}{TP+FN}$为横轴则是ROC曲线。AUC是ROC曲线 与坐标轴之间的面积

![img](https://upload-images.jianshu.io/upload_images/18391151-c428d6ae7af5a784.png?imageMogr2/auto-orient/strip|imageView2/2/w/367/format/webp)

![img](https://upload-images.jianshu.io/upload_images/18391151-3b0021b47e5d0028.png?imageMogr2/auto-orient/strip|imageView2/2/w/633/format/webp)

## 偏差与方差
