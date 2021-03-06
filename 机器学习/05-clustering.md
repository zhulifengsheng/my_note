[TOC]

# 聚类算法

聚类算法是无监督学习算法，将相似的东西分到一组，**难点在于如何评估模型性能**

## K-means算法

算法流程：

1. 输入样本集合 + K值(分成多少个簇)
2. 从D中随机选择k个样本作为质心
3. repeat：
   1. 令$C_i = \emptyset (i <= i <= k)$【簇集合】，循环全部的样本，找到每个样本距离最近的质心，将该样本加入所在质心的簇集合。
   2. 更新质心为每个簇集合中全部样本的均值，若所有的质心没有被改变则循环结束

缺点：1. K值不好确定 2. 复杂度和样本个数整=正相关 

## DBSCAN

DBSCAN是基于密度的聚类算法

算法流程：西瓜书p213【找到核心对象【领域中的样本数量足够多的样本】$\rightarrow$ 不断发展下线【将核心对象邻域内的样本都加入簇中，如果邻域中某个样本还是核心对象则不断发展下线】$\rightarrow$ 直到下线不再发展，形成一个新簇】

缺点：1. 参数多，调参复杂 2. 可以将噪声点分类出来

## 评价指标

- Inertia指标：每个样本与其质心的距离
- 轮廓系数：计算样本i到同簇其他样本的平均距离$a_i$，$a_i$越小，说明样本i越应该被聚类到该簇；同时，再计算样本i到其他簇$C_j$的所有样本的平均距离$b_{ij}$，$b_i = min\left\{b_{i1}, b_{i2}, ..., b_{ik}\right\}$ 。$S(i)$接近1，说明样本i的聚类合理；接近-1，说明样本i更应该分类到别的簇中

$$
S(i)=\left\{
\begin{aligned}
1-\frac{a(i)}{b(i)} & = & a(i)<b(i) \\
0 & = & a(i)=b(i) \\
\frac{b(i)}{a(i)}-1 & = & a(i)>b(i)
\end{aligned}
\right.
$$

