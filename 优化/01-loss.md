[TOC]

# 机器学习的各种任务

有监督学习

- 回归任务：给一个样本算出一个值

- 二分类，多分类任务：给一个样本分到一个类别中

- 多标签分类任务：给一个样本打上多个标签

无监督学习

- 聚类：将相似的样本聚到一个类别中
- 主成分分析(PCA)：能否找到少量的参数来准确地捕捉数据的线性相关属性
- 概率图模型：能否发现数据之间的关系
- 生成对抗网络(GAN)：为我们提供一种合成数据的方法


# 损失函数与极大似然估计

我们先了解一下数理统计中的**似然函数**。似然函数是关于模型参数的函数，给定输出$y$，关于参数$\theta$的似然函数在数值上等于给定参数$\theta$，输出变量$y$的概率，即$L{ \left(\theta \left|y \right. \right) }=P{ \left(Y=y|\theta \right) }$。考虑下式中的一个情况，
$$
若L(\theta_1|y) = P_{\theta_1}(Y=y) > P_{\theta_2}(Y=y) = L(\theta_2|y)
$$
那么，似然函数就反应出这样一个朴素推测：在参数$\theta_1$下，随机变量$Y$取到值$y$的**可能性大于**在参数$\theta_2$下，随机变量$Y$取到值$y$的**可能性**。换句话说，我们更有理由相信：相对于$\theta_2$，$\theta_1$是真实的。

对于一个数据集，假设每个样本$x_i$独立，似然函数为：
$$
likelihood(\theta) = f(X|\theta) = \prod_{i=1}^Nf(x_i|\theta)
$$
取对数似然：
$$
likelihood(\theta) = logf(X|\theta) = \sum_{i=1}^Nlogf(x_i|\theta)
$$
如果一个损失函数可以使似然函数最大，那么就等同于用极大似然估计的方法来求解模型参数$\theta$

# 常用的损失函数

## 1. mean squared error(均方误差)

$$
l_n = (x_n - y_n)^2
$$

## 2. cross entropy loss(交叉熵损失=log对数损失)

​	CE是用于衡量两个分布之间的距离，在了解CE之前，

​	CE loss一般用于二分类、多分类问题，设有$C$个类别，真实值为长度$C$的向量$y$，模型预测值为长度$C$的向量$f(x;\theta)$，$f_c(x;\theta)$表示输出向量的第C维
$$
\begin{aligned}
L(y,f(x;\theta)) &= -ylogf(x;\theta) \\
&= - \sum_{c=1}^Cy_clogf_c(x;\theta)
\end{aligned}
$$

​	因为$y$是one-hot向量，公式$(6)$也可以写作$L(y,f(x;\theta))=-logf_y(x;\theta)$，其中$f_y(x;\theta)$可以看作真实类别$y$的似然函数。因此，CE loss也就是负对数似然函数(Negative Log-Likelihood)

## 3.