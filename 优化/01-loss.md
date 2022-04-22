[TOC]

# 常用的损失函数

## 1. mean squared error(均方误差)

$$
l_n = (x_n - y_n)^2
$$

## 2. cross entropy loss(交叉熵损失=log对数损失)

​	CE是用于衡量两个分布之间的距离，在了解CE之前，我们先了解一下数理统计中的**似然函数**。似然函数是关于模型参数的函数，给定输出$y$，关于参数$\theta$的似然函数在数值上等于给定参数$\theta$，输出变量$y$的概率，即$L{ \left(\theta \left|y \right. \right) }=P{ \left(Y=y|\theta \right) }$。考虑下式中的一个情况，
$$
若L(\theta_1|y) = P_{\theta_1}(Y=y) > P_{\theta_2}(Y=y) = L(\theta_2|y)
$$
那么，似然函数就反应出这样一个朴素推测：在参数$\theta_1$下，随机变量$Y$取到值$y$的**可能性大于** 在参数$\theta_2$下，随机变量$Y$取到值$y$的**可能性**。换句话说，我们更有理由相信$\theta_1$(相对于$\theta_2$来说)是真实的。

​	CE loss一般用于二分类、多分类问题，设有$C$个类别，真实值为长度$C$的向量$y$，模型预测值为长度$C$的向量$f(x;\theta)$
$$
\begin{aligned}
L(y,f(x;\theta)) &= -ylogf(x;\theta) \\
&= - \sum_{c=1}^Cy_clogf_c(x;\theta)
\end{aligned}
$$

## 3.