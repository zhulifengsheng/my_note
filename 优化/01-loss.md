[TOC]

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