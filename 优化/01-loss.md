[TOC]

# 常用的损失函数

## 1. mean squared error(均方误差)

$$
l_n = (x_n - y_n)^2
$$

## 2. cross entropy loss(交叉熵损失=log对数损失)

### a. BCE(binary cross entropy)

现在有两个类别$a,b$，$q(a)$是模型预测为a类别的概率，$q(b)$是模型预测为b类别的概率；$p(a)$是真实标签a类别的概率，$p(b)$是真实标签为b类别的概率，二分类的交叉熵函数公式如下：
$$
L(p, q) = - (p(a)\log q(a) + p(b)\log q(b))
$$
因为是二分类，自然$q(a)=1-q(b), p(a)=1-p(b)$，得：
$$
L(p, q) = - (p(a)\log q(a) + (1-p(a))\log (1-q(a)))
$$


### b. CE

扩展到多分类的情况，设有$C$个类别，真实值为长度$C$的one-hot向量$y$，用于表示该样本属于那个类别。模型预测值为长度$C$的向量$f(x;\theta)$，$f_c(x;\theta)$表示输出向量的第C维

$$
\begin{aligned}
L(y,f(x;\theta)) &= -\sum_{c=1}^Cy_c\log f_c(x;\theta) \\
&= -\log f_y(x;\theta)
\end{aligned}
$$

因为$y$是one-hot向量，故$L(y,f(x;\theta))=-\log f_y(x;\theta)$，其中$f_y(x;\theta)$可以看作真实类别$y$的似然函数。因此，CE loss也就是负对数似然函数(Negative Log-Likelihood)。**得到$f(x;\theta )$ 的一种方法就是softmax函数**。                                          

## 3.

