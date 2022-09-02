[TOC]

# 梯度下降(gradient descent)

在对模型参数进行求解时，往往不能直接求解。我们可以根据loss寻找到一个让loss递减的方向，然后慢慢逼近最优解。这个让loss递减的方向就是loss关于模型参数的导数（梯度）的负方向。

而如果遍历完整个数据集，再计算loss求梯度优化参数会导致训练很慢、成本很高；如果对每个样本都算loss求梯度优化参数又会导致模型参数更新不准确，因为单个样本不能反应整个数据集的性质。所以我们折中，每次计算更新时随机抽取一小批(batch)进行**小批量随机梯度下降**(minibatch stochastic gradient descent)
$$
(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b).
$$

缺点：

1. SGD容易收敛到局部最优，并且在某些情况下可能被困在鞍点
2. 有时我们想对一些参数更新快一些，对其他参数更新慢一些，这时候SGD就不太能满足要求了



## Momentum

momentum是模拟物理里动量的概念，积累之前的动量来代替真正的梯度。
$$
\begin{aligned}
m_0 &= 0 \\
m_t &= \mu * m_{t-1} + g_t(梯度) \\
\theta_t &= \theta_{t-1} -\eta * m_t(参数更新)
\end{aligned}
$$
特点：

1. 初期时，使用上一次参数更新的结果，可以使得下降方向保持一致，从而很好地加速收敛
2. 中后期，在局部最小值来回震荡时，gradient趋近于0，$\mu$使得更新幅度不为0，可以让模型跳出局部最优

总而言之，momentum可以抑制振荡，加速收敛



## Adagrad

自适应修正学习率

$$
\begin{aligned}
r_t &= r_{t-1} + g_t^2 \\
\theta_t &= \theta_{t-1} - \frac{\eta}{\sqrt{r_t+\sigma}}*g_t
\end{aligned}
$$

特点：

1. 在梯度大的方向上约束梯度，使步长变小；在梯度小的方向上放大梯度，使步长变大

缺点：

1. 中后期，分母上的$r_t$会越来越大，使得g趋近于0，让训练过早地停止



## RMSprop

Adagrad中$r_t$是单调递增的，这使得学习率会递减至0。RMSprop为了改进这个缺点，在计算二阶动量时不累积全部历史梯度，而只关注最近某一时间窗口内的梯度。
$$
\begin{aligned}
r_t &= \beta r_{t-1} +(1-\beta)g_t^2 \\
\theta_t &= \theta_{t-1} - \frac{\eta}{\sqrt{r_t+\sigma}}*g_t
\end{aligned}
$$



## Adam

Adam做了很多的平滑操作，所以Adam对学习率非常不敏感
$$
\begin{aligned}
一阶矩估计，m_t &= \mu * m_{t-1} + (1-\mu)*g_t \\
二阶矩估计，r_t &= \beta * r_{t-1} + (1-\beta)*g_t^2 \\
修正(对期望的无偏估计)，\hat m_t &= \frac{m_t}{1-\mu} \\
\hat r_t &= \frac{r_t}{1-\beta} \\
\theta_t &= \theta_{t-1} - \eta\frac{\hat m_t}{\sqrt{\hat r_t+\sigma}}
\end{aligned}
$$
特点：

1. 融合了动量法和自适应方法
2. 修正操作避免了冷启动（因为$m_0,r_0=0$，修正的这个除法可以防止开始时的梯度很小）
