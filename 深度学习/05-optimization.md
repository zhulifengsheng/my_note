[TOC]

# 梯度下降(gradient descent)

在对模型参数进行求解时，往往不能直接求解。我们可以根据loss寻找到一个让loss递减的方向，然后慢慢逼近最优解。这个让loss递减的方向就是loss关于模型参数的导数（梯度）的负方向。

而如果遍历完整个数据集，再计算loss求梯度优化参数会导致训练很慢、成本很高；如果对每个样本都算loss求梯度优化参数又会导致模型参数更新不准确，因为单个样本不能反应整个数据集的性质。所以我们折中，每次计算更新时随机抽取一小批(batch)进行**小批量随机梯度下降**(minibatch stochastic gradient descent)
$$
(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b).
$$

## Momentum 是什么?

