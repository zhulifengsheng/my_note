[TOC]

# 权重衰减

## 1. 什么是权重衰减(=L2正则化=岭回归)

权重衰减（weight decay）是防止过拟合的一种方法，通过限制模型参数的值的范围来控制模型容量，也通常被称为$L_{2}$正则化。

防止过拟合的一种方法首先想到的一种就是减少参数，而减少参数也可以理解为让参数有尽可能多的0（dropout也是类似的一种思想）。这里，我们可以使用0范数来度量这个指标。于是，我们原本的<span style="color: red;">最小化损失函数</span>就变成了<span style="color: red;">最小化损失函数与这个0范数惩罚项之和</span>。
$$
min(L(\mathbf{w}, b) + \|\mathbf{W}\|_0)
$$

> 0范数：向量中非零元素的个数
>
> 1范数：向量中元素绝对值之和
>
> 2范数：向量中元素平方和的平方根

但是，0范数难求（不可微），我们可以变换一种思维：从让参数有尽可能多的0变换为让参数之和尽可能小，即1，2范数。不过，1范数会导致模型将权重集中在一小部分特征上， 而将其他特征的权重清除为零（稀疏化了）。最终，我们使用2范数。

然后，我们再利用一个非负的正则化超参数λ来平衡这个2范数惩罚项，.其中，较小的λ值对应较少约束的w， 而较大的λ值对w的约束更大。于是损失函数就变成了下式：
$$
L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|_2
$$
最后，$L_2$正则化回归的小批量随机梯度下降更新如下式：
$$
\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}
$$

## 2. 总结

为什么这种方法被称为权重衰减（weight decay）呢？因为我们在试图将w的大小缩小到0。与减小模型参数相比，权重衰减为我们提供了一种连续的机制来调整模型参数的复杂度。 



# dropout

## 1. 为什么dropout有效？

1. 以ensemble的角度解释，dropout掉不同的隐藏神经元就类似在训练不同的网络，整个dropout过程就相当于对很多个不同的神经网络取平均。而不同的网络产生不同的过拟合，一些互为“反向”的拟合相互抵消就可以达到整体上减少过拟合。
2. 类似于L1，L2的正则化方法，dropout掉一些神经单元就相当于限制了模型能力，换句话说假如我们的神经网络是在做出某种预测，它不应该对一些特定的线索片段太过敏感，即使丢失特定的线索，它也应该可以从众多其它线索中学习一些共同的模式，提升模型鲁棒性

## 2. train and test

训练时将保留的神经元以1/1-p的权重进行放大；测试时不做任何权重放缩



# 梯度裁剪

$$
\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.
$$

解决梯度爆炸



## code

```python
def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    # 网络全部的梯度
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

