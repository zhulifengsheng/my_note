[TOC]

# 深度学习分类

1. 有监督学习
2. 无监督学习
   - 自监督学习：
     1. 生成式学习：自编码器（Auto-Encoder），主要用于数据的降维或者特征的抽取，训练方法是重建输入
     1. 对比式学习：对比学习，训练方法是区分正负样本
3. 半监督学习：有一些数据有标记，给没有标记的数据指明了方向



# 自监督学习（通过自定义的有监督学习方式，挖掘数据信息）

为下游的有监督任务学习提供良好的representation，即通过自监督学习学习到泛化性能很强的representation【如：BERT】

## 自编码器

结构：1. encoder：它可以把原先的图像压缩成更低维度的向量【得到一个低维的高度抽象的向量，起到**压缩数据、获取主要信息**的效果】。 2. decoder：它可以把压缩后的向量还原成图像，通常它们使用的都是神经网络。

![在这里插入图片描述](01-autoencoder1.png)

**多个自编码器模型：**

1. de-nosing auto-encoder【典型的例子是BERT】

![image-20220619160512551](01-autoencoder2.png)

2. feature disentangle[One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization](https://arxiv.org/abs/1904.05742)

   encoder学习（压缩）得到的特征会包含很多的信息，比如对一段声音进行压缩就会包含语音内容和发言者音色、音调等信息。那有没有一种方法可以将这些信息区分开呢？比如，下图将content和speaker的信息区分开。

   如果可以成功地将声音信号encode出来的向量中**内容和音色区分开**，那么给两段声音（A说你好，B说再见），就可以做到变声的效果（A说再见，B说你好）。

   ![image-20220619160912561](01-autoencoder3.png)

3. variational auto encoder（生成模型）

   现在，我们让AE去训练还原月球照片，接下来我们在code空间中，取全月和半月照片编码的中间点，我们期望模型可以生成一个3/4月的照片，但是AE是做不到的【左图】。为了解决这个问题，我们引入了噪声，使得图片的编码区域得到扩大，从而掩盖掉失真的空白编码点【中图】。不过这还不够充分，我们进一步将噪声拉大，拉成无限长，让他的编码覆盖整个code空间，并保证在原编码附近的概率最大，越远离原编码概率越小，这就是VAE【右图】。

   ![](01-vae2.jpg)![](01-vae3.jpg)![](01-vae4.jpg)

   ------

   **VAE的工作流程：**【从左图到右图，encoder将输入从一个确定的空间点编码成一个空间上的分布（概率分布）】我们将encoder编码的code看作是隐变量z【z服从着某个未知的概率分布】，那么encoder的工作就是建模q(z|x)【得到一个概率分布】，decoder的工作就是建模P(x|z)【从z的概率分布中的采样一个z'，还原回输入x】。

   **我们先进行一个假设**：前面提到过，希望x经过encoder的编码，可以得到一个覆盖整个code空间的分布，所以我们先验地认为q(z|x)服从标准正态分布。

   因此，我们利用encoder来拟合q(z|x)，得到均值$\mu$和方差$\delta$
   
   然后，如果只有这些的话，是显然不行的。因为神经网络在拟合的过程中，会倾向于让方差$\delta$输出为0，就相当于取消了噪声，取消了随机性，就不再是正态分布了。那么又返回了AE那种code是一个确定的点的状态了。所以，VAE要在loss中加入$KL(q(z|x)||p(z))$来防止随机性被网络拟合掉。
   
   现在，VAE就是用encoder将x建模为某个分布z，再从z中采样一个出来，送入decoder进行x的还原。
   
   <img src="01-vae5.jpg" style="zoom:80%;" />
   
   ------
   
   从数学角度入手：我们求解的目标是最大化$P(x) = \int_zp(x,z)dz = \int_zp(z)p(x|z)dz$【极大似然$P(x)$】，不过这个式子无法求解，这导致$p(z|x) = \frac{P(z)p(x|z)}{p(x)}$也是无法求解的，所以我们需要另寻他法，利用神经网络来拟合$p(z|x)$。
   
   再回到我们希望的求解目标上，现在：
   $$
   \begin{aligned}
   \log P(x) &= \int_zq(z|x)\log p(x)dz  \space\space\space (q(z|x) 是 encoder 的输出，就是上面说的拟合p(z|x))\\
   &= \int_zq(z|x) \log(\frac{p(z,x)}{p(z|x)})dz \\
   &= \int_zq(z|x) \log(\frac{p(z,x)}{q(z|x)}\times\frac{q(z|x)}{p(z|x)})dz \\
   &= \int_zq(z|x) \log(\frac{p(z,x)}{q(z|x)})dz+\int_zq(z|x)\log(\frac{q(z|x)}{p(z|x)})dz \\
   &= \int_zq(z|x) \log(\frac{p(x|z)p(z)}{q(z|x)})dz + KL(q(z|x)||p(z|x)) \\
   &\ge \int_zq(z|x) \log(\frac{p(x|z)p(z)}{q(z|x)})dz【下界】\\
   &= \int_zq(z|x) \log(\frac{p(z)}{q(z|x)})dz + \int_zq(z|x) \log p(x|z)dz\\
   &= -KL(q(z|x)||p(z)) + \int_zq(z|x) \log p(x|z)dz \\
   &= E_{q(z|x)}[\log p(x|z)] - KL(q(z|x)||p(z)) \\
   &= 重构loss + KL\space loss
   \end{aligned}
   $$
   
   ------
   
   因为采样是不可以求导的，所以我们要使用重采样技巧：
   
   下图是VAE结构图，我们的做法是让encoder生成两个code，一个是$\mu$，另一个是$\sigma$[它给高斯噪声e分配权重，代表了一个采样的过程]（从$N(\mu,\sigma^2)$中采样一个$Z$等于从$N(0,1)$中采样一个$\epsilon$，让$Z=\mu+\sigma \times \epsilon$）。右下角的loss是为了防止encoder编码出来的随机变量权重$\sigma$为0，导致模型没有给code加入噪声。【即KL loss】
   
   ![](01-vae.png)
   



## 对比学习

### 对比学习的LOSS

1. NCE：给定一个输入$x$，它的正例从$P_d(y|x)$中采样得到，它的负例从$P_n(y|x)$中采样得到。设正例采样个数:负例采样个数为1:k。记正例D=1，采样来自$P_d(y|x)$；负例D=0，采样来自$P_n(y|x)$。
   $$
   \begin{aligned}
   P(D=0,y|x) = \frac{k}{k+1}P_n(y|x) &;\space
   P(D=1,y|x) = \frac{1}{k+1}P_d(y|x) \\
   假设一个样本y既可能是被选做正例，也可能被选做负例，P(y|x) &= P(D=0,y|x)+P(D=1,y|x) \\
   P(D=0|x,y) = \frac{P(D=0,y|x)}{P(y|x)} &= \frac{kP_n(y|x)}{P_d(y|x) + kP_n(y|x)} \\
   P(D=1|x,y) = \frac{P(D=1,y|x)}{P(y|x)} &= \frac{P_d(y|x)}{P_d(y|x) + kP_n(y|x)} \\
   现在我们用神经网络\theta，P(D=1|x,y,\theta) &= \frac{P_\theta(y|x)}{P_\theta(y|x) + kP_n(y|x)} \\
   来拟合正例的分布P_d，P(D=0|x,y,\theta) &= \frac{kP_n(y|x)}{P_\theta(y|x) + kP_n(y|x)}
   \end{aligned}
   $$
   ```
   我们来用一个实例来模拟上面的公式
   输入x为“你”，构建一个正例“好”的概率为0.8；构建一个负例“好”的概率为0.1
   设k等于9
   p(负例的好|你) = 9/10*0.1 = 0.09
   p(正例的好|你) = 1/10*0.8 = 0.08
   p(好|你) = 0.17
   p(负例|你，好) = 9/17
   p(正例|你，好) = 8/17
   ```
   
   **现在需要解决的问题是：给定一个输出$y$，判断它是输入$x$的正例还是负例（二分类）**  
   $$
   \begin{aligned}
   根据交叉熵损失，
   Loss_{NCE} &= -\log P\left( 正例|x,y,\theta \right) -\sum_{j=1}^k{\log P\left( 负例|x,y_{j},\theta \right)} \\
   &= -\log \frac{P_\theta(y|x)}{P_\theta(y|x)+kP_n(y|x)} - \sum_{j=1}^{k}\log \frac{kP_n(y_j|x)}{P_\theta(y_j|x)+kP_n(y_j|x)}
   \end{aligned}
   $$
   
   ------
   
   在Skip-gram的词向量训练实践中，就会使用负采样技术，构建标签为0的负例token上下文对，然后用模型进行逻辑回归的学习（二分类交叉熵损失）。
   
   
   
2. InfoNCE：互信息表示两个相关变量的相互依赖程度，如下图所示，要预测的未来信息$x_{t+k}$和当前时刻的全局信息$c_t$的依赖程度越高，就越可以得到好的预测结果。

    ![](cpc.png)
    
    互信息的公式见[机器学习第一章](my_note\机器学习\01-machine_learning.md)
    
    ------
    
    为了最大化互信息，我们使用InfoNCE，用神经网络来拟合正负样本的概率密度比（它表明正样本的密度比大，负样本的密度比小，可以认为是一种相似性的度量），公式如下：
    $$
    f_k(x_{t+k}, c_t, \theta) \propto \frac{p(x_{t+k}|c_t)}{p(x_{t+k})} \\
    L_N = -\log \frac{f_k(x_{t+k}, c_t, \theta)}{\sum_{x_j \in X}f_k(x_j, c_t, \theta)}
    $$
    
    现在证明它和互信息的关系：
    $$
    \begin{aligned}
    L_N &= -\log \frac{f_k(x_{t+k}, c_t, \theta)}{\sum_{x_j \in X}f_k(x_j, c_t, \theta)} \\
    &= -\log \frac{\frac{p(x_{t+k}|c_t)}{p(x_{t+k})}}{\frac{p(x_{t+k}|c_t)}{p(x_{t+k})} +\sum_{x_j \in X_{neg}}\frac{p(x_j|c_t)}{p(x_j)}} \\
    &= \log(1+\frac{p(x_{t+k})}{p(x_{t+k}|c_t)}\sum_{x_j \in X_{neg}}\frac{p(x_j|c_t)}{p(x_j)}) \\
    &\approx \log(1+\frac{p(x_{t+k})}{p(x_{t+k}|c_t)}(N-1)E_{x_j}\frac{p(x_j|c_t)}{p(x_j)}) \\
    &= \log(1+\frac{p(x_{t+k})}{p(x_{t+k}|c_t)}(N-1)) \\
    &\ge \log\frac{p(x_{t+k})}{p(x_{t+k}|c_t)}N \\
    &= - I(x_{t+k};c_t) +\log N
    \end{aligned}
    $$
    InfoNCE可以认为是NCE的多分类版本，可以将上面的$L_N$看作是经过softmax函数得到的交叉熵。

### 对比学习的经典paper

#### MoCo
