# NO.15.tip： EM算法

## 背景

&emsp;&emsp;期望最大化算法，或者EM算法，是寻找具有潜在变量的概率模型的最大似然解的一种通用的方法.考虑一个概率模型，其中我们将所有的观测变量联合起来记作$\boldsymbol{X}$，将所有的隐含变量记作$\boldsymbol{Z}$。联合概率分布$p(\boldsymbol{X},\boldsymbol{Z}\mid\theta)$由一组参数控制，记作$\theta$。我们的目标是最大化似然函数

$$p(\boldsymbol{X}\mid\theta) = \sum_{\boldsymbol{Z}} p(\boldsymbol{X},\boldsymbol{Z}\mid\theta) \tag{1}$$

这里，我们假设$\boldsymbol{Z}$是离散的，但是当$\boldsymbol{Z}$是连续变量或者离散变量与连续变量的组合时，方法是完全相同的，只需把求和换成适当的积分即可。

我们假设直接最优化$p(\boldsymbol{X}\mid\theta)$比较困难，但是最优化完整数据似然函数$p(\boldsymbol{X},\boldsymbol{Z}\mid\theta)$就容易得多。接下来，我们引入一个定义在潜在变量上的分布$q(\boldsymbol{Z})$。我们观察到，对于任意的$q(\boldsymbol{Z})$，下面的分解成立

$$\ln p(\boldsymbol{X}\mid\theta) = \mathcal{L}(q,\theta) + \mathrm{KL}(q\|p) \tag{2}$$

其中，我们定义了

$$\mathcal{L}(q,\theta) = \sum_{\boldsymbol{Z}} q(\boldsymbol{Z}) \ln\left\{\frac{p(\boldsymbol{X},\boldsymbol{Z}\mid\theta)}{q(\boldsymbol{Z})}\right\} \tag{3}$$

$$\mathrm{KL}(q\|p) = -\sum_{\boldsymbol{Z}} q(\boldsymbol{Z}) \ln\left\{\frac{p(\boldsymbol{Z}\mid\boldsymbol{X},\theta)}{q(\boldsymbol{Z})}\right\} \tag{4}$$

注意，$\mathcal{L}(q,\theta)$是概率分布$q(\boldsymbol{Z})$的一个泛函，并且是参数$\theta$的一个函数。值得仔细研究的是表达式(3)和(4)的形式，特别地，需要注意，二者的符号相反，并且$\mathcal{L}(q,\theta)$包含了$\boldsymbol{X}$和$\boldsymbol{Z}$的联合概率分布，而$\mathrm{KL}(q\|p)$包含了给定$\boldsymbol{X}$的条件下，$\boldsymbol{Z}$的条件概率分布。为了验证公式(2)给出的分解方式，我们首先使用概率的乘积规则，可得

$$\ln p(\boldsymbol{X},\boldsymbol{Z}\mid\theta) = \ln p(\boldsymbol{Z}\mid\boldsymbol{X},\theta) + \ln p(\boldsymbol{X}\mid\theta) \tag{5}$$

然后代入$\mathcal{L}(q,\theta)$的表达式。这得到了两项，一项消去了$\mathrm{KL}(q\|p)$，而另一项给出了所需的对数似然函数$\ln p(\boldsymbol{X}\mid\theta)$，其中我们用到了归一化的概率分布$q(\boldsymbol{Z})$的积分等于1的事实。

&emsp;&emsp;根据公式(4)，我们看到$\mathrm{KL}(q\|p)$是$q(\boldsymbol{Z})$和后验概率分布$p(\boldsymbol{Z}\mid\boldsymbol{X},\theta)$之间的Kullback-Leibler散度。回忆一下，Kullback-Leibler散度满足$\mathrm{KL}(q\|p)\ge 0$，当且仅当$q(\boldsymbol{Z})=p(\boldsymbol{Z}\mid\boldsymbol{X},\theta)$时等号成立。因此，根据公式(2)，$\mathcal{L}(q,\theta)\le \ln p(\boldsymbol{X}\mid\theta)$，换句话说，$\mathcal{L}(q,\theta)$是$\ln p(\boldsymbol{X}\mid\theta)$的一个下界。

## 源与流

&emsp;&emsp;EM算法是一个两阶段的迭代优化算法，用于寻找最大似然解。我们可以使用公式(2)来定义EM算法，证明它确实最大化了对数似然函数。假设参数向量的当前值为$\theta^{\text{旧}}$。在E步骤中，下界$\mathcal{L}(q,\theta^{\text{旧}})$关于$q(\boldsymbol{Z})$被最大化，而$\theta^{\text{旧}}$保持固定。最大化问题的解很容易看出来。我们注意到$\ln p(\boldsymbol{X}\mid\theta^{\text{旧}})$不依赖于$q(\boldsymbol{Z})$，因此$\mathcal{L}(q,\theta^{\text{旧}})$的最大值出现在Kullback-Leibler散度等于零的时候，换句话说，最大值出现在$q(\boldsymbol{Z})$与后验概率分布$p(\boldsymbol{Z}\mid\boldsymbol{X},\theta^{\text{旧}})$相等的时候。此时，下界等于对数似然函数。

&emsp;&emsp;在接下来的M步骤中，分布$q(\boldsymbol{Z})$保持固定，下界$\mathcal{L}(q,\theta)$关于$\theta$进行最大化，得到了某个新值$\theta^{\text{新}}$。这会使得下界$\mathcal{L}$增大（除非已经达到了极大值），这会使得对应的对数似然函数增大。由于概率分布$q$由旧的参数值确定，并且在M步骤中保持固定，因此它不会等于新的后验概率分布$p(\boldsymbol{Z}\mid\boldsymbol{X},\theta^{\text{新}})$，从而KL散度非零。于是，对数似然函数的增加量大于下界的增加量，如图9.13所示。如果我们将$q(\boldsymbol{Z})=p(\boldsymbol{Z}\mid\boldsymbol{X},\theta^{\text{旧}})$代入公式(3)，我们会看到，在E步骤之后，下界的形式为

$$\mathcal{L}(q,\theta) = \sum_{\boldsymbol{Z}} p(\boldsymbol{Z}\mid\boldsymbol{X},\theta^{\text{旧}}) \ln p(\boldsymbol{X},\boldsymbol{Z}\mid\theta) - \sum_{\boldsymbol{Z}} p(\boldsymbol{Z}\mid\boldsymbol{X},\theta^{\text{旧}}) \ln p(\boldsymbol{Z}\mid\boldsymbol{X},\theta^{\text{旧}}) \tag{6}$$

$$= \mathcal{Q}(\theta,\theta^{\text{旧}}) + \text{常数}$$

其中，常数就是分布$q$的熵，因此与$\theta$无关。从而在M步骤中，最大化的量是完整数据对数似然函数的期望，正如我们之前在混合高斯模型的情形中看到的那样。注意，我们进行优化的变量只出现在对数运算内部。如果联合概率分布$p(\boldsymbol{Z},\boldsymbol{X}\mid\theta)$由指数族分布的成员组成，或者由指数族分布成员的乘积组成，那么我们看到对数运算会抵消指数运算，从而使得M步骤通常比最大化对应的不完整数据对数似然函数$p(\boldsymbol{X}\mid\theta)$要容易得多。

&emsp;&emsp;EM算法将最大化似然函数这一困难的问题分解成了两个阶段，即E步骤和M步骤，每个步骤都很容易实现。尽管这样，对于复杂的模型来说，E步骤或者M步骤仍然无法计算。这就引出了对EM算法的两个扩展，叙述如下。

&emsp;&emsp;推广EM算法，或者简称GEM算法，解决的是M步骤无法计算的问题。这个算法不去关于$\theta$最大化$\mathcal{L}(q,\theta)$，而是改变参数的值去增大$\mathcal{L}(q,\theta)$的值。与之前一样，由于$\mathcal{L}(q,\theta)$是对数似然函数的一个下界，因此GEM算法的完整的EM循环保证了对数似然函数值的增大（除非参数已经对应于一个局部极大值）。一种使用GEM的方法是在M步骤中使用某种非线性最优化策略，例如共轭梯度算法。另一种形式的GEM算法，被称为期望条件最大化算法，或者简称ECM算法，涉及到在每个M步骤中进行若干了具有限制条件的最优化。例如，参数可能被划分为若干组，并且M步骤被划分成多个步骤，每个步骤最优化一个子集，同时保持其他的子集固定。

&emsp;&emsp;类似地，我们可以用下面的方法推广EM算法中的E步骤：对$\mathcal{L}(q,\theta)$关于$q(\boldsymbol{Z})$进行一个部分的最优化而不是完全的最优化。正如我们已经看到的，对于任意给定的$\theta$值，$\mathcal{L}(q,\theta)$关于$q(\boldsymbol{Z})$有一个唯一的最大值，它对应于后验概率分布$q_\theta(\boldsymbol{Z})=p(\boldsymbol{Z}\mid\boldsymbol{X},\theta)$，并且对于这个$q(\boldsymbol{Z})$的选择，下界$\mathcal{L}(q,\theta)$等于对数似然函数$\mathcal{L}(q,\theta)$。因此任何收敛于$\mathcal{L}(q,\theta)$的全局最大值的算法都会找到一个$\theta$值，这个值也是对数似然函数$\ln p(\boldsymbol{X}\mid\theta)$的全局最大值。只要$p(\boldsymbol{X},\boldsymbol{Z}\mid\theta)$是$\theta$的一个连续函数，那么根据连续性，$\mathcal{L}(q,\theta)$的任意一个局部极大值也会是$\ln p(\boldsymbol{X}\mid\theta)$的一个局部极大值。