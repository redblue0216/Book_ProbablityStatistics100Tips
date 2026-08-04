# NO.10.tip： AIC和BIC准则

## 背景

&emsp;&emsp;一般来说，训练误差

$$\overline{\mathrm{err}} = \frac{1}{N}\sum_{i=1}^{N} L(y_i, \hat{f}(x_i)) \tag{1}$$

比真实的预测误差 $\mathrm{Err}_{\mathcal{T}}$ 小，因为同样的数据既被用于拟合，又被用于评估其误差（见习题7.9）。拟合算法往往会对训练数据本身（而不是真实的分布）产生适应性，因此训练误差 $\overline{\mathrm{err}}$ 将会是对泛化误差 $\mathrm{Err}_{\mathcal{T}}$ 的过于乐观的估计。

&emsp;&emsp;两者产生差异的部分原因是测试观测的位置或评估点发生位置的不一样。$\mathrm{Err}_{\mathcal{T}}$ 可以认为是样本集外的样本（extra-sample）误差，因为测试样本不必和训练样本一样。$\overline{\mathrm{err}}$ 的乐观本质在我们仅关注样本集内的样本（in-sample）误差时最容易理解

$$\mathrm{Err}_{\mathrm{in}} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{E}_{Y^0}[L(Y_i^0, \hat{f}(x_i))|\mathcal{T}] \tag{2}$$

这里 $Y^0$ 表示我们在每个训练样本 $x_i$ 获得的 $N$ 个新响应。我们将（误差估计的）乐观程度（optimism）定义为训练误差 $\mathrm{Err}_{\mathrm{in}}$ 和 $\overline{\mathrm{err}}$ 的差

$$\mathrm{op} \equiv \mathrm{Err}_{\mathrm{in}} - \overline{\mathrm{err}} \tag{3}$$

这个值一般都是正的，因为 $\overline{\mathrm{err}}$ 通常是对预测误差偏小的估计。最后，平均乐观程度是乐观程度在所有训练集上的期望误差

$$\omega \equiv \mathbb{E}_y(\mathrm{op}) \tag{4}$$

这里，训练集的预测子是固定的，期望是对训练集上响应或结果值上（即 $Y_i^0$）取的，因此我们的记号是 $\mathbb{E}_y$ 而不是 $\mathbb{E}_{\mathcal{T}}$。通常，我们可以计算出期望乐观程度 $\omega$ 而不是 $\mathrm{op}$，正如我们能够估计期望预测误差 $\mathrm{Err}$，而不能估计出条件误差 $\mathrm{Err}_{\mathcal{T}}$。

&emsp;&emsp;对于平方误差、0-1以及其他的损失函数，可以证明如下一般性的结论

$$\omega = \frac{2}{N}\sum_{i=1}^{N} \mathrm{Cov}(\hat{y}_i, y_i) \tag{5}$$

其中 $\mathrm{Cov}$ 是协方差。因此，$\overline{\mathrm{err}}$ 低估真实误差的程度依赖于 $y_i$ 能以多强的程度影响它自己的预测值。我们对数据的拟合程度越高，$\mathrm{Cov}(\hat{y}_i, y_i)$ 会越大，也将因此而增大乐观程度。

&emsp;&emsp;由此，我们有如下重要的关系

$$\mathbb{E}_y(\mathrm{Err}_{\mathrm{in}}) = \mathbb{E}_y(\overline{\mathrm{err}}) + \frac{2}{N}\sum_{i=1}^{N} \mathrm{Cov}(\hat{y}_i, y_i) \tag{6}$$

&emsp;&emsp;如果 $\hat{y}_i$ 是 $d$ 维输入或基函数的线性拟合结果，此表达式还可简化。例如，对于加性误差模型 $Y = f(X) + \varepsilon$

$$\sum_{i=1}^{N} \mathrm{Cov}(\hat{y}_i, y_i) = d\sigma_{\varepsilon}^2 \tag{7}$$

因此

$$\mathbb{E}_y(\mathrm{Err}_{\mathrm{in}}) = \mathbb{E}_y(\overline{\mathrm{err}}) + 2\cdot\frac{d}{N}\sigma_{\varepsilon}^2 \tag{8}$$

式(7)是有效参数个数的基础。乐观程度随着输入维数或者使用的基函数个数 $d$ 线性增长，但是随训练集样本数目的增加而减少。式(8)对于其他的误差模型，比如二值数据与熵损失函数，也能近似的成立。

&emsp;&emsp;一种估计预测误差的明显方式是估计乐观程度，然后将它加到训练误差 $\overline{\mathrm{err}}$ 上。在下一节中介绍的方法——$C_p$、AIC、BIC以及其他的方法——都是这样对一类特殊的估计起作用的，这类估计是它们的参数的线性函数。

## 源与流

&emsp;&emsp;样本内误差的一般形式为

$$\widehat{\mathrm{Err}}_{\mathrm{in}} = \overline{\mathrm{err}} + \hat{\omega} \tag{9}$$

其中 $\hat{\omega}$ 是对平均乐观程度的估计。

&emsp;&emsp;利用式(8)，当 $d$ 个参数通过平方误差损失函数拟合时，可计算出一个所谓的 $C_p$ 统计量：

$$C_p = \overline{\mathrm{err}} + 2\cdot\frac{d}{N}\hat{\sigma}_{\varepsilon}^2 \tag{10}$$

这里 $\hat{\sigma}_{\varepsilon}^2$ 是对噪声方差的估计，它从低偏差模型的均方误差获得。利用这个准则，我们通过一个正比于使用的基函数数量的因子，来调整训练误差。

&emsp;&emsp;Akaike 信息准则是对 $\mathrm{Err}_{\mathrm{in}}$ 的类似但更一般的估计，如果用的是对数似然损失函数。它依赖于与式(8)类似的关系。该关系

$$-2\cdot \mathbb{E}\log \mathrm{Pr}_{\hat{\theta}}(Y) \approx -\frac{2}{N}\cdot\mathbb{E}[\mathrm{loglik}] + 2\cdot\frac{d}{N} \tag{11}$$

在 $N \to \infty$ 时，渐进的成立。这里 $\mathrm{Pr}_{\hat{\theta}}(Y)$ 表示一族 $Y$ 的密度（包含"真实"的密度函数），$\hat{\theta}$ 是对 $\theta$ 的最大似然估计，而 $\mathrm{loglik}$ 是最大化的似然函数

$$\mathrm{loglik} = \sum_{i=1}^{N} \log \mathrm{Pr}_{\hat{\theta}}(y_i) \tag{12}$$

例如，对 logistic 回归模型，使用二项分布的对数似然函数，有

$$\mathrm{AIC} = -\frac{2}{N}\cdot\mathrm{loglik} + 2\cdot\frac{d}{N} \tag{13}$$

对于线性回归模型，其损失函数为平方误差（其方差 $\sigma_{\varepsilon}^2 = \hat{\sigma}_{\varepsilon}^2$ 假定已知），对应的 AIC 统计量等价于 $C_p$，因此我们将它们统称为 AIC。

&emsp;&emsp;为了将 AIC 用于模型选择，我们就选择给定模型中 AIC 值最小的那个。对于非线性和其他复杂的模型，需要将 $d$ 替换为其他的能够衡量模型复杂性的数值。

&emsp;&emsp;给定一族模型 $f_{\alpha}(x)$，其中 $\alpha$ 是可调整的参数，用 $\overline{\mathrm{err}}(\alpha)$ 和 $d(\alpha)$ 表示训练误差以及对应模型的参数个数，则对这族模型，我们定义

$$\mathrm{AIC}(\alpha) = \overline{\mathrm{err}}(\alpha) + 2\cdot\frac{d(\alpha)}{N}\hat{\sigma}_{\varepsilon}^2 \tag{14}$$

函数 $\mathrm{AIC}(\alpha)$ 提供对测试误差曲线的估计，我们去寻找能最小化该值的可调参数 $\hat{\alpha}$。

&emsp;&emsp;尽管 AIC 和 BIC 是非常相似的，BIC 源自一个非常不同的起因，模型选择的贝叶斯方法。

&emsp;&emsp;假定有一族候选模型 $\mathcal{M}_m, m = 1, \ldots, M$ 且对应的模型参数为 $\theta_m$，我们希望从中选择一个最佳模型。假定每个模型 $\mathcal{M}_m$ 的参数的先验分布为 $\Pr(\theta_m|\mathcal{M}_m)$，则给定模型的后验概率为

$$\Pr(\mathcal{M}_m|Z) \propto \Pr(\mathcal{M}_m)\cdot\Pr(Z|\mathcal{M}_m) \propto \Pr(\mathcal{M}_m)\cdot\int \Pr(Z|\theta_m,\mathcal{M}_m)\Pr(\theta_m|\mathcal{M}_m)d\theta_m \tag{21}$$

其中 $Z$ 代表训练数据 $\{x_i, y_i\}_1^N$。为了比较两个模型 $\mathcal{M}_m$ 和 $\mathcal{M}_\ell$，我们计算后验概率比

$$\frac{\Pr(\mathcal{M}_m|Z)}{\Pr(\mathcal{M}_\ell|Z)} = \frac{\Pr(\mathcal{M}_m)}{\Pr(\mathcal{M}_\ell)}\cdot\frac{\Pr(Z|\mathcal{M}_m)}{\Pr(Z|\mathcal{M}_\ell)} \tag{22}$$

如果比值大于 1，我们就选择模型 $m$，否则我们选择模型 $\ell$。最右边的一项

$$\mathrm{BF}(Z) = \frac{\Pr(Z|\mathcal{M}_m)}{\Pr(Z|\mathcal{M}_\ell)} \tag{23}$$

称为贝叶斯因子（Bayes factor），即数据对后验概率比的贡献。

&emsp;&emsp;一般情况下，我们假设模型的先验是均匀分布，这样 $\Pr(\mathcal{M}_m)$ 是常数。我们需要一些近似 $\Pr(Z|\mathcal{M}_m)$ 的方法。通过对积分的所谓拉普拉斯近似以及另外一些简化，式(21)化简为

$$\log\Pr(Z|\mathcal{M}_m) = \log\Pr(Z|\hat{\theta}_m,\mathcal{M}_m) - \frac{d_m}{2}\cdot\log N + O(1) \tag{24}$$

这里 $\hat{\theta}_m$ 是最大似然估计，$d_m$ 是模型 $\mathcal{M}_m$ 的自由参数个数。如果损失函数定义为

$$-2\log\Pr(Z|\hat{\theta}_m,\mathcal{M}_m)$$

&emsp;&emsp;因此，选择最小 BIC 的模型等价于选择能够（近似）最大后验概率的模型。但这个框架给了我们更多的东西。如果我们对 $M$ 个模型计算 BIC，有相应的 $\mathrm{BIC}_m, m = 1, 2, \ldots, M$，那么我们可以用它们来估计每个模型 $\mathcal{M}_m$ 的后验概率

$$\frac{e^{-\frac{1}{2}\cdot\mathrm{BIC}_m}}{\sum_{\ell=1}^{M} e^{-\frac{1}{2}\cdot\mathrm{BIC}_\ell}} \tag{25}$$

这样我们不仅估计了最好的模型，同时还评估了各个模型的相对好处。

&emsp;&emsp;就模型选择的目的而言，在 AIC 和 BIC 之间并没有明确的倾向。BIC 作为选择的准则是渐进一致的。这也就是说，给定一族模型，包含真实的模型，BIC 选择正确模型的概率会随着样本容量 $N \to \infty$ 接近 1。但是对 AIC 却并不是这样，它在 $N \to \infty$ 的时候倾向选择过于复杂的模型。另外一方面，对有限样本，BIC 经常由于对复杂模型的严重惩罚而选择过于简单的模型。