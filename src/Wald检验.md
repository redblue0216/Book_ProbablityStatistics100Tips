# NO.12.tip： Wald检验

## 背景

&emsp;&emsp;定义拒绝域为 $R$ 的假设检验的**势函数**为
$$\beta(\theta) = \mathbb{P}_\theta(X \in R). \tag{1}$$

&emsp;&emsp;定义假设检验的**容度**为
$$\alpha = \sup_{\theta \in \Theta_0} \beta(\theta). \tag{2}$$

如果检验的容度小于等于 $\alpha$ 就称检验的水平为 $\alpha$

&emsp;&emsp;形式为 $\theta = \theta_0$ 的假设称为简单假设。形式为 $\theta > \theta_0$ 或 $\theta < \theta_0$ 的假设称为复合假设。形式为
$$H_0: \theta = \theta_0 \quad \text{对} \quad H_1: \theta \neq \theta_0$$
的假设称为双边检验。形式为
$$H_0: \theta \leq \theta_0 \quad \text{对} \quad H_1: \theta > \theta_0$$
或
$$H_0: \theta \geq \theta_0 \quad \text{对} \quad H_1: \theta < \theta_0$$
的假设称为单边检验。最常用的检验是双边的。

&emsp;&emsp;报告"拒绝 $H_0$"或"保留 $H_0$"并不能给出很多信息。相反，可能会问，对于任意 $\alpha$，该检验是否会拒绝原假设。更一般地，检验在显著性水平 $\alpha$ 拒绝原假设，那么也会在显著性水平 $\alpha' > \alpha$ 拒绝原假设。因此，存在一个拒绝原假设的最小的显著性水平 $\alpha$，称这个值为$p$ 值。

&emsp;&emsp;对于每一个 $\alpha$，可能会问：检验在显著性水平 $\alpha$ 拒绝 $H_0$ 吗？$p$ 值是拒绝 $H_0$ 的最小 $\alpha$ 值。如果拒绝 $H_0$ 的证据足够强，$p$ 值会很小。

&emsp;&emsp;假设对于任意 $\alpha \in (0,1)$，存在显著性水平为 $\alpha$ 的检验，它的拒绝域为 $R_\alpha$，则
$$p\text{值} = \inf\{\alpha: T(X^n) \in R_\alpha\}.$$

即，$p$ 值是可以拒绝 $H_0$ 的最小显著性水平。非正式地，$p$ 值是拒绝 $H_0$ 的证据强弱的度量：$p$ 值越小，拒绝 $H_0$ 的证据越强。

&emsp;&emsp;假设显著性水平为 $\alpha$ 的检验的形式为

$$拒绝 H_0 当且仅当 T(X^n) \geq c_\alpha$$

则
$$p\text{值} = \sup_{\theta \in \Theta_0} \mathbb{P}(T(X^n) \geq T(x^n)).$$

其中，$x^n$ 是 $X^n$ 的观测值。如果 $\Theta_0 = \{\theta_0\}$，则
$$p\text{值} = \mathbb{P}_{\theta_0}(T(X^n) \geq T(x^n)).$$

可以把以上定理表述如下：

$p$ 值是指，如果 $H_0$ 成立，检验统计量的值和实际观测值一样或更大的概率。

## 源与流

&emsp;&emsp;令 $\theta$ 为尺度参数，令 $\hat{\theta}$ 为 $\theta$ 的估计，$\hat{se}$ 为 $\hat{\theta}$ 的标准差的估计。

&emsp;&emsp;考虑检验
$$H_0: \theta = \theta_0 \quad \text{对} \quad H_1: \theta \neq \theta_0.$$

假设 $\hat{\theta}$ 是渐近正态的：
$$\frac{\hat{\theta} - \theta_0}{\hat{se}} \rightsquigarrow N(0,1).$$

显著水平为 $\alpha$ 的 Wald 检验：当 $|W| > z_{\alpha/2}$ 时拒绝 $H_0$，其中，
$$W = \frac{\hat{\theta} - \theta_0}{\hat{se}}. \tag{3}$$

&emsp;&emsp;渐近地，Wald 检验的显著水平为 $\alpha$，即当 $n \to \infty$ 时，
$$\mathbb{P}_{\theta_0}(|W| > z_{\alpha/2}) \to \alpha.$$

&emsp;&emsp;假设 $\theta$ 的真实值为 $\theta_* \neq \theta_0$，势函数 $\beta(\theta_*)$ 是正确拒绝原假设的概率，它的值近似为
$$1 - \Phi\left(\frac{\theta_0 - \theta_*}{\hat{se}} + z_{\alpha/2}\right) + \Phi\left(\frac{\theta_0 - \theta_*}{\hat{se}} - z_{\alpha/2}\right). \tag{4}$$

注意到当样本量增加时，$\hat{se}$ 趋向于 0。进一步检查 (4)，可以得到：(i) 如果 $\theta_*$ 离 $\theta_0$ 较远，则势函数很大；(ii) 如果样本量很大，则势函数很大。

&emsp;&emsp;（**比较两种预测算法**）在样本量为 $m$ 的检验集上检验一个预测算法，在样本量为 $n$ 的检验集上检验第二个预测算法。令 $X$ 表示算法 1 中预测不正确的个数，令 $Y$ 表示算法 2 中预测不正确的个数。则 $X \sim \text{Binomial}(m, p_1)$，$Y \sim \text{Binomial}(n, p_2)$。为了检验原假设 $p_1 = p_2$，记
$$H_0: \delta = 0 \quad \text{对} \quad H_1: \delta \neq 0.$$

其中，$\delta = p_1 - p_2$。极大似然估计为 $\hat{\delta} = \hat{p}_1 - \hat{p}_2$，它的标准差为
$$\hat{se} = \sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{m} + \frac{\hat{p}_2(1-\hat{p}_2)}{n}}.$$

Wald 检验的显著性水平为 $\alpha$，就是当 $|W| > z_{\alpha/2}$ 时拒绝 $H_0$，其中，
$$W = \frac{\hat{\delta} - 0}{\hat{se}} = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{m} + \frac{\hat{p}_2(1-\hat{p}_2)}{n}}}.$$

当 $p_1$ 离 $p_2$ 较远和样本量很大时，势函数会很大。

&emsp;&emsp;如果用同一个检验集去检验两个算法时会怎样呢？这两个样本不再独立。用到下面的策略。当算法 1 正确预测第 $i$ 个观测时，令 $X_i = 1$，否则，$X_i = 0$。当算法 2 正确预测第 $i$ 个观测时，令 $Y_i = 1$，否则，$Y_i = 0$。定义 $D_i = X_i - Y_i$。

&emsp;&emsp;令
$$\delta = \mathbb{E}(D_i) = \mathbb{E}(X_i) - \mathbb{E}(Y_i) = \mathbb{P}(X_i = 1) - \mathbb{P}(Y_i = 1).$$

$\delta$ 非参嵌入式估计为 $\hat{\delta} = \overline{D} = n^{-1}\sum_{i=1}^n D_i$ 和 $\hat{se}(\hat{\delta}) = S/\sqrt{n}$，其中，$S^2 = n^{-1}\sum_{i=1}^n(D_i - \overline{D})^2$。为了检验 $H_0: \delta = 0$ 对 $H_1: \delta \neq 0$，令 $W = \hat{\delta}/\hat{se}$，如果 $|W| > z_{\alpha/2}$，则拒绝 $H_0$。称为**配对检验**。

&emsp;&emsp;（**比较两个均值**）令 $X_1, \cdots, X_m$ 和 $Y_1, \cdots, Y_n$ 是分别从均值为 $\mu_1$ 和 $\mu_2$ 的总体中独立抽取的样本。检验原假设 $\mu_1 = \mu_2$，即检验
$$H_0: \delta = 0 \quad \text{对} \quad H_1: \delta \neq 0,$$

其中，$\delta = \mu_1 - \mu_2$。回忆起 $\delta$ 的非参嵌入式估计为 $\hat{\delta} = \overline{X} - \overline{Y}$，其标准差为
$$\hat{se} = \sqrt{\frac{s_1^2}{m} + \frac{s_2^2}{n}},$$

这里 $s_1^2$ 和 $s_2^2$ 是样本方差。水平为 $\alpha$ 的 Wald 检验在 $|W| > z_{\alpha/2}$ 时拒绝 $H_0$，这里
$$W = \frac{\hat{\delta} - 0}{\hat{se}} = \frac{\overline{X} - \overline{Y}}{\sqrt{s_1^2/m + s_2^2/n}}.$$