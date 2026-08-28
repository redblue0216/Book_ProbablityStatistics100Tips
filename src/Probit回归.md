# NO.32.tip： Probit回归

## 背景

我们将讨论 probit 回归，该回归类似于二元逻辑回归，只是 probit 回归使用 $\mu_n = \Phi(a_n)$ 而不是 $\mu_n = \sigma(a_n)$ 作为均值函数，其中 $\Phi$ 是标准正态分布的累积分布函数，$a_n = \mathbf{w}^\top \mathbf{x}_n$。因此，对应的链接函数为 $a_n = \ell(\mu_n) = \Phi^{-1}(\mu_n)$；高斯分布的累积分布函数的逆函数被称为 **probit 函数**。

高斯分布的累积分布函数 $\Phi$ 与逻辑函数非常相似。因此，probit 回归和"正则"逻辑回归表现得非常相似。然而，probit 回归有一些优点。特别是，probit 回归作为一个潜在变量模型，有一个简单的解释，即源于经济学研究的选择理论。这也简化了贝叶斯参数推理的任务。

## 源与流

**潜在变量解释**

我们可以将 $a_n = \mathbf{w}^\top \mathbf{x}_n$ 解释为这样一个因子：对于给定的输入 $\mathbf{x}_n$，该因子与我们对输入积极响应（即生成 $y_n=1$）的可能性成一定的比例。然而，通常还有其他未观察到的因子会影响个人的反应。让我们使用高斯噪声 $\epsilon_n \sim \mathcal{N}(0,1)$ 对这些隐藏因子进行建模。假设对积极结果的组合偏好由潜在变量 $z_n = \mathbf{w}^\top \mathbf{x}_n + \epsilon_n$ 表示。只有当这个潜在因子是正值而不是负值时，才会选择积极标签。即，

$$
y_n = \mathbb{I}(z_n \geq 0) \tag{1}
$$

当我们将 $z_n$ 边缘化时，我们得到了 probit 模型：

$$
\begin{aligned}
p(y_n=1 \mid \mathbf{x}_n, \mathbf{w}) &= \int \mathbb{I}(z_n \geq 0) \mathcal{N}(z_n \mid \mathbf{w}^\top \mathbf{x}_n, 1) \,\mathrm{d}z_n \\
&= p(\mathbf{w}^\top \mathbf{x}_n + \epsilon_n \geq 0) = p(\epsilon_n \geq -\mathbf{w}^\top \mathbf{x}_n) \\
&= 1 - \Phi(-\mathbf{w}^\top \mathbf{x}_n) = \Phi(\mathbf{w}^\top \mathbf{x}_n)
\end{aligned}
\tag{2}
$$

因此，我们可以将 probit 回归视为应用于噪声输入的阈值函数。

我们可以使用同样的方式来解释逻辑回归。然而，在这种情况下，噪声项 $\epsilon_n$ 来自**逻辑分布**（logistic distribution），其定义如下：

$$
f(y \mid \mu, s) \triangleq \frac{e^{-\frac{y-\mu}{s}}}{s\left(1+e^{-\frac{y-\mu}{s}}\right)^2} = \frac{1}{4s}\mathrm{sech}^2\left(\frac{y-\mu}{s^2}\right) \tag{3}
$$

其中，均值为 $\mu$，方差为 $\dfrac{s^2\pi^2}{3}$。该分布的累积分布函数由下式给出：

$$
F(y \mid \mu, s) = \frac{1}{1+e^{-\frac{y-\mu}{s}}} \tag{4}
$$

很显然，如果我们使用 $\mu=0$ 和 $s=1$ 的逻辑噪声，我们就可以得到逻辑回归。然而，处理高斯噪声在计算上更加容易，接下来将展开讨论。

**最大似然估计**

**使用随机梯度下降的最大似然估计**

我们可以使用标准梯度方法寻找到 probit 回归的最大似然估计。设 $\mu_n = \mathbf{w}^\top \mathbf{x}_n$，$\tilde{y}_n \in \{-1,+1\}$，则单个样例 $n$ 的对数似然的梯度由下式给出：

$$
\mathbf{g}_n \triangleq \frac{\mathrm{d}}{\mathrm{d}\mathbf{w}} \log p(\tilde{y}_n \mid \mathbf{w}^\top \mathbf{x}_n) = \frac{\mathrm{d}\mu_n}{\mathrm{d}\mathbf{w}} \frac{\mathrm{d}}{\mathrm{d}\mu_n} \log p(\tilde{y}_n \mid \mathbf{w}^\top \mathbf{x}_n) = \mathbf{x}_n \frac{\tilde{y}_n \phi(\mu_n)}{\Phi(\tilde{y}_n \mu_n)} \tag{5}
$$

其中，$\phi$ 是标准正态分布的概率密度函数，$\Phi$ 是概率密度函数的累积分布函数。类似地，单个样例的海森矩阵由下式给出：

$$
\mathbf{H}_n = \frac{\mathrm{d}}{\mathrm{d}\mathbf{w}^2} \log p(\tilde{y}_n \mid \mathbf{w}^\top \mathbf{x}_n) = -\mathbf{x}_n \left( \frac{\phi(\mu_n)^2}{\Phi(\tilde{y}_n \mu_n)^2} + \frac{\tilde{y}_n \mu_n \phi(\mu_n)}{\Phi(\tilde{y}_n \mu_n)} \right) \mathbf{x}_n^\top \tag{6}
$$

这可以传递给任何基于梯度的优化器。

**使用期望最大化的最大似然估计**

我们可以使用 probit 回归的潜在变量解释，推导出一个期望最大化算法来拟合模型。

假设在 $\mathbf{w}$ 上有一个先验分布 $\mathcal{N}(\mathbf{0}, \mathbf{V}_0)$，完全数据对数似然具有以下形式：

$$
\begin{aligned}
\ell(\mathbf{z}, \mathbf{w} \mid \mathbf{V}_0) &= \log p(\mathbf{y} \mid \mathbf{z}) + \log \mathcal{N}(\mathbf{z} \mid \mathbf{X}\mathbf{w}, \mathbf{I}) + \log \mathcal{N}(\mathbf{w} \mid \mathbf{0}, \mathbf{V}_0) \\
&= \sum_n \log p(y_n \mid z_n) - \frac{1}{2}(\mathbf{z} - \mathbf{X}\mathbf{w})^\top(\mathbf{z} - \mathbf{X}\mathbf{w}) - \frac{1}{2}\mathbf{w}^\top \mathbf{V}_0^{-1}\mathbf{w}
\end{aligned}
\tag{7}
$$

期望步骤中的后验分布是一个**截断的高斯概率分布**（truncated Gaussian）：

$$
p(z_n \mid y_n, \mathbf{x}_n, \mathbf{w}) = 
\begin{cases}
\mathcal{N}(z_n \mid \mathbf{w}^\top \mathbf{x}_n, 1)\,\mathbb{I}(z_n &gt; 0) & \text{如果 } y_n=1 \\[6pt]
\mathcal{N}(z_n \mid \mathbf{w}^\top \mathbf{x}_n, 1)\,\mathbb{I}(z_n &lt; 0) & \text{如果 } y_n=0
\end{cases}
\tag{8}
$$

在式 (7) 中，我们发现，$\mathbf{w}$ 仅线性地依赖于 $\mathbf{z}$，所以我们只需要计算 $\mathbb{E}[\mathbf{z} \mid y_n, \mathbf{x}_n, \mathbf{w}]$，也就是说仅需要计算后验均值。可以证明，其计算公式为：

$$
\mathbb{E}[z_n \mid \mathbf{w}, \mathbf{x}_n] = 
\begin{cases}
\mu_n + \dfrac{\phi(\mu_n)}{1-\Phi(-\mu_n)} = \mu_n + \dfrac{\phi(\mu_n)}{\Phi(\mu_n)} & \text{如果 } y_n=1 \\[10pt]
\mu_n - \dfrac{\phi(\mu_n)}{\Phi(-\mu_n)} = \mu_n - \dfrac{\phi(\mu_n)}{1-\Phi(\mu_n)} & \text{如果 } y_n=0
\end{cases}
\tag{9}
$$

其中，$\mu_n = \mathbf{w}^\top \mathbf{x}_n$。

在最大化步骤中，我们使用岭回归来估计 $\mathbf{w}$，其中 $\boldsymbol{\mu} = \mathbb{E}[\mathbf{z}]$ 是我们试图预测的输出。具体来说，我们有：

$$
\hat{\mathbf{w}} = (\mathbf{V}_0^{-1} + \mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \boldsymbol{\mu} \tag{10}
$$

期望最大化算法很简单，但可能比直接梯度法慢得多，如图 15-12 所示。这是因为期望步骤中的后验熵相当高，因为我们只观察到 $z$ 是正值还是负值，但没有给出关于其大小的似然信息。使用更强的正则化因子可以帮助加快收敛速度，因为这种方法限制了看似合理的 $z$ 值的范围。此外，还可以使用各种加速技巧，例如数据增强方法。

**贝叶斯推理**

可以使用 probit 回归的潜在变量公式，来推导用于近似后验 $p(\mathbf{w} \mid \mathcal{D})$ 的简单吉布斯采样算法。

其关键思想是使用辅助潜在变量，在给定这个潜在变量的条件下，使得整个模型成为共轭线性高斯模型。潜在变量的完全条件由下式给出：

$$
p(z_i \mid y_i, \mathbf{x}_i, \mathbf{w}) = 
\begin{cases}
\mathcal{N}(z_i \mid \mathbf{w}^\top \mathbf{x}_i, 1)\,\mathbb{I}(z_i &gt; 0) & \text{如果 } y_i=1 \\[6pt]
\mathcal{N}(z_i \mid \mathbf{w}^\top \mathbf{x}_i, 1)\,\mathbb{I}(z_i &lt; 0) & \text{如果 } y_i=0
\end{cases}
\tag{11}
$$

因此，后验是截断的高斯分布。我们可以从截断高斯 $\mathcal{N}(z \mid \mu, \sigma)\mathbb{I}(a \leq z \leq b)$ 中分两步进行采样：第一次采样 $u \sim \mathcal{U}(\Phi((a-\mu)/\sigma), \Phi((b-\mu)/\sigma))$，然后设置 $z = \mu + \sigma\Phi^{-1}(u)$。

参数的完全条件由下式给出：

$$
p(\mathbf{w} \mid \mathcal{D}, \mathbf{z}, \boldsymbol{\lambda}) = \mathcal{N}(\mathbf{w}_N, \mathbf{V}_N) \tag{12}
$$

$$
\mathbf{V}_N = (\mathbf{V}_0^{-1} + \mathbf{X}^\top \mathbf{X})^{-1} \tag{13}
$$

$$
\mathbf{w}_N = \mathbf{V}_N(\mathbf{V}_0^{-1}\mathbf{w}_0 + \mathbf{X}^\top \mathbf{z}) \tag{14}
$$

也可以使用变分贝叶斯，其速度往往要快得多。

**有序probit回归**

probit 回归潜在变量解释的一个优点是，可以很容易扩展到响应变量以某种方式排序的情况，例如输出低、中和高。这被称为**有序回归**（ordinal regression）。其基本思想如下：如果有 $C$ 个输出值，我们引入 $C+1$ 个阈值 $\gamma_j$ 并设置

$$
y_n = j \quad \text{如果} \quad \gamma_{j-1} < z_n \leq \gamma_j \tag{15}
$$

其中，$\gamma_0 \leq \cdots \leq \gamma_C$。出于可识别性的原因，我们设置 $\gamma_0 = -\infty$，$\gamma_1 = 0$ 和 $\gamma_C = \infty$。例如，如果 $C=2$，这将简化为标准的二元 probit 模型，其中当 $z_n &lt; 0$ 时，$y_n=0$；当 $z_n \geq 0$ 时，$y_n=1$。如果 $C=3$，我们将实数线上的数据划分为 3 个区间：$(-\infty, 0]$，$(0, \gamma_2]$，$(\gamma_2, \infty)$。我们可以通过改变参数 $\gamma_2$ 来确保每个区间中概率质量下降的相对量正确，从而匹配每个类别标签的经验频率。

由于我们需要对 $\mathbf{w}$ 和 $\boldsymbol{\gamma}$ 进行优化，并且后者必须服从排序约束，因此寻找到该模型的最大似然估计比二元 probit 回归要困难一些。例如，关于基于期望最大化的方法。也可以推导出该模型的简单吉布斯采样算法。

**多项式probit模型**

现在考虑响应变量可以采用 $C$ 个无序分类值的情况，即 $y_n \in \{1, \cdots, C\}$。多项式 probit 模型定义如下：

$$
z_{nc} = \mathbf{w}_c^\top \mathbf{x}_n + \epsilon_{nc} \tag{16}
$$

$$
\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{R}) \tag{17}
$$

$$
y_n = \arg\max_c z_{nc} \tag{18}
$$

为了了解更多有关该模型及其与多项式逻辑回归之间联系的信息。

如果使用 $y_{nc} = \mathbb{I}(z_{nc} > 0)$ 而不是设置 $y_n = \arg\max_c z_{ic}$，那么我们可以得到一个称为**多元 probit**（multivariate probit；又称为多变量 probit）的模型，这是对 $C$ 相关的二元结果进行建模的一种方法。