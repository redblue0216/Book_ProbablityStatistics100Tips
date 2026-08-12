# NO.16.tip： KL散度

## 背景

&emsp;&emsp;对于在同一空间上定义的两个概率分布 $P$ 和 $Q$，我们将讨论对这两个概率分布进行比较的各种方法。例如，假设分布是根据样本定义的，$\mathcal{X}=\{x_1,\cdots,x_N\}\sim P$ 和 $\mathcal{X}'=\{\tilde{x}_1,\cdots,\tilde{x}_M\}\sim Q$。确定样本是否来自相同的分布被称为**双样本检验**。这可以通过定义一些合适的**散度度量**$D(P,Q)$ 并将其与阈值进行比较来计算。（我们使用术语“散度”而不是距离，因为我们不要求 $D$ 是对称的。）或者，假设 $P$ 是数据的经验分布，$Q$ 是模型导致的分布。我们可以通过将 $D(P,Q)$ 与阈值进行比较来检查模型对数据的近似程度，这被称为**拟合优度**检验。计算一对分布之间的散度主要有以下两种方法：根据这两个分布的差值 $P-Q$或者根据这两个分布的比值 $P/Q$。我们将基于分布的密度比 $r(x)=p(x)/q(x)$ 来比较概率分布。特别是，考虑 $f$-散度，其定义如下：

$$D_f(p\|q) = \int q(x) f\left(\frac{p(x)}{q(x)}\right) dx \tag{1}$$

其中，$f:\mathbb{R}_+\to\mathbb{R}$ 是满足 $f(1)=0$ 的凸函数。根据**詹森不等式**，得出 $D_f(p\|q)\ge 0$，显然 $D_f(p\|p)=0$，因此 $D_f$ 是一个有效的散度。

&emsp;&emsp;从信息论的角度，我们需要一些方法来度量或量化信息本身。假设我们从描述对随机变量信念度的分布开始，称之为 $q(x)$。然后，我们想将信念度更新为一些新的分布 $p(x)$，也许是因为我们已经进行了一些新的测量，或者只是对这个问题思考了更长的时间。我们所寻求的是使用一种数学方法来量化这次更新的幅度，我们将其表示为 $I[p\|q]$。很显然，我们希望这种有效的度量都应满足以下性质（**需求条件**）：

1. **参数的连续性**：如果我们对开始或结束分布进行轻微扰动，该扰动的更新的幅度也会产生类似的较小影响。

2. **非负性**：对于所有 $p(x)$ 和 $q(x)$，$I[p\|q]\ge 0$。

3. **置换不变量**：更新的幅度不应该取决于我们选择元素的顺序。

4. **均匀分布的单调性**：虽然很难表述信念的更新总体上有多大，但在一些特殊情况下，我们有很强的直觉。如果我们的信念从 $N$ 个元素中的均匀分布更新为 $N'$ 个元素中的均匀分布，那么信息增益应该是 $N$ 的递增函数和 $N'$ 的递减函数。

5. **满足自然链式法则**：如果我们将 $x$ 划分为两个部分 $[x_L, x_R]$，这样就可以记作 $p(x) = p(x_L|x_R)p(x_R)$。类似地，对于 $q$，更新的幅度应该为：

$$I[p(x)\|q(x)] = I[p(x_L)\|q(x_L)] + \mathbb{E}_{p(x_L)}\left[I[p(x_R|x_L)\|q(x_R|x_L)]\right]$$

请注意，这个要求打破了两个分布之间的对称性：等式右侧要求我们取关于边缘的条件概率，如果我们在所有分布中保持顺序一致，那么应该得到相同的答案。

## 源与流

&emsp;&emsp;我们现在将定义一个量，该量是满足上述需求条件的唯一度量（只差一个乘法常数）。

**Kullback-Leibler 散度**或 **KL 散度**，也称为**信息增益**或**相对熵**，其定义如下：

$$D_{\text{KL}}(p\|q) \triangleq \sum_{k=1}^K p_k \log\frac{p_k}{q_k} \tag{2}$$

这自然延伸到连续分布：

$$D_{\text{KL}}(p\|q) \triangleq \int dx\, p(x) \log\frac{p(x)}{q(x)} \tag{3}$$

&emsp;&emsp;KL散度具有一些很值得我们讨论的性质。

**KL的单位**

&emsp;&emsp;上面我们说过，我们列出的需求条件决定了 KL 散度，只差一个乘法常数。因为 KL 散度是对数的，并且不同基数的对数在乘法常数之前是相同的，所以计算 KL 散度时，我们对对数底数的选择类似于选择测量信息的单位。如果 KL 散度使用以 2 为底的对数来测量，那么 KL 散度就被称为以**比特**（bit）为测量单位。比特是“二进制数字”（binary digit）的缩写。如果像通常为了数学计算方便所做的那样使用自然对数来测量，那么 KL 散度就被称为以“自然单位”（natural unit）**奈特**（nat）作为测量单位。

为了在系统之间进行转换，我们使用 $\log_2 y = \frac{\log y}{\log 2}$。因此：

$$1\text{ 比特} = \log_2 e \text{ 奈特} \sim 0.693\text{ 奈特} \tag{4}$$

$$1\text{ 奈特} = \frac{1}{\log_2 e}\text{ 比特} \sim 1.44\text{ 比特} \tag{5}$$

**KL散度的不对称性**

&emsp;&emsp;KL 散度中所使用的两个参数具有不对称性。虽然许多人一开始觉得这种不对称性令人困惑，但我们可以看到，这种不对称性源于我们对自然链式法则的要求。当我们将分布分解为条件分布时，我们需要相对于条件变量取一个期望值。这就破坏了两种分布之间的对称性。

在更直观的层面上，我们可以看到，从 $q$ 移动到 $p$ 所需的信息通常与从 $p$ 移动到 $q$ 所需的信息不同。例如，考虑两个伯努利分布之间的 KL 散度，第一个分布的成功概率为 0.443，第二个分布的成功概率为 0.975：

$$D_{\text{KL}} = 0.975\log\frac{0.975}{0.443} + 0.025\log\frac{0.025}{0.557} = 0.692\text{ 奈特}\sim 1.0\text{ 比特} \tag{6}$$

因此，从 $[0.443, 0.557]$ 分布更新到 $[0.975, 0.025]$ 伯努利分布需要一个比特信息。反过来呢？

$$D_{\text{KL}} = 0.443\log\frac{0.443}{0.975} + 0.557\log\frac{0.557}{0.025} = 1.38\text{ 奈特}\sim 2.0\text{ 比特} \tag{7}$$

所以反之需要两个比特的信息，或者说需要两倍的信息才能向另一个方向移动。因此，我们需要两个不同的假设需要选择，将其标记为 $P$ 和 $Q$。我们收集了一些数据。贝叶斯规则的这种推广有时被称为 **Jeffrey 条件化规则**。

**压缩引理**

&emsp;&emsp;KL 散度一个重要的通用结论是**压缩引理**。

> **定理** 对于具有明确定义的 KL 散度的任何分布 $P$ 和 $Q$，以及对于在分布上定义的任何标量函数 $\phi$，以下公式成立：
$$\mathbb{E}_P[\phi] \le \log\mathbb{E}_Q[e^\phi] + D_{\text{KL}}(P\|Q) \tag{8}$$

**证明** 我们知道，任何两个分布之间的 KL 散度都是非负值。考虑以下形式的分布：

$$g(x) = \frac{q(x)}{Z}e^{\phi(x)} \tag{9}$$

其中，配分函数定义如下：

$$Z = \int dx\, q(x)e^{\phi(x)} \tag{10}$$

取 $p(x)$ 和 $g(x)$ 之间的 KL 散度并重新排列，从而得到边界：

$$D_{\text{KL}}(P\|G) = D_{\text{KL}}(P\|Q) - \mathbb{E}_P[\phi(x)] + \log(Z) \ge 0 \tag{11}$$

理解压缩引理的一种方法是，该压缩引理提供了 KL 散度的所谓 **Donsker-Varadhan 变分表示**：

$$D_{\text{KL}}(P\|Q) = \sup_\phi \mathbb{E}_P[\phi(x)] - \log\mathbb{E}_Q[e^{\phi(x)}] \tag{12}$$

在与分布定义在同一域上的所有可能函数 $\phi$ 的空间中，假设上面的所有值都是有限的，KL 散度是实现的上确界。对于任何固定函数 $\phi(x)$，上式的右侧提供了真实 KL 散度的下界。

压缩引理的另一个用途是，该压缩引理提供了一种估计某个函数相对于未知分布的期望的方法。基于该基本思想，压缩引理可以用来为一组所谓的PAC贝叶斯损失界提供能量，该损失界相对于有限训练集的测量损失的真实分布。

**KL散度和指数族**

&emsp;&emsp;同一族的两个指数族分布之间的 KL 散度具有很好的闭合形式，如下所述。

&emsp;&emsp;考虑 $p(x)$，具有自然参数 $\boldsymbol{\eta}$、基本度量 $h(x)$ 和充分统计量 $\boldsymbol{T}(x)$：

$$p(x)=h(x)\exp\left[\boldsymbol{\eta}^{\mathsf{T}}\boldsymbol{T}(x)-A(\boldsymbol{\eta})\right]$$

其中：

$$A(\boldsymbol{\eta})=\log\int h(x)\exp\left(\boldsymbol{\eta}^{\mathsf{T}}\boldsymbol{T}(x)\right)\mathrm{d}x$$

是对数配分函数，$\boldsymbol{\eta}$ 的凸函数。

&emsp;&emsp;来自同一族的两个指数族分布之间的 KL 散度如下：

$$\begin{aligned}
D_{\mathrm{KL}}\left(p(x\mid\boldsymbol{\eta}_1)\,\big\|\,p(x\mid\boldsymbol{\eta}_2)\right)
&=\mathbb{E}_{\boldsymbol{\eta}_1}\left[(\boldsymbol{\eta}_1-\boldsymbol{\eta}_2)^{\mathsf{T}}\boldsymbol{T}(x)-A(\boldsymbol{\eta}_1)+A(\boldsymbol{\eta}_2)\right]\\
&=(\boldsymbol{\eta}_1-\boldsymbol{\eta}_2)^{\mathsf{T}}\boldsymbol{\mu}_1-A(\boldsymbol{\eta}_1)+A(\boldsymbol{\eta}_2)
\end{aligned}$$

其中，$\boldsymbol{\mu}_j\triangleq\mathbb{E}_{\boldsymbol{\eta}_j}[\boldsymbol{T}(x)]$。

**使用Fisher信息矩阵近似KL散度**

&emsp;&emsp;设 $p_{\boldsymbol{\theta}}(x)$ 和 $p_{\boldsymbol{\theta}'}(x)$ 是两个分布，其中 $\boldsymbol{\theta}'=\boldsymbol{\theta}+\boldsymbol{\delta}$。我们可以测量第二个分布在预测分布方面与第一个分布的接近程度（而不是在参数空间中比较 $\boldsymbol{\theta}$ 和 $\boldsymbol{\theta}'$），如下所示：

$$D_{\mathrm{KL}}(p_{\boldsymbol{\theta}}\,\|\,p_{\boldsymbol{\theta}'})=\mathbb{E}_{p_{\boldsymbol{\theta}}(x)}\left[\log p_{\boldsymbol{\theta}}(x)-\log p_{\boldsymbol{\theta}'}(x)\right]\tag{13}$$

让我们使用二阶泰勒级数展开来近似：

$$D_{\mathrm{KL}}(p_{\boldsymbol{\theta}}\,\|\,p_{\boldsymbol{\theta}'})\approx-\boldsymbol{\delta}^{\mathsf{T}}\mathbb{E}\left[\nabla\log p_{\boldsymbol{\theta}}(x)\right]-\frac{1}{2}\boldsymbol{\delta}^{\mathsf{T}}\mathbb{E}\left[\nabla^2\log p_{\boldsymbol{\theta}}(x)\right]\boldsymbol{\delta}\tag{14}$$

由于预期得分函数为零（根据式 (3.44)），第一项消失，因此我们得到：

$$D_{\mathrm{KL}}(p_{\boldsymbol{\theta}}\,\|\,p_{\boldsymbol{\theta}'})\approx\frac{1}{2}\boldsymbol{\delta}^{\mathsf{T}}\boldsymbol{F}(\boldsymbol{\theta})\boldsymbol{\delta}\tag{15}$$

其中，$\boldsymbol{F}$ 是 Fisher 信息矩阵。

$$\boldsymbol{F}=-\mathbb{E}\left[\nabla^2\log p_{\boldsymbol{\theta}}(x)\right]=\mathbb{E}\left[\left(\nabla\log p_{\boldsymbol{\theta}}(x)\right)\left(\nabla\log p_{\boldsymbol{\theta}}(x)\right)^{\mathsf{T}}\right]\tag{16}$$

因此，我们已经证明，使用 Fisher 信息矩阵作为度量，KL 散度近似等于马氏距离（的平方）。

**Bregman散度**

&emsp;&emsp;设 $f:\Omega\to\mathbb{R}$ 是定义在闭凸集上的连续可微严格凸函数 $\Omega$。我们将与 $f$ 相关的 Bregman 散度定义如下：

$$B_f(\boldsymbol{w}\,\|\,\boldsymbol{v})=f(\boldsymbol{w})-f(\boldsymbol{v})-(\boldsymbol{w}-\boldsymbol{v})^{\mathsf{T}}\nabla f(\boldsymbol{v})\tag{17}$$

为了理解这一点，设：

$$\hat{f}_{\boldsymbol{v}}(\boldsymbol{w})=f(\boldsymbol{v})+(\boldsymbol{w}-\boldsymbol{v})^{\mathsf{T}}\nabla f(\boldsymbol{v})\tag{18}$$

是 $f$ 在 $\boldsymbol{v}$ 处的一阶泰勒级数近似。于是，Bregman 散度是与该线性近似的差：

$$B_f(\boldsymbol{w}\,\|\,\boldsymbol{v})=f(\boldsymbol{w})-\hat{f}_{\boldsymbol{v}}(\boldsymbol{w})\tag{19}$$

由于 $f$ 是凸函数，我们有 $B_f(\boldsymbol{w}\,\|\,\boldsymbol{v})\ge 0$，因为 $\hat{f}_{\boldsymbol{v}}$ 是 $f$ 上的线性下界。

&emsp;&emsp;接下来，我们将讨论一些 Bregman 散度的重要特例。

- 如果 $f(\boldsymbol{w})=\|\boldsymbol{w}\|^2$，那么 $B_f(\boldsymbol{w}\,\|\,\boldsymbol{v})=\|\boldsymbol{w}-\boldsymbol{v}\|^2$ 是欧几里得距离的平方。
- 如果 $f(\boldsymbol{w})=\boldsymbol{w}^{\mathsf{T}}\boldsymbol{Q}\boldsymbol{w}$，那么 $B_f(\boldsymbol{w}\,\|\,\boldsymbol{v})$ 是马氏距离的平方。
- 如果 $\boldsymbol{w}$ 是指数族分布的自然参数，并且 $f(\boldsymbol{w})=\log Z(\boldsymbol{w})$ 是对数归一化器，则 Bregman 散度与 Kullback-Leibler 散度相同。

&emsp;&emsp;回想一下，对数配分函数 $A(\boldsymbol{\eta})$ 是一个凸函数。因此，我们可以使用对数配分函数来定义两个分布 $p$ 和 $q$ 之间的 Bregman 散度，如下所示：

$$\begin{align}
B_f(\boldsymbol{\eta}_q\,\|\,\boldsymbol{\eta}_p)
&=A(\boldsymbol{\eta}_q)-A(\boldsymbol{\eta}_p)-(\boldsymbol{\eta}_q-\boldsymbol{\eta}_p)^{\mathsf{T}}\nabla_{\boldsymbol{\eta}_p}A(\boldsymbol{\eta}_p)\tag{20}\\
&=A(\boldsymbol{\eta}_q)-A(\boldsymbol{\eta}_p)-(\boldsymbol{\eta}_q-\boldsymbol{\eta}_p)^{\mathsf{T}}\mathbb{E}_p[\boldsymbol{T}(x)]\tag{21}\\
&=D_{\mathrm{KL}}(p\,\|\,q)\tag{22}
\end{align}$$

其中我们利用了这样一个事实，即对数配分函数的梯度可以计算预期的充分统计量。

事实上，KL 散度是唯一一种既是 Bregman 散度又是 $f$-散度的散度。