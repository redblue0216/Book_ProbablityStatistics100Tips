# NO.18.tip： Fisher信息矩阵

## 背景

&emsp;&emsp;我们将讨论一个称为 **Fisher 信息矩阵** 的重要统计量，该信息矩阵与对数似然函数的曲率有关。这在频率学派统计学中起着关键作用，用于表征最大似然估计的采样分布。然而，Fisher 信息矩阵也用于贝叶斯统计（推导 Jeffreys 的无信息性先验概率），以及优化（作为自然梯度下降处理过程的一部分）。

## 源与流

&emsp;&emsp;**得分函数**（score function；又称评分函数）被定义为对数似然的梯度：

$$
\mathbf{s}(\boldsymbol{\theta}) \triangleq \nabla \log p(\mathbf{x} \mid \boldsymbol{\theta}) \tag{1}
$$

Fisher 信息矩阵（FIM）被定义为得分函数的协方差：

$$
\boldsymbol{F}(\boldsymbol{\theta}) \triangleq \mathbb{E}_{\mathbf{x} \sim p(\mathbf{x} \mid \boldsymbol{\theta})} \left[ \nabla \log p(\mathbf{x} \mid \boldsymbol{\theta}) \nabla \log p(\mathbf{x} \mid \boldsymbol{\theta})^{\mathsf{T}} \right] \tag{2}
$$

&emsp;&emsp;因此，第 $(i,j)$ 项为：

$$
F_{ij} = \mathbb{E}_{\mathbf{x} \sim \boldsymbol{\theta}} \left[ \left( \frac{\partial}{\partial \theta_i} \log p(\mathbf{x} \mid \boldsymbol{\theta}) \right) \left( \frac{\partial}{\partial \theta_j} \log p(\mathbf{x} \mid \boldsymbol{\theta}) \right) \right] \tag{3}
$$

&emsp;&emsp;接下来，我们将对这个统计量进行解释。

**Fisher信息矩阵和负对数似然的黑塞矩阵之间的等价性**

&emsp;&emsp;我们将证明 Fisher 信息矩阵等价于负对数似然的期望黑塞矩阵：

$$
\mathrm{NLL}(\boldsymbol{\theta}) = -\log p(\mathcal{D} \mid \boldsymbol{\theta}) \tag{4}
$$

由于黑塞矩阵衡量了似然概率的曲率，我们看到 Fisher 信息矩阵指出了似然概率函数可以在多大程度上识别最佳参数集。（如果似然函数是平坦的，我们无法推断出任何关于参数的信息，但如果似然函数是单点的 $\delta$ 函数，则最佳参数向量将被唯一确定。）因此，对于最大似然估计的不确定性，似然函数与频率学派的概念密切相关，如果我们在从模型中提取的多个不同数据集上进行计算，则期望最大似然估计中的方差会捕捉到这一点。

&emsp;&emsp;更确切地说，我们有以下定理。

> 定理：如果 $\log p(\mathbf{x} \mid \boldsymbol{\theta})$ 是二阶可微的，则在一定的正则性条件下，Fisher 信息矩阵等于负对数似然的期望黑塞矩阵，即
$$
\boldsymbol{F}(\boldsymbol{\theta})_{ij} \triangleq \mathbb{E}_{\mathbf{x} \sim \boldsymbol{\theta}} \left[ \left( \frac{\partial}{\partial \theta_i} \log p(\mathbf{x} \mid \boldsymbol{\theta}) \right) \left( \frac{\partial}{\partial \theta_j} \log p(\mathbf{x} \mid \boldsymbol{\theta}) \right) \right] = \mathbb{E}_{\mathbf{x} \sim \boldsymbol{\theta}} \left[ \frac{\partial^2}{\partial \theta_i \theta_j} \log p(\mathbf{x} \mid \boldsymbol{\theta}) \right] \tag{5}
$$

&emsp;&emsp;在证明这个定理之前，我们先引入以下重要引理。

> 引理：得分函数的期望值为零。即
$$
\mathbb{E}_{p(\mathbf{x} \mid \boldsymbol{\theta})} \left[ \nabla \log p(\mathbf{x} \mid \boldsymbol{\theta}) \right] = \mathbf{0} \tag{6}
$$

&emsp;&emsp;我们在标量情况下证明了这个引理。首先，请注意，由于 $\int p(\mathbf{x} \mid \boldsymbol{\theta}) \, d\mathbf{x} = 1$，我们有：

$$
\frac{\partial}{\partial \boldsymbol{\theta}} \int p(\mathbf{x} \mid \boldsymbol{\theta}) \, d\mathbf{x} = 0 \tag{7}
$$

&emsp;&emsp;将上式与以下恒等式相结合：

$$
\frac{\partial}{\partial \boldsymbol{\theta}} p(\mathbf{x} \mid \boldsymbol{\theta}) = \left[ \frac{\partial}{\partial \boldsymbol{\theta}} \log p(\mathbf{x} \mid \boldsymbol{\theta}) \right] p(\mathbf{x} \mid \boldsymbol{\theta}) \tag{8}
$$

&emsp;&emsp;我们有：

$$
0 = \int \frac{\partial}{\partial \boldsymbol{\theta}} p(\mathbf{x} \mid \boldsymbol{\theta}) \, d\mathbf{x} = \int \left[ \frac{\partial}{\partial \boldsymbol{\theta}} \log p(\mathbf{x} \mid \boldsymbol{\theta}) \right] p(\mathbf{x} \mid \boldsymbol{\theta}) \, d\mathbf{x} = \mathbb{E}[\mathbf{s}(\boldsymbol{\theta})] \tag{9}
$$

&emsp;&emsp;接下来，我们回到主定理的证明。为了简单起见，我们将主要讨论标量情况。

**证明** 

&emsp;&emsp;求式 (9) 的导数，我们得到：

$$
\begin{align}
0 &= \frac{\partial}{\partial \boldsymbol{\theta}} \int \left[ \frac{\partial}{\partial \boldsymbol{\theta}} \log p(\mathbf{x} \mid \boldsymbol{\theta}) \right] p(\mathbf{x} \mid \boldsymbol{\theta}) \, d\mathbf{x} \notag \\
  &= \int \left[ \frac{\partial^2}{\partial \boldsymbol{\theta}^2} \log p(\mathbf{x} \mid \boldsymbol{\theta}) \right] p(\mathbf{x} \mid \boldsymbol{\theta}) \, d\mathbf{x} + \int \left[ \frac{\partial}{\partial \boldsymbol{\theta}} \log p(\mathbf{x} \mid \boldsymbol{\theta}) \right] \frac{\partial}{\partial \boldsymbol{\theta}} p(\mathbf{x} \mid \boldsymbol{\theta}) \, d\mathbf{x} \tag{10} \\
  &= \int \left[ \frac{\partial^2}{\partial \boldsymbol{\theta}^2} \log p(\mathbf{x} \mid \boldsymbol{\theta}) \right] p(\mathbf{x} \mid \boldsymbol{\theta}) \, d\mathbf{x} + \int \left[ \frac{\partial}{\partial \boldsymbol{\theta}} \log p(\mathbf{x} \mid \boldsymbol{\theta}) \right]^2 p(\mathbf{x} \mid \boldsymbol{\theta}) \, d\mathbf{x} \tag{11}
\end{align}
$$

&emsp;&emsp;因此，可以证得：

$$
-\mathbb{E}_{\mathbf{x} \sim \boldsymbol{\theta}} \left[ \frac{\partial^2}{\partial \boldsymbol{\theta}^2} \log p(\mathbf{x} \mid \boldsymbol{\theta}) \right] = \mathbb{E}_{\mathbf{x} \sim \boldsymbol{\theta}} \left[ \left( \frac{\partial}{\partial \boldsymbol{\theta}} \log p(\mathbf{x} \mid \boldsymbol{\theta}) \right)^2 \right] \tag{12}
$$

&emsp;&emsp;现在给定 $N$ 个独立同分布的样本 $\mathcal{D} = \{ \mathbf{x}_n : n = 1:N \}$，考虑该样本的负对数似然的黑塞矩阵：

$$
H_{ij} \triangleq -\frac{\partial^2}{\partial \theta_i \theta_j} \log p(\mathcal{D} \mid \boldsymbol{\theta}) = -\sum_{n=1}^{N} \frac{\partial^2}{\partial \theta_i \theta_j} \log p(\mathbf{x}_n \mid \boldsymbol{\theta}) \tag{13}
$$

&emsp;&emsp;根据上述定理，我们得到：

$$
\mathbb{E}_{p(\mathcal{D} \mid \boldsymbol{\theta})} \left[ \boldsymbol{H}(\mathcal{D}) \mid_{\boldsymbol{\theta}} \right] = N \boldsymbol{F}(\boldsymbol{\theta}) \tag{14}
$$

这在推导最大似然估计的采样分布时很有用。