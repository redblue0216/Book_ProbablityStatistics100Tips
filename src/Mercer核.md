# NO.23.tip： Mercer核

## 背景

泛化特性可以归结为我们如何编码关于两个输入向量相似性的先验知识。如果我们知道 $x_i$ 与 $x_j$ 相似，那么就可以引导模型使这两个位置(即 $f(x_i)$ 和 $f(x_j)$)的预测输出相似。

为了定义相似性，我们引入了核函数(kernel function)的概念。"核"一词在数学中有许多不同的含义；此处我们考虑一个 Mercer核，也称为正定核(positive definite kernel)。它是指任何满足以下条件的对称函数 $\mathcal{K}: \mathcal{X} \times \mathcal{X} \to \mathbb{R}^+$：

$$\sum_{i=1}^N \sum_{j=1}^N \mathcal{K}(x_i, x_j) c_i c_j \geq 0 \tag{1}$$

对于 $N$ 个(唯一)点的任何集合 $x_i \in \mathcal{X}$，以及任意数值 $c_i \in \mathbb{R}$ 的选择。我们假设 $\mathcal{K}(x, x) > 0$，因此在上面公式中，只有对于所有 $i$，有 $c_i = 0$ 时，其等式才成立。

理解这种情况的另一种方法如下。给定一组 $N$ 个数据点，让我们将格拉姆矩阵(Gram matrix)定义为以下 $N \times N$ 的相似矩阵：

$$K = \begin{pmatrix} \mathcal{K}(x_1, x_1) & \cdots & \mathcal{K}(x_1, x_N) \\ \vdots & & \vdots \\ \mathcal{K}(x_N, x_1) & \cdots & \mathcal{K}(x_N, x_N) \end{pmatrix} \tag{2}$$

当且仅当对于任何一组(不同的)输入 $\{x_i\}_{i=1}^N$，如果格拉姆矩阵是正定的，我们称 $\mathcal{K}$ 是 Mercer核。

## 源与流

**平稳核**

对于实值输入 $\mathcal{X} = \mathbb{R}^D$，通常使用平稳核(stationary kernel)，也称为移位不变核(shift-invariant kernel)，其形式为 $\mathcal{K}(x, x') = \mathcal{K}(r)$ 的函数，其中 $r = x - x'$；因此输出仅取决于输入之间的相对差。此外，在许多情况下，重要的是差异的大小：

$$r = \|r\|_2 = \|x - x'\| \tag{3}$$

**平方指数核**

平方指数(SE)核，有时也称为指数二次核或径向基函数(Radial Basis Function, RBF)核，其定义为：

$$\mathcal{K}(r; \ell) = \exp\left(-\frac{r^2}{2\ell^2}\right) \tag{4}$$

其中，$\ell$ 对应于核的长度尺度(length-scale)，即我们期望差异产生影响的距离。

根据式(3)，我们可以将该核重写为：

$$\mathcal{K}(x, x'; \ell) = \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right) \tag{5}$$

这是我们前面遇到的径向基函数核，有时也被称为高斯核(Gaussian kernel)。

**ARD核**

我们可以通过用马氏距离代替欧几里得距离，来推广径向基函数核，如下所示：

$$\mathcal{K}(r; \Sigma, \sigma^2) = \sigma^2 \exp\left(-\frac{1}{2} r^\top \Sigma^{-1} r\right) \tag{6}$$

其中，$r = x - x'$。如果 $\Sigma$ 是对角矩阵，则可以记作

$$\mathcal{K}(r; \ell, \sigma^2) = \sigma^2 \exp\left(-\frac{1}{2} \sum_{d=1}^D \frac{1}{\ell_d^2} r_d^2\right) = \prod_{d=1}^D \mathcal{K}(r_d; \ell_d, \sigma^{2/D}) \tag{7}$$

$$\mathcal{K}(r; \ell, \tau^2) = \tau^2 \exp\left(-\frac{1}{2} \frac{1}{\ell^2} r^2\right) \tag{8}$$

我们可以将 $\sigma^2$ 解释为总体方差，将 $\ell_d$ 解释为定义维度 $d$ 的特征长度尺度(characteristic length scale)。如果 $d$ 是不相关的输入维度，我们可以设置 $\ell_d = \infty$，因此相应的维度将被忽略。这被称为自动相关性确定。因此，相应的核被称为 ARD核。

**Matérn**

平方指数核产生了无限可微的函数，因此是非常光滑的。对于许多应用程序，最好使用 Matérn核，这会产生"更粗糙"的函数，这类函数可以更好地对局部"摆动"进行建模，而不必使整体长度尺度非常小。

Matérn核具有以下形式：

$$\mathcal{K}(r; \nu, \ell) = \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}r}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}r}{\ell}\right) \tag{9}$$

其中，$K_\nu$ 是一个修正的贝塞尔函数，$\ell$ 是长度尺度。当且仅当 $\nu > k$ 时，从这个高斯过程采样的函数是 $k$ 次可微的。当 $\nu \to \infty$ 时，该函数将接近平方指数核。

对于值 $\nu \in \{\frac{1}{2}, \frac{3}{2}, \frac{5}{2}\}$，该函数简化如下：

$$\mathcal{K}\left(r; \frac{1}{2}, \ell\right) = \exp\left(-\frac{r}{\ell}\right) \tag{10}$$

$$\mathcal{K}\left(r; \frac{3}{2}, \ell\right) = \left(1 + \frac{\sqrt{3}r}{\ell}\right) \exp\left(-\frac{\sqrt{3}r}{\ell}\right) \tag{11}$$

$$\mathcal{K}\left(r; \frac{5}{2}, \ell\right) = \left(1 + \frac{\sqrt{5}r}{\ell} + \frac{5r^2}{3\ell^2}\right) \exp\left(-\frac{\sqrt{5}r}{\ell}\right)$$

值 $\nu = \frac{1}{2}$ 对应于 Ornstein-Uhlenbeck 过程，该过程描述了粒子经历布朗运动的速度。相应的函数是连续的，但不可微，因此具有非常明显的"锯齿状"。

**周期核**

创建周期性一维随机函数的一种方法是将 $x$ 映射到二维空间 $u(x) = (\cos(x), \sin(x))$，然后在 $u$-空间中使用平方指数核：

$$\mathcal{K}(x, x') = \exp\left(-\frac{2\sin^2((x-x')/2)}{\ell^2}\right) \tag{12}$$

上式之所以成立，是因为 $(\cos(x) - \cos(x'))^2 + (\sin(x) - \sin(x'))^2 = 4\sin^2((x-x')/2)$。我们可以对此进行推广，通过指定周期 $p$ 来得到周期核(periodic kernel)，也称为"指数-正弦-平方"核(exp-sine-squared kernel)：

$$\mathcal{K}_{\text{per}}(r; \ell, p) = \exp\left(-\frac{2}{\ell^2} \sin^2\left(\pi \frac{r}{p}\right)\right) \tag{13}$$

其中，$p$ 是周期，$\ell$ 是长度尺度。

相关的核为余弦核(cosine kernel)：

$$\mathcal{K}(r; p) = \cos\left(2\pi \frac{r}{p}\right) \tag{14}$$

**有理二次核**

我们将有理二次核定义为：

$$\mathcal{K}_{\text{RQ}}(r; \ell, \alpha) = \left(1 + \frac{r^2}{2\alpha\ell^2}\right)^{-\alpha} \tag{15}$$

我们认识到这与学生 $t$ 密度成正比。因此，有理二次核可以被解释为不同特征长度的平方指数核的尺度混合。具体来说，设 $\tau = 1/\ell^2$，并假设 $\tau \sim \text{Ga}(\alpha, \ell^2)$。那么，可以证明

$$\mathcal{K}_{\text{RQ}}(r) = \int p(\tau \mid \alpha, \ell^2) \mathcal{K}_{\text{SE}}(r \mid \tau) \, d\tau \tag{16}$$

当 $\alpha \to \infty$ 时，这简化为平方指数核。

**来自谱密度的核**

考虑满足 $\mathcal{K}(x, x') = \mathcal{K}(\delta)$ 平稳核的情况，其中 $\delta = x - x'$，$x, x' \in \mathbb{R}^d$。让我们进一步假设 $\mathcal{K}(\delta)$ 是正定的。在这种情况下，Bochner 定理表明，我们可以通过傅里叶变换来表示 $\mathcal{K}(\delta)$：

$$\mathcal{K}(\delta) = \int_{\mathbb{R}^d} p(\omega) e^{j\omega^\top \delta} \, d\omega \tag{17}$$

其中，$j = \sqrt{-1}$，$e^{j\theta} = \cos(\theta) + j\sin(\theta)$，$\omega$ 是频率，$p(\omega)$ 是谱密度。

我们可以很容易地从谱密度推导出几个核，并获得直观理解。如果我们对径向基函数核进行傅里叶变换，我们会发现谱密度 $p(\omega) = \sqrt{2\pi\ell^2} \exp(-2\pi^2 \omega^2 \ell^2)$。因此，谱密度也是高斯的，但带宽与长度尺度超参数 $\ell$ 成反比。也就是说，当 $\ell$ 变大时，谱密度塌陷为点质量。这一结果很直观：随着长度尺度的增加，我们的模型将点视为在大距离上相关，它变化缓慢并变得非常平滑，因此是低频的。一般来说，由于高斯分布具有相对较轻的尾部，因此我们可以看到径向基函数核通常不支持高频解。

相反，我们可以使用学生 $t$ 谱密度。学生 $t$ 谱密度具有较重的尾部，将为更高的频率提供更大的支持。对这个谱密度进行傅里叶逆变换，我们可以得到 Matérn核，其自由度对应于谱密度中的自由度。事实上，$\nu$ 值越小，使用 Matérn核对数据进行拟合时得到的曲线就越不平滑，且包含的频率成分也越高。

我们还可以通过将谱密度建模为高斯的"尺度-位置"混合并进行傅里叶逆变换来推导出谱混合核。由于高斯的"尺度-位置"混合在分布集中是稠密的，因此可以近似任何谱密度，所以该核可以任意精度地近似任何平稳核。因此，谱混合核形成了一种强大的核学习方法。

**非平稳核**

平稳核假设两个输入之间的相似性度量与输入的位置无关，即 $\mathcal{K}(x, x')$ 仅取决于 $r = x - x'$。非平稳核(nonstationary kernel)则放宽了这一假设。非平稳核有助于解决各种问题，例如环境建模，其中位置之间的相关性可能会根据环境中的潜在因素而变化。

**多项式核**

非平稳核的一种简单形式是 $M$ 阶多项式核(polynomial kernel)，也称为点积核(dot product kernel)，定义如下：

$$\mathcal{K}(x, x') = (x^\top x')^M \tag{18}$$

这包含所有 $M$ 阶的单项式。例如，如果 $M = 2$，那么我们将得到二次核(quadratic kernel)；在二维空间中，这变成：

$$(x^\top x')^2 = (x_1 x_1' + x_2 x_2')^2 = (x_1 x_1')^2 + (x_2 x_2')^2 + 2(x_1 x_1')(x_2 x_2') \tag{19}$$

通过使用非齐次多项式核(inhomogeneous polynomial kernel)，我们可以将其推广为包含 $M$ 次以下的所有项：

$$\mathcal{K}(x, x') = (x^\top x' + c)^M \tag{20}$$

例如，如果 $M = 2$ 并且输入是二维数据，则我们有

$$(x^\top x' + 1)^2 = (x_1 x_1')^2 + (x_1 x_1')(x_2 x_2') + (x_1 x_1') + (x_2 x_2)(x_1 x_1') + (x_2 x_2')^2 + (x_2 x_2') + (x_1 x_1') + (x_2 x_2') + 1 \tag{21}$$


**吉布斯核**

考虑径向基函数核，其中长度尺度超参数和信号方差超参数都与输入有关；这被称为吉布斯核(Gibbs kernel)，其定义如下：

$$\mathcal{K}(x, x') = \sigma(x)\sigma(x') \sqrt{\frac{2\ell(x)\ell(x')}{\ell(x)^2 + \ell(x')^2}} \exp\left(-\frac{\|x - x'\|^2}{\ell(x)^2 + \ell(x')^2}\right) \tag{22}$$

如果 $\ell(x)$ 和 $\sigma(x)$ 是常数，这就简化为标准的径向基函数核。我们可以通过使用另一个高斯过程来对这些核参数相对于输入的函数依赖性进行建模。

**非向量（结构化）输入的核**

当输入是结构化对象(例如字符串和图形)时，核特别有用，因为通常很难"特征化"可变大小的输入。例如，我们可以定义一个字符串核，该核根据字符串的公共 $n$ 元数量来比较字符串。

我们也可以在图上定义核。例如，随机游走核在概念上同时对两个图执行随机游走，然后计算两个随机游走产生的路径数量。这可以有效地进行计算。

**从旧核中创建新核**

给定两个有效核 $\mathcal{K}_1(x, x')$ 和 $\mathcal{K}_2(x, x')$，我们可以使用以下任何一种方法创建一个新核：

$$\mathcal{K}(x, x') = c \mathcal{K}_1(x, x'), \quad \text{对于任何常量 } c, c > 0 \tag{23}$$

$$\mathcal{K}(x, x') = f(x) \mathcal{K}_1(x, x') f(x'), \quad \text{对于任何函数 } f \tag{24}$$

$$\mathcal{K}(x, x') = q(\mathcal{K}_1(x, x')) \quad \text{对于任何具有非负系数的函数多项式 } q \tag{25}$$

$$\mathcal{K}(x, x') = \exp(\mathcal{K}_1(x, x')) \tag{26}$$

$$\mathcal{K}(x, x') = x^\top A x', \quad \text{对于任何 psd 矩阵 } A \tag{27}$$

例如，假设我们从线性核 $\mathcal{K}(x, x') = x^\top x'$ 开始。我们知道这是一个有效的 Mercer核，因为相应的格拉姆矩阵只是数据的(缩放的)协方差矩阵。从以上规则中，我们可以看出，多项式核 $\mathcal{K}(x, x') = (x^\top x')^M$ 是有效的 Mercer核。

我们也可以使用上述规则来确定高斯核是有效的核。为了理解这一点，需要注意的是：

$$\|x - x'\|^2 = x^\top x + (x')^\top x' - 2x^\top x' \tag{28}$$

因此，

$$\mathcal{K}(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right) = \exp\left(-\frac{x^\top x}{2\sigma^2}\right) \exp\left(\frac{x^\top x'}{\sigma^2}\right) \exp\left(-\frac{(x')^\top x'}{2\sigma^2}\right) \tag{29}$$

是有效的核。

我们还可以使用加法或乘法对核进行组合：

$$\mathcal{K}(x, x') = \mathcal{K}_1(x, x') + \mathcal{K}_2(x, x') \tag{30}$$

$$\mathcal{K}(x, x') = \mathcal{K}_1(x, x') \times \mathcal{K}_2(x, x') \tag{31}$$

将两个正定核相乘总是得到另一个正定核。对于每个核的单个属性，这是一种获得各个属性合取的方法。

此外，将两个正定核加在一起总是会得到另一个正定核。对于每个核的单个属性，这是一种获取各个属性析取的方法。

**Mercer定理**

回想一下，任何正定矩阵 $K$ 都可以使用 $K = U^\top \Lambda U$ 形式的特征分解来表示，其中 $\Lambda$ 是特征值 $\lambda_i > 0$ 的对角矩阵，$U$ 是包含特征向量的矩阵。现在考虑 $K$ 的元素 $(i,j)$：

$$k_{ij} = (\Lambda^{\frac{1}{2}} U_{:i})^\top (\Lambda^{\frac{1}{2}} U_{:j}) \tag{32}$$

其中，$U_{:i}$ 是 $U$ 的第 $i$ 列。如果我们定义 $\phi(x_i) = U_{:i}$，那么我们有：

$$k_{ij} = \sum_{m=1}^M \lambda_m \phi_m(x_i) \phi_m(x_j) \tag{33}$$

其中，$M$ 是核矩阵的秩。因此，我们看到，可以通过执行一些特征向量的内积来计算核矩阵中的各个条目，而这些特征向量由核矩阵的特征向量隐式地定义。

该基本思想可以推广到核函数，而不仅仅是核矩阵，如下文所述。首先，对于核 $\mathcal{K}$，如果该核具有测度 $\mu$ 的特征值 $\lambda$，则可以将核 $\mathcal{K}$ 的特征函数(eigenfunction) $\phi(\cdot)$ 定义为满足以下条件的函数：

$$\int \mathcal{K}(x, x') \phi(x) \, d\mu(x) = \lambda \phi(x') \tag{34}$$

我们通常按特征值递减的顺序对特征函数进行排序，$\lambda_1 \geq \lambda_2 \geq \cdots$。特征函数相对于 $\mu$ 是正交的：

$$\int \phi_i(x) \phi_j(x) \, d\mu(x) = \delta_{ij} \tag{35}$$

其中，$\delta_{ij}$ 是 Kronecker $\delta$ 函数。有了这个定义，我们就可以陈述 Mercer定理。非正式而言，任何正定核函数都可以表示为以下无限和：

$$\mathcal{K}(x, x') = \sum_{m=1}^\infty \lambda_m \phi_m(x) \phi_m(x') \tag{36}$$

其中，$\phi_m$ 是核的特征函数，$\lambda_m$ 是相应的特征值。这是式(33)的函数类比。

退化核(degenerate kernel)只有有限数量的非零特征值。在这种情况下，我们可以将核函数重写为两个有限长度向量之间的内积。例如，考虑式(19)中的二次核 $\mathcal{K}(x, x') = (x^\top x')^2$。如果我们定义 $\phi(x_1, x_2) = [x_1^2, \sqrt{2}x_1x_2, x_2^2] \in \mathbb{R}^3$，那么我们可以将其记作 $\mathcal{K}(x, x') = \phi(x)^\top \phi(x)$。因此我们看到这个核是退化的。

现在考虑径向基函数核。在这种情况下，相应的特征表示是无限维的。然而，通过使用核函数，我们可以避免处理无限维向量。

从上述内容中可以看到，我们可以使用对核函数的调用来替换显式(可能是无限维)特征空间中的内积运算，即我们使用 $\mathcal{K}(x, x')$ 替换 $\phi(x)^\top \phi(x)$。这被称为核技巧(kernel trick)。

**具有随机特征的近似核**

尽管核的强大之处在于避免使用输入的非标准化表示的能力，但这种核化方法可能需要 $O(N^3)$ 的时间来计算格拉姆矩阵 $K$ 的逆。结果使得在大规模数据上使用这种方法变得困难。幸运的是，我们可以使用随机选择的 $M$ 个基函数的有限集来近似许多核的特征图，从而将成本降低到 $O(NM + M^3)$。

我们将通过使用式(17)中的 Bochner 定理来展示如何对移位不变核实现这一点。在高斯径向基函数核的情况下，我们已经看到谱密度是一个高斯分布。因此，我们可以通过对随机高斯向量进行采样来轻松地计算该积分的蒙特卡罗近似值。这就产生了以下的近似：$\mathcal{K}(x, x') \approx \phi(x)^\top \phi(x)$，其中(实值)特征向量由下式给出：

$$\phi(x) = \sqrt{\frac{1}{D}} \left[\sin(z_1^\top x), \ldots, \sin(z_D^\top x), \cos(z_1^\top x), \ldots, \cos(z_D^\top x)\right] \tag{37}$$

$$= \sqrt{\frac{1}{D}} \left[\sin(Z^\top x), \cos(Z^\top x)\right] \tag{38}$$

在上式中，$Z = (1/\sigma)G$，并且 $G \in \mathbb{R}^{d \times D}$ 是随机高斯矩阵，随机高斯矩阵中的条目是从 $\mathcal{N}(0,1)$ 中采样的，并且满足独立同分布。式(38)中的表示被称为随机傅里叶特征(random Fourier features, RFF)或"随机厨房水槽的加权和"(weighted sums of random kitchen sinks)。

可以为其他类型的核创建类似的随机特征表示。然后，我们可以通过定义 $f(x; \theta) = W\phi(x) + b$ 将这些特征用于监督学习，其中 $Z$ 是随机高斯矩阵，$\varphi$ 的形式取决于所选的核。这相当于一层多层感知器，这个多层感知器具有随机"输入到隐藏层"的权重；由于我们只优化"隐藏层到输出"的权重 $\theta = (W, b)$，因此该模型等效于具有固定随机特征的线性模型。如果我们使用足够的随机特征，我们可以近似核化预测模型的性能，但现在的计算成本是 $O(N)$ 而不是 $O(N^2)$。

遗憾的是，随机特性可能会导致比使用非退化核更差的性能，因为随机特性没有足够的表达能力。