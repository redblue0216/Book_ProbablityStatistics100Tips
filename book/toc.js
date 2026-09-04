// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><a href="简介.html">简介</a></li><li class="chapter-item expanded affix "><a href="条件期望.html">NO.1.tip： 条件期望</a></li><li class="chapter-item expanded affix "><a href="矩估计.html">NO.2.tip： 矩估计</a></li><li class="chapter-item expanded affix "><a href="最大似然估计.html">NO.3.tip： 最大似然估计</a></li><li class="chapter-item expanded affix "><a href="贝叶斯估计.html">NO.4.tip： 贝叶斯估计</a></li><li class="chapter-item expanded affix "><a href="变分推理.html">NO.5.tip： 变分推理</a></li><li class="chapter-item expanded affix "><a href="马尔可夫链蒙特卡洛方法.html">NO.6.tip： 马尔可夫链蒙特卡罗方法</a></li><li class="chapter-item expanded affix "><a href="指数族分布.html">NO.7.tip： 指数族分布</a></li><li class="chapter-item expanded affix "><a href="核密度估计.html">NO.8.tip： 核密度估计</a></li><li class="chapter-item expanded affix "><a href="K近邻密度估计.html">NO.9.tip： K近邻密度估计</a></li><li class="chapter-item expanded affix "><a href="AIC和BIC准则.html">NO.10.tip： AIC和BIC准则</a></li><li class="chapter-item expanded affix "><a href="假设检验.html">NO.11.tip： 假设检验</a></li><li class="chapter-item expanded affix "><a href="Wald检验.html">NO.12.tip： Wald检验</a></li><li class="chapter-item expanded affix "><a href="似然比检验.html">NO.13.tip： 似然比检验</a></li><li class="chapter-item expanded affix "><a href="置信区间.html">NO.14.tip： 置信区间</a></li><li class="chapter-item expanded affix "><a href="EM算法.html">NO.15.tip： EM算法</a></li><li class="chapter-item expanded affix "><a href="KL散度.html">NO.16.tip： KL散度</a></li><li class="chapter-item expanded affix "><a href="马尔可夫链.html">NO.17.tip： 马尔可夫链</a></li><li class="chapter-item expanded affix "><a href="Fisher信息矩阵.html">NO.18.tip： Fisher信息矩阵</a></li><li class="chapter-item expanded affix "><a href="状态空间模型.html">NO.19.tip： 状态空间模型</a></li><li class="chapter-item expanded affix "><a href="卡尔曼滤波.html">NO.20.tip： 卡尔曼滤波</a></li><li class="chapter-item expanded affix "><a href="泊松过程.html">NO.21.tip： 泊松过程</a></li><li class="chapter-item expanded affix "><a href="隐马尔可夫链.html">NO.22.tip： 隐马尔可夫链</a></li><li class="chapter-item expanded affix "><a href="Mercer核.html">NO.23.tip： Mercer核</a></li><li class="chapter-item expanded affix "><a href="高斯过程.html">NO.24.tip： 高斯过程</a></li><li class="chapter-item expanded affix "><a href="诱导点近似.html">NO.25.tip： 诱导点近似</a></li><li class="chapter-item expanded affix "><a href="互信息.html">NO.26.tip： 互信息</a></li><li class="chapter-item expanded affix "><a href="共轭先验.html">NO.27.tip： 共轭先验</a></li><li class="chapter-item expanded affix "><a href="无信息先验.html">NO.28.tip： 无信息先验</a></li><li class="chapter-item expanded affix "><a href="广义线性模型.html">NO.29.tip： 广义线性模型</a></li><li class="chapter-item expanded affix "><a href="线性回归.html">NO.30.tip： 线性回归</a></li><li class="chapter-item expanded affix "><a href="逻辑回归.html">NO.31.tip： 逻辑回归</a></li><li class="chapter-item expanded affix "><a href="Probit回归.html">NO.32.tip： Probit回归</a></li><li class="chapter-item expanded affix "><a href="边界优化.html">NO.33.tip： 边界优化</a></li><li class="chapter-item expanded affix "><a href="贝叶斯优化.html">NO.34.tip： 贝叶斯优化</a></li><li class="chapter-item expanded affix "><a href="随机优化.html">NO.35.tip： 随机优化</a></li><li class="chapter-item expanded affix "><a href="自然梯度下降.html">NO.36.tip： 自然梯度下降</a></li><li class="chapter-item expanded affix "><a href="熵.html">NO.37.tip： 熵</a></li><li class="chapter-item expanded affix "><a href="支持向量机.html">NO.38.tip： 支持向量机</a></li><li class="chapter-item expanded affix "><a href="结构时间序列模型.html">NO.39.tip： 结构时间序列模型</a></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0];
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
