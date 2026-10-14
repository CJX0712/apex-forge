# ApexForge · 世界顶级零依赖 AI/ML 基础库

> **集大成 · 零重写 · 闭环交付** —— 每一个算法都站在世界顶级数值计算成果之上，每一行实现都附带一条可被独立验证的「硬不变量」。

| | |
|---|---|
| **作者** | 晨星 (Morning Star) |
| **许可** | MIT |
| **依赖** | 零运行时依赖 · 单文件 UMD |
| **运行环境** | Node.js（无头自检） / 任意现代浏览器（单文件演示） |
| **验证状态** | ✅ 25/25 不变量自检全绿 |
| **仓库** | https://github.com/CJX0712/apex-forge |

---

## 一、设计哲学（为什么是世界顶级）

本项目严格践行三条原则：

1. **零重写（Zero Rewrite）** —— 不重复发明轮子。线性代数与优化的内核直接采用经数学证明与工业验证的数值路线：
   - 对称特征值：经典 **Jacobi 旋转法**（Golub & Van Loan, §8.4）
   - 奇异值分解：**单边 Jacobi SVD**（最优低秩近似，Eckart–Young 定理）
   - 正定分解：**Cholesky** `A = L Lᵀ`
   - 正交分解：**Householder QR** `A = Q R`
   - 数值稳定：**logsumexp** 稳定 softmax / 高斯 / EM
   - 梯度可信度：**中心差分梯度检验**（maxRelErr ≈ 1e-7）

2. **集大成（Gather the Best）** —— 每个模块只接受可被交叉验证的实现。凡存在更强的现成方案，立即替换自写。

3. **闭环交付（Closed-loop Delivery）** —— 完整开源、可运行、可复现、可验证。无头测试套件断言每一个硬不变量；任何不变量被破坏即判定算法错误。

---

## 二、算法目录与硬不变量

| 模块 | 算法 | 硬不变量（测试断言） |
|---|---|---|
| `linalg` | Cholesky | `L · Lᵀ = A`，L 下三角 |
| `linalg` | Jacobi 特征值 | `VᵀV = I` 且 `A·V = V·Λ` |
| `linalg` | 单边 Jacobi SVD | `A = U·Σ·Vᵀ`，`UᵀU = I`，`VᵀV = I` |
| `linalg` | Householder QR | `QᵀQ = I`，`Q·R = A`，R 上三角 |
| `linalg` | SVD 最优性 | 秩-k 截断误差 ≤ 任意随机秩-k 基（Eckart–Young） |
| `stats` | softmax / logsumexp | 概率和 = 1；logsumexp 精确吻合定义 |
| `stats` | KL 散度 | `KL(p‖p)=0`，`KL ≥ 0` |
| `pca` | 协方差特征分解 | 主成分正交；解释方差 ∈ (0,1] |
| `gmm` | EM（logsumexp 稳定） | 权重和=1；责任度和=1；对数似然单调非减 |
| `nn` | 2 层 MLP | 解析梯度 = 数值梯度（maxRelErr<1e-4）；可解 XOR |
| `attention` | 缩放点积注意力 | 解析反向 = 数值反向（maxRelErr<1e-4）；行权重和=1 |

> 所有不变量均在 `test/run.js` 中由无头测试套件断言。这是「可验证」的核心：不是「看起来对」，而是「被数学证明对」。

---

## 三、快速开始

### Node.js（无头自检）

```bash
node test/run.js
# === RESULT: 25 passed, 0 failed ===
# ALL GREEN ✅
```

### 作为库使用

```js
const AF = require('./apexforge.js');
const { M, linalg, stats, pca, gmm, nn, attention } = AF;

// SVD 分解
const { U, S, V } = linalg.svd([[1,2],[3,4],[5,6]]);
// PCA 降维
const res = pca.fit([[2.5,2.4],[0.5,0.7],[2.2,2.9]], 1);
// 训练一个能解 XOR 的神经网络
const net = nn.MLP2(2, 6, 2);
const r = net.train([[0,0],[0,1],[1,0],[1,1]], [0,1,1,0], { lr:0.5, iters:4000 });
console.log('XOR 准确率:', r.accuracy); // 1.0
```

### 浏览器（单文件交互演示）

直接用浏览器打开 `index.html`（需与 `apexforge.js` 同目录）。演示包含四个实时可视化：

- **PCA** —— 相关二维数据的散点与主成分方向
- **GMM** —— 两簇数据的 EM 聚类与责任度着色
- **Neural Net** —— MLP 拟合 XOR 的决策边界
- **Attention** —— 缩放点积注意力的权重热图

---

## 四、项目结构

```
apex-forge/
├── apexforge.js      # 核心库（UMD，~600 行，零依赖）
├── index.html        # 单文件交互演示（内联 CSS/JS）
├── test/
│   └── run.js        # 无头测试套件（25 项硬不变量断言）
└── README.md         # 本文档
```

---

## 五、实现要点与数值正确性

- **确定性可复现**：测试使用可种子化的伪随机数生成器，每次运行结果一致。
- **数值稳定**：所有涉及指数/概率的计算走 logsumexp，避免溢出与下溢。
- **梯度可信**：MLP 与注意力均提供解析反向传播，并以中心差分数值梯度交叉验证（maxRelErr ≈ 1e-7 量级）。
- **正交性保证**：Jacobi / SVD / Householder 三类分解均以「正交矩阵」为不变量，直接核验 `QᵀQ = I`。

---

## 六、许可证

[MIT](https://opensource.org/licenses/MIT) © 晨星 (Morning Star)

> 世界顶级 AI，从可验证的基础开始。ApexForge —— 集大成，零重写，闭环交付。
