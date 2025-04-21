# Quantformer: Transformer-based Strategy for US Stock Market

A PyTorch-based quantitative trading framework by quantformer, a transformer-like model architecture for numerical time-series data, particularly in the US stock market.

## Datasets
Data is sourced from [Hugging Face](https://huggingface.co/datasets/paperswithbacktest):

- `Stocks-Daily-Price`: 24M rows of daily prices from 1962 to 2025 for ~6,644 US stocks.
- `Stocks-Quarterly-BalanceSheet`: ~237k quarterly balance sheet records for those stocks.

These are used to compute features like adjusted daily return and turnover rate.

## Model Methodology

Before the first trade date of the timestamp $t$, all sequences $\mathcal{X}^{t}_{n}$ from the stock set $S^t$ are put into the model and the list of outputs $\hat{Y}^t$ is required. The portfolio value at time $t$, denoted as $P^t$, is updated based on the previous period’s weights and returns as follows:

$$
P^t = P^{t-1} \left( \sum_{n=1}^{N} w^{t-1}_n (1 + r^{t-1}_n) \right)
$$

where $w^t_n$ is the weight of stock $s^t_n$, and $r^t_n$ is the return of the stock. We assume $\sum_{n=1}^{N} w^t_n = 1$.

To determine the weight of each stock, a trading strategy $\Phi \in \mathbb{R}^{\varrho}$ and the decision factor $\mathbf{b}$ are used, where $\mathbf{b}$ determines how many quantile groups are selected, satisfying $1 \leq \mathbf{b} < \varrho$. The strategy does not allow short-selling, so $\Phi$ only contains $0$ and $1$:

$$
\Phi \in \{0,1\}^{\varrho}, \quad ||\Phi||_0 = \mathbf{b}
$$

Examples:
- If $\varrho = 3$, $\mathbf{b} = 1$, then $\Phi$ could be: $[1, 0, 0]$, $[0, 1, 0]$, or $[0, 0, 1]$.
- If $\mathbf{b} = 2$, then possible values of $\Phi$ include: $[1, 1, 0]$, $[1, 0, 1]$, or $[0, 1, 1]$.

The obtained predicted outputs will be sorted by $\Psi$.
```math
\Psi(\hat{y}^{(t)}_{n,1}) = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}_{\{ \hat{y}^{(t)}_{i,1} \leq \hat{y}^{(t)}_{n,1} \}}
```

The sorted label is then shown below:
```math
\tilde{y}^{(t)}_n = \left[ \mathbb{1}_{\{ (i-1)(\varrho+\xi) \leq \Psi(\hat{y}^{(t)}_{n,1}) < i\varrho + (i-1)\xi \}} \right]_{i=1}^{\varrho}
```

We can get the weight of each stock in the stocklist:
```math
w^{(t)}_n = \frac{1}{b \cdot \varrho} \cdot (\tilde{y}^{(t-1)}_n \Phi^T \tilde{y}^{(t)}_n)^T \cdot \mathbf{1}
```

### Other Settings
**Transaction fee**: 0.3% for each time long or short

**Trading period**: 01/2020-12/2024

## ⚙️ Implementation Details
- Framework: PyTorch
- Model: quantformer
- Hidden Dim: 16
- Layers: 6 encoder blocks
- Optimizer: Adam
- Batch Size: 64
- GPUs: RTX 2070, A100

## Backtest Summary (2020 - 2024)
| Year | Strategy Return | Sharpe Ratio | Max Drawdown | 95% VaR |
|------|------------------|---------------|----------------|-----------|
| 2020 | 5.60%            | 1.96          | 0.41%          | 0.27%     |
| 2021 | 7.29%            | 1.72          | 0.43%          | 0.43%     |
| 2022 | 13.51%           | 2.14          | 0.44%          | 0.41%     |
| 2023 | 8.26%            | 1.94          | 0.51%          | 0.39%     |
| 2024 | 3.15%            | 1.17          | 0.71%          | 0.48%     |

-> Lower drawdowns and steady performance during market downturns.

## Further collaboration or questions
We are willing to collaborate and discuss this topic with those interested. If you want to further connect, you can contact the corresponding author via the paper in ArXiv by mail [zhangzf@umich.edu](mailto:zhangzf@umich.edu). 

## 🔗 Links
- [HuggingFace Dataset - Stocks](https://huggingface.co/datasets/paperswithbacktest/Stocks-Daily-Price)
- [Project GitHub Page](https://github.com/zhangmordred/STATS507_QFwithUSStocks)

---
© 2025 | Zhaofeng Zhang | University of Michigan


