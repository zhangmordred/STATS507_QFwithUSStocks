# STATS507_QFwithUSStocks

This is the project of STATS 507, Umich.

## Data collection
The training and backtesting data are collected from [Stocks-Daily-Price](https://huggingface.co/datasets/paperswithbacktest/Stocks-Daily-Price) and [Stocks-BalanceSheet](https://huggingface.co/datasets/paperswithbacktest/Stocks-Quarterly-BalanceSheet) from 2010 to 2024. For each stock, the **adjusted** cumulated return and cumulated turnover rate in the setting timestamp will be collected (if training model by return directly, the result may be influenced).

## Model implementation
The model is run in Python 3.8.3 (64-bit), torch version is 2.1.0+cpu and numpy version is 1.23.1. We are not sure if it will work properly under a lower version.

## Backtest
### Trading strategy
Before the first trade date of the timestamp $t$, all sequences $\mathcal{X}^{t} _{n}$ from the stock set $S^t$ are put in the model and the list of outputs $\hat{Y}^t$ is required. The portfolio value at time $t$, denoted as $P^t$ is updated based on the previous period’s weights and returns as follows:
\begin{equation}
P^t = P^{t-1} (\sum^N_{n=1} w^{t-1}_n (1 + r^{t-1}_n))
\label{valueofstrategy}
\end{equation}
where $w^t_n$ is the weight of stock $s^t_n$ and $r^t_n$ is the return of the stock and $\sum^N_{n=1} w^t_n = 1$. To determine the weight of each stock, a trading strategy $\Phi \in \mathbb{R}^{\varrho}$ and the decision factor $\mathbf{b}$ is used at the strategy, where $\mathbf{b}$ determines how many quantile groups are selected, satisfying $ 1 \leq \mathbf{b} < \varrho$. The strategy does not allow short-sell, so $\Phi$ only contains $0$ and $1$. 
$$
\Phi \in \{0,1\}^{\varrho}, \quad ||\Phi||_0 = \mathbf{b}
$$
For example: 
\begin{itemize}
    \item If $\varrho = 3$, $\mathbf{b} = 1$, then $\boldsymbol{\Phi}$ could be one of: $[1, 0, 0]$, $[0, 1, 0]$, or $[0, 0, 1]$.
    \item If $\mathbf{b} = 2$, then possible values of $\boldsymbol{\Phi}$ include: $[1, 1, 0]$, $[1, 0, 1]$, or $[0, 1, 1]$.
\end{itemize}
Recall the predicted output $\hat{y}^t_n = \{ \hat{y}^t_{n,i} \}^{\varrho}_{i=1}$, we sort $S^t$ by $\Psi(\hat{y}^t_{n,1})$, which is the empirical quantile CDF for $\hat{y}^t_{n,1}$. Similar to (\ref{empirical quantile }), $\Psi(\hat{y}^t_{n,1})$ is computed in this way:
\begin{equation}
\Psi(\hat{y}^t_{n,1}) = \frac{1}{n} \sum^n_{i=1} \mathbf{1}_{\{\hat{y}^t_{i,1} \leq \hat{y}^t_{n,1} \}}
\label{empirical quantile_y}
\end{equation}
The sorted predicted vector $\tilde{y}^t_n$ for stock $s^t_n$ at time t is shown below.Here the stocks are ranked into $\varrho$ parts and each part contains $\varphi \times100\%$ of the stocks, and the parameters $\varrho$ and $\varphi$ satisfy $\varphi\varrho \leq 1$:
$$
    \tilde{y}^t_n =[\mathbf{1}_{\{ (i-1)(\varrho+\xi) \leq \Psi(\hat{y}^t_{n,1}) < i\varrho\ + (i-1)\xi)}  \vert i = 1, \dots, \varrho] \in \mathbb{R}^{\varrho}
$$
Then the weight for each selected stock is computed by (\ref{weight}) shown below, and all the chosen stocks are equal-weighted in this way:
\begin{equation}
w^t_n = \frac{1}{\mathbf{b} \cdot \varrho } (\tilde{y}^{t-1}_n  \Phi^T  \tilde{y}^t_n )^T \cdot \mathbf{1}
\label{weight}
\end{equation} 
The backtest starts from January 2020, in other words, the result of the sequences from May 2018 to December 2019 will be used as the first stock pool to trade.


### Other Settings
**Transaction fee**: 0.3% for each time long or short

**Trading period**: 01/2020-12/2024

## Further collaboration or questions
We are willing to collaborate and discuss this topic with those interested. If you want to further connect, you can contact the corresponding author via the paper in ArXiv by mail [zhangzf@umich.edu](mailto:zhangzf@umich.edu). 
