For your AI trading system for the Nepal Stock Exchange (NEPSE), the following data has been extracted from the research paper **"Performance Evaluation of Technical Analysis in the Nepalese Stock Market: Implications for Investment Strategies"** by Dipendra Karki et al. (2023).

### 1. STUDY BASICS
*   **Time period of data used:** 1st September 2012 to 31st August 2022.
*   **Sample size:** 10 years of daily closing index data, totaling **2,294 observations** per index.
*   **Stocks or index tested:** The NEPSE Index (Composite) and six sub-indices: Commercial Bank, Development Bank, Hydropower, Life Insurance, Non-life Insurance, and Finance.

### 2. TECHNICAL INDICATORS TESTED
The performance metrics below specifically refer to the **NEPSE Index** results:

| Indicator Name | Exact Parameters | Profitable? | Win Rate | Annualized Return |
| :--- | :--- | :--- | :--- | :--- |
| **SMA** | Fast vs. Slow curve crossover | **Yes** | **53.85%** | **21.33%** |
| **MACD** | EMA 12/26, Signal 9 | **Yes** | **53.62%** | **23.64%** |
| **RSI** | Look-back period (Wilder rules) | **No** | **57.69%** | **-4.81%** |
| **Stochastic** | %K (look-back), %D (3-day EMA of %K) | **Yes** | **47.40%** | **44.00%** |
| **Bollinger Bands** | SMA (n-days) ± k * Sigma | **Yes** | **55.56%** | **18.05%** |

*   **Best performing parameter setting:** The study utilized standard default settings (e.g., MACD 12/26/9, RSI 30/70 thresholds) as the basis for evaluation.

### 3. CRITICAL QUESTIONS
*   **RSI Oversold Level:** The study used the standard **30** level.
*   **RSI Overbought Level:** The study used the standard **70** level.
*   **Moving Average Periods:** MACD used **12 and 26-day EMAs** with a **9-day signal line**. SMA was tested using a fast/slow crossover model, though specific day-lengths for "fast" and "slow" were not explicitly tabulated as variables.
*   **Volume Confirmation:** Not directly tested in the backtesting metrics; however, literature review notes that Nepalese investors are highly interested in trade volume indicators.
*   **Did any indicator beat buy-and-hold (B&H) strategy?** **Yes.** SMA (21.33%), MACD (23.64%), and the Stochastic Oscillator (44.00%) all beat the NEPSE B&H return of **16.09%**.
*   **Which indicator performed BEST overall?** The **Stochastic Oscillator**, generating a **44.00%** annualized return on the NEPSE and up to **69.77%** in the Life Insurance sector.
*   **Which indicator performed WORST?** The **Relative Strength Index (RSI)**, which generated a negative return of **-4.81%** on the NEPSE.
*   **Did combining indicators improve performance?** স্ট্যান্ড-অ্যালোন (standalone) models were the focus of the results, but the paper cites that hybrid models and combinations with fundamental analysis generally outperform single indicators in other markets.

### 4. CANDLESTICK PATTERNS
*   **Patterns tested:** This specific paper did not include candlestick patterns in its quantitative backtesting. It notes that most Nepalese investors are "still unaware" of the actual performance of candlestick charts.

### 5. MARKET EFFICIENCY FINDINGS
*   **Is NEPSE weak-form efficient?** **No.** The paper concludes that the Nepalese stock market is **"inefficient at a semi-strong level"**.
*   **Profitable after transaction costs?** The study found technical analysis profitable in its raw backtesting; however, it notes that the "Bootstrap method" (simulating random data) contradicted the predictability, raising questions about whether these profits are robust or due to chance.
*   **Transaction costs used:** While the paper references that technical analysis can beat the market even after transaction costs in other studies, it does not use a specific Nepalese brokerage % fee in its own tabulated return calculations.

### 6. THRESHOLDS AND OPTIMAL VALUES
*   **Optimal Holding Period (NEPSE Index):**
    *   Stochastic Oscillator: **4.41 days**
    *   MACD: **17.16 days**
    *   SMA: **33.31 days**
    *   Bollinger Bands: **130.44 days**
*   **Sector Differences:**
    *   **Stochastic Oscillator** performed best in **Life Insurance** (69.77% return) and **Non-Life Insurance** (65.72% return).
    *   **MACD** performed best in **Life Insurance** (44.09%).
    *   **SMA** performed best in **Life Insurance** (38.51%).
    *   **Hydropower** generally showed the lowest returns across all technical strategies.

### 7. DIRECT QUOTES
**Conclusion Paragraph (Word for Word):**
> "In conclusion, while backtesting the technical trading rules on the Nepalese stock market, the stochastic oscillator provides the best returns. However, it is risky with a low-profit factor and win rates. SMA and MACD also generate substantial profits with good profit factors and win rates, with MACD having the upper hand over SMA. But the bootstrap results discard the predictability power of technical indicators. The two sets of results hence contradict themselves. Nevertheless, this study demonstrates that stock returns generation is likely more complex than it is shown by the research, which employs trading rules and analyzes the returns. Technical limitations may detect hidden patterns, but “how” would remain a question for further studies."

**Table 1: Annualized Return from Different Strategies in Different Sectors**
| Scripts | NEPSE | Com. Bank | Hydro | Dev. Bank | Life-Ins | Non-Life | Finance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B&H return** | 16.09% | 13.91% | 8.85% | 29.55% | 28.70% | 26.79% | 19.32% |
| **SMA return** | 21.33% | 16.50% | 14.90% | 32.32% | 38.51% | 36.77% | 27.34% |
| **MACD return**| 23.64% | 20.13% | 22.68% | 29.80% | 44.09% | 38.80% | 23.97% |
| **RSI return** | -4.81% | -4.79% | -6.49% | 0.60% | -9.95% | -9.39% | -5.50% |
| **Stochastic** | 44.00% | 45.35% | 39.43% | 49.41% | 69.77% | 65.72% | 33.43% |
| **BB return** | 18.05% | 12.90% | 15.03% | 27.83% | 33.24% | 32.92% | 22.64% |

**Table 2: Performance Metrics (Win Rates) - NEPSE**
| Metrics | SMA | MACD | RSI | Stochastic | BB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Wins (%)** | **53.85%** | **53.62%** | **57.69%** | **47.40%** | **55.56%** |
| **No. of Trades**| 39 | 69 | 26 | 327 | 9 |
| **Profit Factor**| 3.86 | 2.97 | 0.66 | 2.56 | 12.19 |
| **Payoff Ratio** | 3.31 | 2.57 | 0.44 | 2.82 | 9.75 |