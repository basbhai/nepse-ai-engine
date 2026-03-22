Based on the provided research papers, here is the structured extraction for each source to assist in building an AI trading system for the Nepal Stock Exchange (NEPSE).

---

### **Source 1: A Study on Relationship Between Stock Market and Economic Growth in Nepal**

**1. STUDY BASICS**
*   **Title, authors, year, journal:** *A Study on Relationship Between Stock Market and Economic Growth in Nepal*; Babita Aryal; 2024; MBS Dissertation, Shanker Dev Campus.
*   **Time period of data used:** Fiscal Year 2014/15 to 2023/24.
*   **Sample size:** 10 years of annual data.
*   **Stock market / index studied:** NEPSE Index (Composite).

**2. VARIABLES TESTED**
*Note: This study reversed the typical model, testing how stock market variables affect GDP (Economic Growth).*
| Variable Name | Significant? | Direction | Coefficient / Correlation | Lag Used |
| :--- | :--- | :--- | :--- | :--- |
| **Number of Listed Securities (NOLS)** | **Yes** (0.013) | **Positive (+)** | Coefficient: **0.016**; Correlation: **0.979** | None |
| **Market Capitalization** | **No** (0.515) | **Negative (-)** | Coefficient: **-0.022**; Correlation: **0.745** | None |
| **Trading Turnover** | **No** (0.511) | **Positive (+)** | Coefficient: **0.006**; Correlation: **0.595** | None |
| **NEPSE Index** | **No** (0.679) | **Positive (+)** | Coefficient: **18.780**; Correlation: **0.596** | None |

**3. KEY FINDINGS**
*   **GDP or economic growth:** The number of listed securities is the only significant predictor of economic growth in this model.
*   **Money supply / liquidity:** Not directly tested.
*   **Interest rate:** Not directly tested.
*   **Other findings:** Market capitalization and turnover reflect market activity but have a non-significant direct impact on GDP in Nepal.

**4. THRESHOLDS OR SPECIFIC VALUES**
*   **Sector-specific findings:** None specified for trading strategies.

**5. METHODOLOGY**
*   **Statistical method:** Descriptive and causal comparative research design; OLS regression; Pearson correlation.
*   **Granger causality / Cointegration:** Mentioned in the literature review but not the primary methodology for results.

**6. DIRECT QUOTES**
*   **Conclusion:** "The results of this study challenge the notion that the stock market is only a speculative casino and offer a more nuanced understanding of the connection between the stock market and economic growth in Nepal. The study... shows that some stock market indicators have a major contribution to the economy, while others do not significantly affect economic growth.".
*   **Coefficient Table (Table 8):** B-values: Constant (166279.998), Market Capitalization (-.022), Turnover (.006), NOLS (.016), NEPSE Index (18.780).

---

### **Source 2: Determinant of Stock Price of Non-Life Insurance Companies**

**1. STUDY BASICS**
*   **Title, authors, year, journal:** *Determinant of Stock Price of Non-Life Insurance Companies*; Bimala Karki; 2024; MBS Dissertation, Shanker Dev Campus.
*   **Time period of data used:** Fiscal Year 2069/70 to 2078/79.
*   **Sample size:** 10 years; 50 observations from 5 companies.
*   **Stock market / index studied:** Non-life insurance sector (SICL, NICL, NIL, SALICO, SPIL).

**2. VARIABLES TESTED**
| Variable Name | Significant? | Direction | Coefficient / Correlation | Lag Used |
| :--- | :--- | :--- | :--- | :--- |
| **Earnings Per Share (EPS)** | **Yes** (<0.01) | **Positive (+)** | Coefficient: **30.008**; Correlation: **0.369** | None |
| **Price Earnings Ratio (P/E)** | **Yes** (<0.01) | **Positive (+)** | Coefficient: **27.182**; Correlation: **0.638** | None |
| **Dividend Per Share (DPS)** | **No** (0.457) | **Negative (-)** | Coefficient: **-2.240**; Correlation: **0.373** | None |
| **Book Value Per Share (BVPS)** | **No** (0.510) | **Positive (+)** | Coefficient: **1.146**; Correlation: **0.424** | None |

**3. KEY FINDINGS**
*   **Internal Drivers:** EPS and P/E Ratio are the major determinants for share price in the non-life insurance sector.
*   **Dividends:** DPS interestingly showed a negative coefficient in regression but a positive Pearson correlation, indicating its effect is weak or mediated.

**4. THRESHOLDS OR SPECIFIC VALUES**
*   **Sector-specific:** Results are specific to **Non-life insurance**.

**5. METHODOLOGY**
*   **Statistical method:** Descriptive research design; Correlation and Multiple Regression using SPSS.

**6. DIRECT QUOTES**
*   **Conclusion:** "The study concludes that earnings per share, price earnings ratio are the major determinants of share price of non-life insurance companies in Nepal.".
*   **Coefficient Table (Table 7):** Constant (-1101.244), Eps (30.008), p/e (27.182), Dps (-2.240), Bvps (1.146).

---

### **Source 3: Determinants of Stock Price in Nepal Stock Exchange**

**1. STUDY BASICS**
*   **Title, authors, year, journal:** *Determinants of Stock Price in Nepal Stock Exchange*; Krishnaji Bhandari; 2024; MBS Dissertation, Shanker Dev Campus.
*   **Time period of data used:** 2068/69 to 2077/78.
*   **Sample size:** 10 years; 100 observations from 10 companies across different sectors.
*   **Stock market / index studied:** Sectoral representation (Commercial Bank, Development Bank, Finance, Life Insurance, Non-life, Hydro, Microfinance, Manufacturing, Investment, Other).

**2. VARIABLES TESTED**
| Variable Name | Significant? | Direction | Coefficient / Correlation | Lag Used |
| :--- | :--- | :--- | :--- | :--- |
| **Price Earnings Ratio (P/E)** | **Yes** (0.000) | **Positive (+)** | Coefficient: **16.995**; Correlation: **0.597** | None |
| **Dividend Per Share (DPS)** | **Yes** (0.000) | **Positive (+)** | Coefficient: **21.112**; Correlation: **0.443** | None |
| **Dividend Yield (DY)** | **Yes** (0.006) | **Negative (-)** | Coefficient: **-33.301**; Correlation: **-0.334** | None |
| **Earnings Per Share (EPS)** | **No** (0.513) | **Positive (+)** | Coefficient: **2.809**; Correlation: **0.290** | None |
| **Return on Equity (ROE)** | **No** (0.276) | **Positive (+)** | Coefficient: **5.653**; Correlation: **0.229** | None |
| **Profit Margin (PM)** | **No** (0.450) | **Negative (-)** | Coefficient: **-0.890**; Correlation: **0.099** | None |
| **Book Value Per Share (BVPS)** | **No** (0.609) | **Positive (+)** | Coefficient: **0.192**; Correlation: **0.056** | None |

**3. KEY FINDINGS**
*   **Money supply / liquidity:** Not directly tested.
*   **Interest rate:** Mentioned in literature as a negative influence but not included in primary model testing.
*   **Internal Determinants:** DPS, P/E, and DY are the primary factors changing stock prices in this multi-sector study.

**4. THRESHOLDS OR SPECIFIC VALUES**
*   **Sector-specific:** Tested 10 different sectors; found multi-sector stability in P/E and DPS as drivers.

**5. METHODOLOGY**
*   **Statistical method:** Descriptive and causal comparative research; multiple regression.

**6. DIRECT QUOTES**
*   **Conclusion:** "In conclusion, the DPS, P/E and DY are major determinants factor for change the stock price. These variables are positive significant effect the market price per share.".
*   **Coefficient Table (Table 9):** Constant (-160.550), PM (-.890), EPS (2.809), P/E (16.995), DPS (21.112), ROE (5.653), BVPS (.192), DY (-33.301).

---

### **Source 4: Economic and Non-Economic Factors Affecting Stock Prices in Nepal**

**1. STUDY BASICS**
*   **Title, authors, year, journal:** *Economic and Non-economic Factors Affecting Stock Prices in Nepal*; Dipendra Karki; 2017; MA Thesis, Patan Multiple Campus.
*   **Time period of data used:** 1994 to 2016.
*   **Sample size:** 23 years of annual data; 2241 daily observations for event study.
*   **Stock market / index studied:** NEPSE Index (Composite).

**2. VARIABLES TESTED**
| Variable Name | Significant? | Direction | Coefficient / Correlation | Lag Used |
| :--- | :--- | :--- | :--- | :--- |
| **Real GDP (RGDP)** | **Yes** (1% level) | **Positive (+)** | Coefficient: **990.02**; Correlation: **0.738** | None |
| **Money Supply (M2)** | **Yes** (1% level) | **Positive (+)** | Coefficient: **16.98**; Correlation: **0.406** | None |
| **Inflation (INF)** | **No** (Full model) | **Positive (+)** | Coefficient: **3.99**; Correlation: **0.401** | None |
| **Interest Rate (IR)** | **No** (Full model) | **Negative (-)** | Coefficient: **-4.72**; Correlation: **-0.501** | None |

**3. KEY FINDINGS**
*   **Interest rate / policy rate:** Significant negative relationship in simple models; insignificant when combined with GDP and M2.
*   **Inflation (CPI):** Responds positively (investors use equity as an inflation hedge), though insignificant in the full model.
*   **GDP or economic growth:** Strong, significant positive impact on NEPSE.
*   **Money supply / liquidity:** Significant positive impact.
*   **Event Reaction:** Stock market reacts significantly to political changes and policy measures.
*   **Signalling effect:** Information leakages cause abnormal returns **3 to 5 days ahead** of announcements.

**4. THRESHOLDS OR SPECIFIC VALUES**
*   **Optimal lag periods:** Optimal lag of **1** based on Schwarz-Bayesian Criteria.
*   **Adjustment Speed:** Market readjusts quickly; effects usually do not last more than **3 to 5 days**.

**5. METHODOLOGY**
*   **Statistical method:** OLS Regression, Event Study (Dummy Variable Regression Model).
*   **Cointegration:** Tested (Engle-Granger). **Result: No cointegration** (p=0.957).

**6. DIRECT QUOTES**
*   **Conclusion:** "The major conclusion of this study is that macroeconomic variables and stock prices are not cointegrated. This shows that the stock prices movements in Nepal are not explained by the macroeconomic variables. In general, the finding of this study supports the random walk hypothesis... the results indicated the important role of social, economic and political events.".
*   **Regression Results (Table 4.3):** Specification V (Full): Intercept (-12741.8), l_RGDP (990.02***), INF (3.99), IR (-4.72), M2 (16.98***)..

---

### **Source 5: Impact of Interest Rate on Stock Market in Nepal**

**1. STUDY BASICS**
*   **Title, authors, year, journal:** *Impact of Interest Rate on Stock Market in Nepal*; Sangam Neupane; 2018; MBS Thesis, Central Department of Management (TU).
*   **Time period of data used:** Mid-July 2003 to mid-July 2017.
*   **Sample size:** 15 years of annual data.
*   **Stock market / index studied:** NEPSE Index (Composite).

**2. VARIABLES TESTED**
| Variable Name | Significant? | Direction | Coefficient / Correlation | Lag Used |
| :--- | :--- | :--- | :--- | :--- |
| **Lending Interest Rate** | **Yes** (0.004) | **Negative (-)** | Coefficient: **-439.409**; Correlation: **-0.583** | None |
| **T-bills Interest Rate** | **Yes** (0.008) | **Negative (-)** | Coefficient: **-142.801**; Correlation: **-0.423** | None |
| **Deposit Interest Rate** | **Yes** (0.029) | **Positive (+)** | Coefficient: **262.838**; Correlation: **-0.193** | None |
| **Bank Rate** | **No** (0.370) | **Negative (-)** | Coefficient: **-126.202**; Correlation: **0.465** | None |

**3. KEY FINDINGS**
*   **Interest rate / policy rate:** Lending rates and T-bill rates have a significant negative impact on NEPSE.
*   **Deposit Rate:** Showed an inverse Pearson correlation but a positive coefficient in multiple regression, suggesting a complex interplay when other rates are held constant.

**4. THRESHOLDS OR SPECIFIC VALUES**
*   **Peak/Trough:** NEPSE highest point in the sample was 1718.2 (2016), lowest 204.86 (2003).

**5. METHODOLOGY**
*   **Statistical method:** Descriptive and analytical; Bivariate Correlation; Multiple Regression.

**6. DIRECT QUOTES**
*   **Conclusion:** "The major conclusion of this study is that deposit rate, lending rate, T-bills rate have significant impact on Share prices (NEPSE Index)... deposit rate, lending rate, T-bills rate have significant negative impact on Share prices... But there is no impact of bank rate on share price.".
*   **Regression Results (Table 4.7):** Intercept (7399.326), W.A Deposit Rate (262.838), W.A Lending Rate (-439.409), Bank Rate (-126.202), T-bill Rate (-142.801)..

---

### **Source 6: Impact of Macroeconomic Variables on Stock Prices in Nepal**

**1. STUDY BASICS**
*   **Title, authors, year, journal:** *Impact of Macroeconomic Variables on Stock Prices in Nepal*; Jharana Shrestha; 2025; MBS Dissertation, Shanker Dev Campus.
*   **Time period of data used:** 1994 to 2024.
*   **Sample size:** 30 years of annual data.
*   **Stock market / index studied:** NEPSE Index (Composite).

**2. VARIABLES TESTED**
| Variable Name | Significant? | Direction | Coefficient / Correlation | Lag Used |
| :--- | :--- | :--- | :--- | :--- |
| **Real GDP (RGDP)** | **Yes** (<0.01) | **Positive (+)** | Coefficient: **1805.04**; Correlation: **0.828** | None |
| **Broad Money Supply (M2)** | **Yes** (Short-run ECM) | **Positive (+)** | Coefficient: **22.098** (Multivariate); Correlation: **0.061** | Lag 1 (ECM) |
| **Inflation (INF)** | **Yes** (Long-run) | **Negative (-)** | Coefficient: **-57.380** (Multivariate); Correlation: **-0.131** | Lag 1 (ECM) |
| **Interest Rate (IR)** | **No** (0.087) | **Positive (+)** | Coefficient: **44.47** (Multivariate); Correlation: **-0.118** | Lag 1 (ECM) |

**3. KEY FINDINGS**
*   **GDP:** Identified as the most important and consistently dependable factor affecting stock values.
*   **Money supply:** Positive and statistically significant impact in the short-run (ECM).
*   **Inflation:** Negative relationship with stock prices, supporting the Fama proxy hypothesis.

**4. THRESHOLDS OR SPECIFIC VALUES**
*   **Optimal lag periods:** Optimal lag of **1** for the VAR model based on Schwarz-Bayesian Criteria.
*   **Adjustment Speed:** Significant negative error correction term (-0.481), indicating a market correction of ~48% of disequilibrium each year.

**5. METHODOLOGY**
*   **Statistical method:** Descriptive and causal comparative; OLS Regression.
*   **Cointegration:** **Confirmed** (Engle-Granger). Result: Stationarity in residuals (p=0.0015).

**6. DIRECT QUOTES**
*   **Conclusion:** "The study concludes that macroeconomic factors in particular, real GDP, interest rates, inflation, and the broad money supply have a significant impact on the dynamics of the Nepali stock market.".
*   **Long-run Model Table (Table 8):** Constant (-18371.39), l_GDP (1905.660), INF (-86.812), IR (56.136), M2 (19.418).

---

### **Source 7: Effects of Interest Rate, Exchange Rate and Volatilities on Stock Price of Nepalese Commercial Banks**

**1. STUDY BASICS**
*   **Title, authors, year, journal:** *Effects of Interest Rate, Exchange Rate and their Volatilities on Stock Price of Nepalese Commercial Banks*; Manoj Shrestha; 2024; Nepalese Journal of Economics.
*   **Time period of data used:** 2013/14 to 2022/23.
*   **Sample size:** 10 years; 140 observations from 14 commercial banks.
*   **Stock market / index studied:** Stock prices and returns of 14 specific Commercial Banks.

**2. VARIABLES TESTED (Dependent: Stock Price)**
| Variable Name | Significant? | Direction | Coefficient / Correlation | Lag Used |
| :--- | :--- | :--- | :--- | :--- |
| **Deposit Interest Rate** | **Yes** (<0.01) | **Negative (-)** | Coefficient: **-0.117**; Correlation: **-0.650** | None |
| **Lending Interest Rate** | **Yes** (<0.01) | **Negative (-)** | Coefficient: **-0.222**; Correlation: **-0.669** | None |
| **Base Rate** | **Yes** (<0.01) | **Negative (-)** | Coefficient: **-0.434**; Correlation: **-0.669** | None |
| **Exchange Rate (USD/NPR)** | **Yes** (<0.01) | **Negative (-)** | Coefficient: **-0.014**; Correlation: **-0.385** | None |
| **Bank Rate** | **No** (Full model) | **Positive (+)** | Coefficient: **0.034**; Correlation: **0.432** | None |
| **Interest Rate Volatility** | **No** (Full model) | **Negative (-)** | Coefficient: **-0.387**; Correlation: **-0.150** | None |

**3. KEY FINDINGS**
*   **Interest rate / policy rate:** Deposit, lending, and base rates have a strong negative impact on bank stock prices.
*   **Exchange rate:** Significant negative effect on Nepalese commercial bank stock prices.
*   **Bank Rate:** Interestingly, showed a positive effect in simple and full regression, differing from typical theory.

**4. THRESHOLDS OR SPECIFIC VALUES**
*   **Sector-specific:** Results are specific to **Commercial Banks**.

**5. METHODOLOGY**
*   **Statistical method:** Panel linear regression; Correlation coefficients matrix.

**6. DIRECT QUOTES**
*   **Conclusion:** "The major conclusion of the study is that changes in lending rates can affect the discount rate used to value future cash flows in stock valuation models... lending interest rate, followed by base rate is the most influencing factor that explains the fluctuation in the stock price and stock return of Nepalese commercial banks.".
*   **Regression Results (Table 4, Model 11):** Intercept (10.04), BR (0.034), DIR (-0.117), LIR (-0.222), BSR (-0.434), VIR (-0.014).




### Source 8: Performance Evaluation of Technical Analysis in the Nepalese Stock Market: Implications for Investment Strategies" by Dipendra Karki et al. (2023).
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
