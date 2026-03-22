### 1. STUDY BASICS
*   **Time period (start–end):** April 24, 2005, to July 2024.
*   **Total trading days:** 4,504.
*   **Index/stocks analyzed:** NEPSE composite index.
*   **Data frequency:** Daily closing index.
*   **Return formula:** $R_t = \ln(P_t / P_{t-1}) \times 100$.
*   **Data source:** NEPSE website.
*   **Statistical methods:** Descriptive statistics, OLS regression with dummy variables, EGARCH (1,1) estimation, nonparametric Kruskal–Wallis H Test, ANOVA (one-way), Post Hoc (Tukey’s) test, Unit-Root (ADF) test, Jarque-Bera (JB) test, Mann–Whitney U tests.
*   **Software used:** **NOT REPORTED**.

### 2. WEEKDAY RETURN STRUCTURE
| Weekday | Mean % | Median % | Std Dev % | Skewness | Kurtosis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Sunday | -0.0402 | -0.1446 | 1.5824 | 0.346 | 2.13 |
| Monday | -0.024 | -0.0396 | 1.3218 | 0.132 | 3.587 |
| Tuesday | 0.0674 | -0.0081 | 1.3296 | 0.08 | 3.607 |
| Wednesday | 0.1284 | 0.0322 | 1.2369 | 0.568 | 3.444 |
| Thursday | 0.1192 | 0.0127 | 1.0749 | 0.613 | 4.53 |
| **Overall** | **0.0502** | **-0.0208** | **1.3202** | **0.288** | **3.433** |
(Source:)

*   **Min return:** Sunday: -7.2281; Monday: -6.2262; Tuesday: -6.2056; Wednesday: -5.1592; Thursday: -6.2052.
*   **Max return:** Sunday: 5.8659; Monday: 5.831; Tuesday: 5.8846; Wednesday: 5.836; Thursday: 5.6972.
*   **Range (max - min):** Sunday: 13.0940; Monday: 12.0572; Tuesday: 12.0902; Wednesday: 10.9952; Thursday: 11.9024 (Extracted from Min/Max data).

### 3. DISTRIBUTION & RISK PROFILE
*   **% positive days:** **NOT REPORTED**.
*   **% negative days:** **NOT REPORTED**.
*   **% zero days:** **NOT REPORTED**.
*   **Avg gain (positive days):** **NOT REPORTED**.
*   **Avg loss (negative days):** **NOT REPORTED**.
*   **Largest gain %:** 5.8846% (Tuesday).
*   **Largest loss %:** -7.2281% (Sunday).

### 4. STATISTICAL SIGNIFICANCE
*   **p-values for weekday returns (OLS compared to Sunday):** Monday: 0.7953; Tuesday: 0.0842; Wednesday: 0.0068; Thursday: 0.0105.
*   **Significant weekdays (p < 0.05):** Wednesday and Thursday (OLS); Tuesday, Wednesday, and Thursday (EGARCH mean equation).
*   **Pairwise weekday comparisons:** Wednesday vs. Sunday (p=0.043) and Thursday vs. Sunday (p=0.078) in Post Hoc. Sunday vs. Tuesday (p=0.001), Sunday vs. Wednesday (p=0.000), and Sunday vs. Thursday (p=0.000) in Mann-Whitney.
*   **F-statistic (ANOVA):** 3.217 (P-value = 0.012).
*   **Regression coefficients (OLS, Sunday base):** Constant: -0.0402; Monday: -0.0160; Tuesday: 0.1080; Wednesday: 0.1690; Thursday: 0.1590.

**Answer:**
*   **Is weekday effect significant overall (Yes/No)?** **Yes**.
*   **Strength of evidence:** **Strong** (Confirmed across OLS, EGARCH, ANOVA, and non-parametric tests).

### 5. VOLATILITY
*   **Highest volatility weekday:** **Sunday** (Std Dev: 1.5824%).
*   **Lowest volatility weekday:** **Thursday** (Std Dev: 1.0749%).
*   **Any clustering pattern:** **Yes**, confirmed by highly significant ARCH (α) and GARCH (δ) coefficients.
*   **Any ARCH/GARCH findings:** ARCH (α) = 0.5360 (p=0.0000); GARCH (δ) = 0.8546 (p=0.0000); Leverage effect (γ) = -0.0082 (p=0.4942, **insignificant**).
*   **Risk concentration on specific days:** Volatility is **highest on Sundays** and systematically declines through the week.

### 6. WIN RATE
*   **Sunday/Monday/Tuesday/Wednesday/Thursday positive %:** **NOT REPORTED**.
*   **Risk-adjusted return (Sharpe or proxy):** **NOT REPORTED**.
*   **Return-to-volatility ranking:** **Wednesday** (Highest return/lowest volatility among major return days).

### 7. TRADING IMPLICATIONS
*   **Best day to enter:** **Start of the week** (Sunday/Monday).
*   **Worst day to enter:** **End of the week** (implicitly, given high prices on Wed/Thu).
*   **Best day to exit:** **End of the week** (Wednesday/Thursday).
*   **Worst day to exit:** **Start of the week** (Sunday/Monday).
*   **Safest day (lowest volatility):** **Thursday**.
*   **Most risky day:** **Sunday**.
*   **Explicit strategy statement:** "investors in Nepal... should **buy stocks at the start of the week and sell them by the end**".

### 8. MARKET ANOMALIES
*   **Monday effect:** **Not present** as typically defined; however, negative Sunday returns serve as the local equivalent.
*   **Weekend effect:** **Present** (Negative Sunday and Monday mean returns).
*   **Turn-of-week effect:** **NOT REPORTED**.
*   **Thursday/end-of-week behavior:** **Present** (Significantly higher positive returns on Wednesday and Thursday).
*   **Strength:** **Strong evidence**.

### 9. TIME VARIATION
*   **Changes over time:** **NOT REPORTED** (Study notes literature on "disappearing" effects elsewhere but does not apply this to the NEPSE dataset).
*   **Bull vs bear differences:** **NOT REPORTED** for NEPSE index (discussed only in the context of Japanese literature).
*   **Crisis vs normal period:** **NOT REPORTED**.

### 10. SECTOR / STOCK DIFFERENCES
*   **Sector-level weekday effects:** **NOT REPORTED**.
*   **Specific stocks with patterns:** **NOT REPORTED**.
*   **Sectors amplifying effect:** **NOT REPORTED**.

### 11. LIMITATIONS
*   **Sample size issues:** **None reported** (4,504 observations used).
*   **Biases:** **NOT REPORTED**.
*   **Statistical weaknesses:** Returns series deviates from normal distribution (JB test significance); Low R-squared (0.003 or 0.03%) in OLS.
*   **Survivorship bias:** **NOT REPORTED**.

### 12. CONCLUSION
*   **Full conclusion paragraph:** "The study reveals a **significant trading-day effect in NEPSE stock returns**. The negative mean return of **Sundays is significantly lower than Wednesdays**, providing empirical evidence of the seasonality in weekday returns. This consistent return pattern suggests predictability in market behavior. The findings challenge the weak-form efficiency of NEPSE, as it contradicts the random walk hypothesis. The returns can be forecasted based on weekly trading days. The findings of this study are beneficial for Investors, providing insights into optimal trading days and risk mitigation strategies. For policymakers: Sheds light on market inefficiencies, potentially guiding regulatory improvements. For Academics: Enriches the behavioral finance literature in emerging markets."
*   **Sentences containing % returns or p-values:**
    *   "The results exhibit that **Wednesday and Thursday returns are significantly higher compared to Sunday**."
    *   "The overall F-test (**F-statistic = 3.217, P = 0.012**) confirms a calendar day anomaly."
    *   "Table 6 shows the Post Hoc test, which reveals that **Wednesday returns are significantly higher than Sunday returns (P value = 0.043)**."
    *   "The Kruskal–Wallis test outcomes suggested that **at least one day’s median of returns differs significantly from the others**."

### 13. ALPHA EXTRACTION
*   **Rank weekdays best → worst:** Wednesday (0.1284%) → Thursday (0.1192%) → Tuesday (0.0674%) → Monday (-0.024%) → Sunday (-0.0402%).
*   **Identify:**
    *   **High return / high risk days:** **Wednesday** (highest return) has lower risk (Std Dev 1.2369) than **Sunday** (lowest return, highest risk Std Dev 1.5824).
    *   **Low risk / low return days:** **Thursday** (Lowest volatility 1.0749%) with high return (0.1192%).
    *   **Asymmetric opportunities:** Significant return premium on **Wednesday** (0.1284%) with relatively low volatility compared to the start of the week.

**