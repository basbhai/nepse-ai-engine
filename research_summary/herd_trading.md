Based on the provided research paper, here is the extraction of the requested information:

### 1. STUDY BASICS
*   **Time period, sample size, data frequency:** The study used a **cross-sectional research design**. The sample size consisted of **250 active retail investors** with a minimum of **two years of trading experience**. Data frequency was a single-point-in-time survey rather than time-series market data.
*   **Which stocks/index tested:** The study targeted retail investors active on the **Nepal Stock Exchange (NEPSE)**.
*   **Statistical method used:** The analysis employed **Principal Component Analysis (PCA)**, **correlation analysis**, **ANOVA**, **reliability tests (Cronbach’s alpha)**, **regression models**, and **Structural Equation Modeling (SEM)** using **bootstrapping** to validate mediation effects.

### 2. KEY FINDINGS
*   **Was herding behavior found significant in NEPSE?** **Yes.** Herding behavior was found to significantly influence investment decisions with a coefficient ($\beta$) between **0.351 and 0.428** and a p-value of **0.020 to 0.052**.
*   **During which market conditions was herding strongest?** The paper notes that retail investors in Nepal frequently follow market consensus during **optimistic periods**. It also cites supporting literature stating herding is higher in **bull markets** and during **global crises**.
*   **Was herding stronger in specific sectors?** The provided sources **do not contain data** regarding specific sectors.
*   **Was herding stronger on specific days or periods?** The sources **do not identify** specific days or periods.
*   **What triggers herding?** Herding is triggered by **cognitive biases**, specifically **loss aversion**, **regret aversion**, and **overconfidence**. It is also described as a reaction to **uncertainty** and **perceived safety in numbers**.

### 3. QUANTITATIVE MEASURES
*   **CSAD or CSSD values reported:** These standard market-level herding measures were **not used**; the study instead measured herding behavior using a **5-point Likert scale**.
*   **Any threshold values identified:** The study reports a mean score for herding behavior of **3.29** on a 5-point scale. Model fit indices included **CFI over 0.90** and **RMSEA below 0.08**. 
*   **How strong was the effect? (coefficient values):**
    *   Correlation between herding and investment decision: **r = 0.480** (p < 0.001).
    *   Regression coefficient ($\beta$) for herding on investment decisions: **0.351 – 0.428**.
    *   The model (cognitive biases and herding) accounted for **50% of the variance ($R^2 = 0.50$)** in investment decision-making.

### 4. PRACTICAL IMPLICATIONS
*   **Does herding create predictable price patterns?** The study notes herding **amplifies volatility and speculative bubbles**, but does not specify technical "predictable patterns".
*   **Does herding cause overreaction then reversal?** The paper states herding results in phenomena such as **bubbles and crashes**, implying overreaction and subsequent collapse.
*   **Any specific price deviation % before reversal identified?** The sources **do not identify** a specific percentage.

### 5. CONCLUSION
*   **Conclusion (word for word):** "From the overall study, the analysis of result confirms herding act as the mediating element in the investment decision of Nepalese Stock market. The cognitive bias and herding behavior act as the strong component in the investment decision of Nepalese. Regarding to above finding the different aspects and other psychological aspects of cognitive bias acts as the prevailing elements with mediating factor of herding in investment decision. Due to different restricted access of information, infrastructure limitations and increasing impact of social media, these findings were consistent with research overall the world. However, by experimentally testing a mediation model in an emerging market, herding affects the cognitive bias in the investment decision in the major context of Nepal. Therefore, regulatory monitoring, policy implications required to improve further progress in investment decision. Further studies should examine moderating elements including investor attitude, digital platforms and financial knowledge help to clarify the dynamics of herding behavior in Nepalese stock market".
*   **Specific trading implications mentioned:** The study highlights that herding **increases the impact of cognitive biases** on investment results. It suggests a need for **regulatory monitoring**, **improved regulatory supervision**, and **investor education** to improve market stability.



=======================================================================================================================================================
Title: Trading Day Effect and Volatility Clustering in NEPSE Returns

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

**Rule:**
**Buy on Sunday/Monday** to capture the beginning-of-week low, and **exit on Wednesday/Thursday** to capture the mid-to-end-of-week premium.