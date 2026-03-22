# Optimum Portfolio Construction Using Single Index Model
## An Empirical Study of the Nepal Stock Exchange (NEPSE)

**Source:** Khadka, B. K., & Rajopadhyaya, U. (2023). Optimum Portfolio Construction Using Single Index Model: An Empirical Study of Nepal Stock Exchange. *The International Research Journal of Management Science*, Kathmandu University School of Management.

---

## 1. Study Overview

| **Category** | **Details** |
| :--- | :--- |
| **Full Title** | Optimum Portfolio Construction Using Single Index Model: An Empirical Study of Nepal Stock Exchange |
| **Authors** | Bal Krishna Khadka, Dr. Umesh Rajopadhyaya |
| **Year** | 2023 |
| **Journal/Institution** | The International Research Journal of Management Science / Kathmandu University School of Management |
| **Time Period** | July 29, 2018, to August 29, 2021 |
| **Sample Size** | 8 sectorial indices; 674 trading days |
| **Assets Studied** | Overall NEPSE Index (market benchmark) and 8 sectorial indices: Non-Life Insurance, Hydro Power, Finance, Micro Finance, Development Bank, Life Insurance, Banking, and Manufacturing. |
| **Data Frequency** | Daily |
| **Data Sources** | Official website of NEPSE (index data); Nepal Rastra Bank (NRB) website (risk-free rate). |

---

## 2. Single Index Model Parameters by Sector

*Note: Alpha (α), R-squared (R²), and Correlation (r) were not reported in the study. Standard deviation (σ) is derived from the reported variance (σ²) where possible.*

| Sector (Index) | Mean Return (Rᵢ) | Beta (β) | Variance of Error Term (σ²ₑᵢ) | Excess Return to Beta Ratio | Cut-off Rate (Cᵢ) | Optimal Weight (Wᵢ%) | Selected? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Non-Life Insurance** | 0.165 | 0.034 | 3.681 | 2.732 | 0.002 | 17.6% | Yes |
| **Hydro Power** | 0.146 | 0.042 | 3.508 | 1.737 | 0.004 | 14.3% | Yes |
| **Finance** | 0.238 | 0.611 | 1.742 | 0.270 | 0.078 | 36.6% | Yes |
| **Micro Finance** | 0.227 | 0.931 | 1.194 | 0.165 | 0.120 | 20.8% | Yes |
| **Development Bank** | 0.201 | 0.845 | 1.247 | 0.151 | 0.129 | 11.0% | Yes |
| **Life Insurance** | 0.201 | 1.216 | 1.282 | 0.106 | 0.121 | Excluded | No |
| **Banking** | 0.124 | 1.000 | 0.410 | 0.051 | 0.091 | Excluded | No |
| **Manufacturing** | 0.038 | 0.795 | 1.856 | -0.044 | 0.083 | Excluded | No |

---

## 3. Portfolio Construction Methodology

- **Model Formula:** $R_{it} = \alpha_{it} + \beta_{it}R_{mt} + e_{it}$
- **Risk-Free Rate ($R_f$):** Stated as 3.18% per year in the text, but used as 3.8% in calculation tables.
- **Market Benchmark:** Overall NEPSE Index.
- **Market Variance ($\sigma^2_m$):** 1.820
- **Cut-off Rate ($C^*$) Calculation:** The final cut-off rate was determined using the formula:
    $$C=\frac{\sigma_{m}^{2}\sum_{i=1}^{i}\frac{(R_{i}-R_{f})\beta_{i}}{\sigma_{ei}^{2}}}{1+\sigma_{m}^{2}\sum_{i=1}^{i}\frac{\beta_{i}^{2}}{\sigma_{ei}^{2}}}$$
- **Final Cut-off Value ($C^*$):** 0.129
- **Selection Criterion:** Sectors were included if their excess return-beta ratio was greater than the final cut-off rate ($\frac{R_i - R_f}{\beta_i} > C^*$).

---

## 4. Optimal Portfolio Results

The optimal portfolio consists of five sectors with the following weights:

| Sector | Optimal Weight (Wᵢ%) |
| :--- | :--- |
| Finance | 36.6% |
| Micro Finance | 20.8% |
| Non-Life Insurance | 17.6% |
| Hydro Power | 14.3% |
| Development Bank | 11.0% |
| **Total** | **100.0%** |

- **Sectors Excluded:** Life Insurance, Banking, and Manufacturing were excluded because their excess return to beta ratios (0.106, 0.051, and -0.044 respectively) were lower than the cut-off value of 0.129.

*Note: The study did not calculate or report aggregate portfolio metrics such as expected return (Rp), beta (βp), variance (σ²p), or Sharpe ratio.*

---

## 5. Key Analytical Findings

### Beta Analysis
- **Aggressive Sectors (β > 1):** Life Insurance (1.216)
- **Defensive Sectors (β < 1):** Non-Life Insurance, Hydro Power, Finance, Micro Finance, Development Bank, Manufacturing.
- **Highest Beta:** Life Insurance (1.216)
- **Lowest Beta:** Non-Life Insurance (0.034)
- **Average Beta (8 sectors):** 0.659
- **Performance:** High beta did not lead to high risk-adjusted returns. The highest beta sector (Life Insurance) was excluded, while the lowest beta sector (Non-Life Insurance) had the best risk-adjusted performance.

### Risk-Return Findings
- **Best Risk-Adjusted Return:** Non-Life Insurance (Excess return to beta ratio = 2.732)
- **Worst Risk-Adjusted Return:** Manufacturing (-0.044)
- **Unsystematic Risk:** Non-Life Insurance and Hydro Power were identified as the riskiest sectors due to their high unsystematic risk (variance of error term: 3.681 and 3.508, respectively). For Non-Life Insurance, nearly all of its total variance is unsystematic, suggesting significant diversification potential.
- **Diversification Benefit:** The study demonstrates that a diversified optimal portfolio can be constructed by combining just five specific sectors.

### Market Efficiency & Model Applicability
- The study suggests that herding behavior in the Nepalese market causes stock prices to deviate from true value, contradicting the strong form of the Efficient Market Hypothesis.
- The authors conclude that the Single Index Model is a "groundbreaking" and effective method for constructing an optimum portfolio using daily returns from the NEPSE.

---

## 6. Direct Quotes from the Study

> **On Portfolio Composition:**
> "The study identifies five sectors: non-life insurance, microfinance, finance, hydro power, and development banks-as constituting the optimum portfolio. This suggests that, based on the criteria and parameters utilized in the study, these sectors provide a favorable balance between risk and return for investors."

> **On Investment Allocation:**
> "Notably, the finance sector attracts a greater proportion of investment compared to the development bank sector, indicating a potential higher confidence or preference among investors for the finance sector, possibly due to perceived better returns."

> **On the Cut-off Rate:**
> "The cut off value to determine the optimum sector is $C^{*}=0.129$."

> **On Study Limitations:**
> "It's noteworthy that this study excludes five sectors, as well as took the sample of only recent bull period."

---

## 7. Implications for a Trading System

- **Asset Universe:** The analysis is based on 8 major sectorial indices of the NEPSE, which remain active and relevant.
- **Risk Classification:**
    - **Aggressive:** Sectors with β > 1 (e.g., Life Insurance).
    - **Defensive:** Sectors with β < 1 (e.g., Banking).
- **Diversification Strategy:** The findings suggest that optimal diversification across NEPSE sectors can be achieved with a focused portfolio of 5 out of the 8 sectors.
- **Dynamic Ranking Signal:** The core selection mechanism—the excess return to beta ratio compared to a moving cut-off rate ($C^*$)—provides a clear, programmable logic gate for including or excluding sectors in real-time.
- **Risk-Free Rate Input:** The trading system should dynamically source the 364-day Nepal Rastra Bank (NRB) Treasury Bill yield as the $R_f$ variable.
- **Performance Potential:** By excluding underperforming sectors like Manufacturing (negative excess return) and those with poor risk-adjusted metrics (Banking, Life Insurance), the 5-sector weighted portfolio is theoretically positioned to achieve a higher Sharpe ratio than a static equal-weight index.