Title: Risk Behavior of Different Weekdays in NEPSE
### **1. STUDY BASICS**
*   **Time period and number of trading days studied:** The study utilized **6,209** closing indices from **1997-07-20 to 2024-07-04**.
*   **Which index or stocks were studied:** The **NEPSE index** (day-wise closing indices).
*   **Data frequency:** **Daily** closing indices.
*   **Statistical method used:** **ARIMAX** (Autoregressive Integrated Moving Average with exogenous inputs) and **GARCH** (Generalized Autoregressive Conditional Heteroscedasticity) hybrid models. Risk was assessed using **Value-at-Risk (VaR)** and **Expected Shortfall (ES)** at a **95% confidence level**.

### **2. RETURNS BY WEEKDAY**
The source reports the mean and standard deviation for the **absolute index values**, rather than percentage returns. Additionally, percentage win rates (positive/negative days) were not reported in the paper.
| Weekday | Mean Index Value | Std Dev (Index Points) | Positive Days % | Negative Days % |
| :--- | :--- | :--- | :--- | :--- |
| **Sunday** | 1022.2048 | 713.5928 | Not Reported | Not Reported |
| **Monday** | 861.9567 | 707.5231 | Not Reported | Not Reported |
| **Tuesday** | 873.0018 | 720.3080 | Not Reported | Not Reported |
| **Wednesday** | 874.5441 | 714.9899 | Not Reported | Not Reported |
| **Thursday** | 863.9226 | 710.0149 | Not Reported | Not Reported |

### **3. STATISTICAL SIGNIFICANCE**
*   **Which weekdays had SIGNIFICANTLY different returns? (p-value < 0.05):** **None.** The paper states that coefficients for weekdays as exogenous variables showed "**no much effect**" on the index and that risk was not noticeably different across days.
*   **Which weekday had the HIGHEST average return?** **Monday and Wednesday** (both had the highest positive coefficient of **0.1365** in the ARIMAX-GARCH models).
*   **Which weekday had the LOWEST average return?** **Sunday** (the only day showing a "minor fall" with a negative coefficient of **-0.1189**).
*   **Which weekday had the HIGHEST volatility?** **Thursday** had the highest Value-at-Risk (**21.86019**) and Expected Shortfall (**38.74582**). Tuesday had the highest absolute standard deviation (**720.3080**).
*   **Which weekday had the LOWEST volatility?** **Wednesday** had the lowest Value-at-Risk (**17.86046**). Monday had the lowest absolute standard deviation (**707.5231**).
*   **Was the weekday effect statistically significant overall?** **No**. **Test used:** ARIMAX-GARCH model coefficients and Value-at-Risk (95% quantile of residuals).

### **4. WIN RATE BY WEEKDAY**
*   **% of Sundays positive:** Not Reported.
*   **% of Mondays positive:** Not Reported.
*   **% of Tuesdays positive:** Not Reported.
*   **% of Wednesdays positive:** Not Reported.
*   **% of Thursdays positive:** Not Reported.

### **5. PRACTICAL THRESHOLDS**
*   **Best day to ENTER a trade:** **Monday, Tuesday, or Wednesday** (showed a "noticeable rise" in index values).
*   **Worst day to ENTER a trade:** **Sunday** (showed a "minor fall" in index value).
*   **Best day to EXIT a trade:** **Thursday**, as it carries "**slightly greater**" risk compared to other days.
*   **Any finding on Monday effect or weekend effect?** The index showed a "noticeable rise" on Monday, while Sunday (Nepal's first trading day) showed a "minor fall".
*   **Any finding on Thursday behavior?** Thursday carrying transactions is "**slightly more**" risky in terms of Value-at-Risk and Expected Shortfall compared to other days.

### **6. SECTOR OR STOCK DIFFERENCES**
*   **Did weekday effect differ by sector?** Not mentioned; the study only analyzed the **NEPSE index**.
*   **Any specific stocks mentioned?** **None**.

### **7. CONCLUSION (word for word)**
*   "Regarding different values of NEPSE indices as independent observations indicate that the risk of transactions on Sunday is slightly less, however, when dependency of observations is taken into account by using ARIMAX-GARCH model the risk is nearly same on all days of week except on Thursday on which value-at-risk and expected shortfall are slightly more in comparison to other days of week. Thus it can be concluded that risk of carrying transactions on different days of week in NEPSE market are not noticeably different."
*   "Next, values of value-at-risk on these days, calculated as 95% quantile of residuals of corresponding models are found to be 18.07, 17.86, 18.09, 17.86 and 21.86 on Sunday, Monday, Tuesday, Wednesday and Thursday, respectively."
*   "Similarly, expected shortfall on these respective days, calculated as mean of values below value-at-risk, are found to be 38.23, 38,29, 38.23, 38.28 and 38.74, respectively."
*   "There is no noticeable effect of different days of week on NEPSE index when viewed with the aspect of value-at-risk and expected shortfall."
*   "The p-values of Engle’s ARCH test for indices on different weekdays starting from Sunday to Thursday are observed to be 8.649243e-194, 8.547239e-255, 1.500364e-252, 1.345995e-255 and 1.631318e-253, respectively."



####################################################################################################################################################################################################################

Title: Prediction of NEPSE Index Movement Using Technical Analysis" 

### **1. STUDY BASICS**
*   **Time period and number of observations:** The study covers **6 months** from **December 2022 to May 2023**. It includes **62 survey responses**.
*   **Which index or stocks studied:** The **NEPSE Index**.
*   **Which candlestick patterns were tested:** The paper recognizes near about **42 recognition patterns** but specifically discusses the following formed in the 6-month chart: **Hammer**, **Hanging Man**, **Inverted Hammer**, **Shooting Star**, **Green Marubozu**, **Red Marubozu**, **Evening Star**, and **Morning Star**.
*   **Which other technical indicators were tested alongside candles:** The empirical analysis focuses on **candlesticks** and a **Questionnaire Survey**. While the literature review mentions Moving Average Convergence Divergence (MACD) and RSI, they were not empirically tested in the study's own results section.
*   **Statistical method used:** **Triangulation mixed methods** (quantitative analysis of chart patterns and survey data using **SPSS** and **Microsoft Excel**).

### **2. CANDLESTICK PATTERN RESULTS**
The source **does not provide** a quantitative table detailing win rates, average returns, or sample sizes for individual candlestick patterns. It provides a visual analysis of where these patterns appeared on a chart and their theoretical implications (e.g., a Hammer "indicating that the market will move upward").

The only reported quantitative results are from the **survey regarding the belief** in technical analysis:
| Item | N | Mean | Std. Deviation | Variance |
| :--- | :--- | :--- | :--- | :--- |
| **Prediction of price movement** | 62 | 1.11 | 0.319 | 0.102 |



### **3. BEST AND WORST PATTERNS**
*   **Highest/Lowest win rate and average return:** **Not reported** in the paper.
*   **Which patterns were statistically significant (p < 0.05):** The paper does not apply p-value testing to individual candlestick patterns. It only provides descriptive statistics for the survey of **62 respondents**.
*   **Which patterns should be REMOVED:** The paper does not list specific patterns to remove, but it notes that in the bearish NEPSE market, candlestick patterns produced "**mixed results**" and were sometimes "**not genuine**".

### **4. VOLUME CONFIRMATION**
*   **Did volume confirmation improve pattern win rates?** The paper defines technical analysis as using data such as price movement and **volume**, but it does not report specific volume confirmation results or improvement percentages.
*   **Any specific volume threshold mentioned?** **None.**

### **5. PATTERN PERFORMANCE BY SECTOR**
*   **Results for Banking, Insurance, or Hydropower sectors:** The study focused specifically on the **NEPSE Index**; sector-specific results were **not reported**.

### **6. HOLDING PERIOD**
*   **Holding period tested:** The study mentions that traders can benefit from patterns in the "**very short term**," but it does not specify or test a defined holding period (e.g., 1, 5, or 10 days).
*   **Effect on win rate/Optimal period:** **Not reported.**

### **7. COMBINED SIGNALS**
*   **Combining candlestick + RSI/MACD:** The study **does not report** multi-indicator combination results for the NEPSE index.

### **8. MARKET EFFICIENCY FINDING**
*   **Does the paper conclude candlesticks work in NEPSE?** **Yes**, but with caveats. The author notes it is "**conceivable with the help of technical indicators**" but results were "**inconsistent over the short term**" and "**not certain**" due to the current economic and political situation.
*   **Bootstrap or randomization test done?** **No.**

### **9. CONCLUSION (word for word)**
*   **Conclusion paragraph:** "Predicting price or index movements is one of the most difficult challenges in the stock market of Nepal. This is conceivable with the help of technical indicators, but not certain in the context of the current economic and political situation in Nepal. However, the results of the candlestick pattern have been inconsistent over the short term, as illustrated by the 6-month technical chart of the figures. Since somewhere the candlestick pattern was not genuine, the trader or investor may have to face losses. However, according to data collected via Google Form, the majority of respondents (55 out of 62) agreed with technical analysis used to predict price and index movements in the stock market. The standard deviation suggests that the predictability of prices and indices was moderate. The mean predicts the mean, while standard deviation and variance anticipate the predicted spread or variability around the mean."
*   **Verbatim table:** There is no table showing pattern win rates. The only statistical table provided is as follows:

| **Item** | **N** | **Mean** | **Std. Deviation** | **Variance** |
| :--- | :--- | :--- | :--- | :--- |
| Prediction of price movement | 62 | 1.11 | 0.319 | 0.102 |

**Table 1: Descriptive Statistics of Prediction of Price Movement by Technical Analysis**







##########################################################################################################################################################################################################
Title: *Fiscal Year and Festive Effects on Market Price Movements**" by Rajesh Gurung and Paritosh Subedi, here is the extracted information:

### **1. STUDY BASICS**
*   **Time period and number of years studied:** **9 years**, from **2015 to 2023**.
*   **Which index or stocks studied:** **17 commercial banks** listed at NEPSE. The study also examines correlations with the **NEPSE index** and **inflation**.
*   **Which festivals/events were analyzed:** **Dashain**, **Tihar**, and **Chhath**. The study also considers the **start of the fiscal year** (mid-July) and **national budget announcements**.
*   **Statistical method used:** **Descriptive analysis** (using seasonal indices) and **correlational analysis**.

### **2. FISCAL YEAR EFFECT**
*   **What happens to NEPSE at fiscal year end (mid-July):** Investors face pressure from debt interest payments and income tax obligations, compelling them to liquidate assets, which contributes to **increased selling pressure**. Institutional investors like mutual funds also rebalance portfolios at this time.
*   **Average abnormal return in the last month (Ashad / mid-June to mid-July):** While the paper cites Maharjan (2018) for positive returns in Ashad due to the budget, this study reports a **Seasonal Index of 0.991 for June**.
*   **Average abnormal return in the first month of the new fiscal year:** Stock prices typically **peak** during the month marking the start of the fiscal year (mid-July to mid-August). The **Seasonal Index for July is 1.028**, rising to **1.083 in August**.
*   **Is the fiscal year effect statistically significant?** **Yes.** The paper reveals "significant effects" related to the start of the fiscal year.
*   **Budget announcement effect:** In the lead-up to the start of the fiscal year, uncertainty regarding tax changes can prompt investors to choose liquid assets. Following policy announcements, sentiment shifts to **heightened enthusiasm and optimism**, leading prices to peak.

### **3. FESTIVE EFFECTS**
| Festival | Pre-event Return % | Post-event Return % | Significant? |
| :--- | :--- | :--- | :--- |
| **Dashain** | **+3.5%** | **-2.5%** (combined with Tihar) | Yes |
| **Tihar** | Included in Dashain lead-up | **-2.5%** | Yes |
| **Chhath** | Elevated levels | Not specifically quantified | Yes |

### **4. BUDGET SEASON (May–June)**
*   **Average NEPSE Mean Price in May:** **571.05**.
*   **Average NEPSE Mean Price in June:** **574.68**.
*   **Reaction time:** The paper attributes price peaks in August to the announcement of fiscal and monetary policies made around the mid-July fiscal year start.
*   **Post-budget direction:** Investor sentiment shifts **positively**, resulting in prices typically reaching their **peak** during the period following policy announcements.

### **5. IPO DRAIN EFFECT**
*   The provided source **does not mention** the IPO/FPO drain effect or subscription periods.

### **6. MONTHLY SEASONALITY**
The paper reports seasonal indices based on AD months, but maps them to the Nepali calendar in the text:
| Month (AD) | Corresponding BS Month | Seasonal Index |
| :--- | :--- | :--- |
| **January** | Poush/Magh | 0.996 |
| **February** | Magh/Phagun | 0.982 |
| **March** | Phagun/Chaitra | 0.981 |
| **April** | Chaitra/Baishakh | 0.954 |
| **May** | Baishakh/Jestha | 0.985 |
| **June** | Jestha/**Ashad** | 0.991 |
| **July** | Ashad/**Shrawan** | 1.028 |
| **August** | Shrawan/**Bhadra** | **1.083** (Peak) |
| **September** | Bhadra/Ashwin | 1.074 |
| **October** | Ashwin/Kartik | 1.034 |
| **November** | Kartik/Mangsir | 0.975 |
| **December** | Mangsir/Poush | **0.917** (Lowest) |

### **7. PRACTICAL THRESHOLDS**
*   **Best month for ENTERING positions:** **December** (lowest seasonal index of 0.917) or **April** (index 0.954).
*   **Worst month for ENTERING:** **August** (highest index 1.083).
*   **Dashain rally start:** In the **lead-up** to Dashain, average prices increase by ~3.5%.
*   **Post-Tihar selling pressure:** Trading volumes decrease and momentum slows **after the Dashain and Tihar holidays**, with prices falling ~2.5% below average.

### **8. CONCLUSION (word for word)**
*   **Conclusion paragraph:** "The stock prices of Nepalese commercial banks showed relatively stable average values throughout the year, with seasonal indices revealing a peak in August, followed by increases in October and November, and the lowest prices occurring in December. While the January effect is a well-documented phenomenon in many stock markets, this study revealed that it does not manifest in the Nepalese context. Instead, the fiscal year effect, which occurs between mid-July and mid-August (the month of Shrawan in the Nepali calendar), along with significant festive effects in October and November, plays a crucial role in influencing stock price movements among commercial banks. The absence of the January effect suggests that investors in Nepal do not exhibit the same seasonal trading behaviors observed in other markets, instead, the focus shifts to how local cultural events and the timing of the fiscal year impact investor sentiment and stock performance. However, the prices movement stem from analogous facts affecting January trends worldwide. Investors often divest financial assets to meet financial commitments as the fiscal year-end, typically in mid-July approaches. Nepali businesses and investors commonly settle taxes and other financial obligations before the deadline, resulting in selling pressure in the stock market as they release assets. Similarly, institutional investors like mutual funds often rebalance their portfolios ahead of the fiscal year-end to enhance their financial statements. Further downward pressure on stock prices during the months before the fiscal year is also associated with the uncertainty surrounding fiscal and monetary policy adjustments, particularly tax changes, which can prompt investors to exercise caution and frequently choose liquidity assets. Historical trends and investor sentiment, coupled with herding behavior, also contribute to these periodic declines, solidifying predictable price patterns in the Nepalese stock market. The primary holiday season in Nepal, taking place in October and November (Dashain, Tihar, and Chhath festive holidays), considerably affects stock price patterns for commercial banks. During these periods, the stock prices ascend above average due to the increased investors' trading activities and their optimism. The periodic movements in stock prices emphasize the impact of cultural events on market patterns, which introduces a distinctive aspect to stock price dynamics in Nepal. The stock markets of Nepal have long recognized the influence of calendar and cultural effects, and the Nepali commercial banking industry exhibits a divergence from the traditional efficient market hypothesis. The observed deviation underscores the need for traders to reevaluate their trading strategies".
*   **Festival return sentence:** "In the lead-up to Dashain, average stock prices increase by approximately 3.5%, driven by heightened buying activity and investor optimism. After the Dashain and Tihar holidays, however, prices tend to fall around 2.5% below the average as trading volumes decrease and market momentum slows".
*   **Budget season sentence:** "Based on the Nepalese Bikram Sambhat calendar, research concluded that the genral market and commercial banks have noticeably shown positive returns during month of Ashad (mid-June to mid-July). This could be attributed to the announcement of the national budget during this time, which might increase investor confidence".

################################################################################################################################################################################################################
Title :Bank Specific Factors, Macroeconomic Variables and Market Value of Nepalese Commercial Banks" by Ramesh Poudel, here is the extracted information:

### **1. STUDY BASICS**
*   **Time period, number of banks, number of observations:** The study covers the period from **2016/17 to 2021/22** using the entire population of **20 commercial banks** for a total of **120 observations**.
*   **Which banks were studied:** The entire population of **20 commercial banks** in Nepal.
*   **Dependent variables:** **Market price per share (MPS)**, **price earnings ratio (P/E)**, and **dividend yield ratio (D/Y)**.
*   **Statistical method:** **Panel regression** utilizing the **Ordinary Least Square (OLS)** model.

### **2. BANK-SPECIFIC VARIABLES TESTED**
The following data is extracted from the primary regression model for **Market Value Per Share (MPS)** in Table 4 and the Correlation Matrix in Table 3:
| Variable | Significant? | Direction | Coefficient ($\beta$) | Correlation | p-value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dividend Payout Ratio (DPR)** | **Yes (1%)** | Positive | **0.659** | 0.669** | 0.000 |
| **Return on Equity (ROE)** | **Yes (5%)** | Positive | **0.315** | 0.274** | 0.001 |
| **Book Value Per Share (BVPS)** | **Yes (5%)** | Positive | **0.196** | 0.270** | 0.003 |
| **Loan & Advance Ratio (LAT)** | **Yes (5%)** | **Negative** | **-0.144** | 0.284** | 0.033 |
| **Return on Assets (ROA)** | No (at 5%) | Positive | 0.438 | 0.244** | 0.078 |
| **Overhead Efficiency (OER)** | No | Positive | 0.099 | 0.030 | 0.175 |
| **Operational Self-Sufficiency (OSR)**| No | Positive | 0.071 | 0.184* | 0.386 |
| **Assets Utilization (AUR)** | No | Positive | 0.045 | 0.247** | 0.856 |

*Note: NPL ratio, CAR, and NIM were not included as primary variables in the empirical regression analysis of this study.*

### **3. MACROECONOMIC VARIABLES TESTED**
| Variable | Significant? | Direction | Coefficient ($\beta$) | p-value |
| :--- | :--- | :--- | :--- | :--- |
| **Inflation (IF)** | **Yes (5%)** | **Negative** | **-0.165** | 0.011 |
| **GDP Growth Rate** | No (at 5%) | Negative | -0.126 | 0.107 |

### **4. MOST IMPORTANT FINDINGS**
*   **Strongest bank-specific effect:** **Dividend Payout Ratio (DPR)** had the strongest positive effect on market value (coefficient **0.659**, correlation **0.669**).
*   **Weakest effect:** **Assets Utilization Ratio (AUR)** had the weakest and most insignificant effect (coefficient **0.045**, p-value **0.856**).
*   **NPL ratio:** While not in the regression, the study concludes that high NPLs "can decrease profitability and increase market risk" and that "careful lending areas selection are crucial to mitigate no performing loan risk".
*   **CAR (Capital Adequacy Ratio):** Not empirically tested in this study's primary model.
*   **NIM (Net Interest Margin):** Not empirically tested, but the study suggests banks should "increase net interest income" to increase firm value.
*   **Healthy NIM/NPL levels:** The paper **does not report** specific numerical thresholds for what is considered a "healthy" NIM or a "danger" level for NPLs.

### **5. THRESHOLDS**
*   **NPL, CAR, or CD Ratio thresholds:** No specific percentage thresholds were mentioned as "danger" levels for these metrics.
*   **P/BV fair value range:** The paper **does not provide** a range for "fair value" P/BV. It only reports the average BVPS for the study period was **Rs 178.72**.

### **6. SECTOR IMPLICATIONS**
*   **Buying banks when NPL falls:** The study **supports this**, noting that NPL risk decreases profitability and reduces investor confidence that the bank can provide continuous returns.
*   **Avoiding banks when interest rates rise:** While not directly testing interest rates, the study shows a **significant negative impact of inflation** on stock prices (coefficient **-0.165**) and notes that concerns about "interest rate increases" may contribute to negative market reactions.

### **7. CONCLUSION (word for word)**
*   **Conclusion paragraph:** "The objective of the study is to explore the relationship and influence of bank specific factors, macro- economic variables on market value of Nepalese commercial banks. The research concludes that dividend payout ratio, operating efficiency ratio, return on equity, return on asset, book value per share and loan to total assets ratios are most influencing factors contributing to market value per share. It is also concluded that dividend payout ratio, book value per share, gross domestic product, inflation rate and loan to total assets are the most influencing factors to dividend yield ratio of Nepalese commercial banks. Dividend payout ratio and inflation rate have positive significant influence on price earnings ratio".

*   **Regression Results Table (Verbatim Table 4):**
| **Independent Variables** | **β** | **std. error** | **beta** | **t** | **Sig.** | **Tolerance** | **VIF** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| (Constant) | 749.931 | 244.321 | | 3.069 | 0.003 | | |
| OER | 0.099 | 0.072 | 0.099 | 1.366 | 0.175 | 0.722 | 1.386 |
| OSR | 0.071 | 0.081 | 0.071 | 0.871 | 0.386 | 0.571 | 1.751 |
| AUR | 0.045 | 0.249 | -0.045 | -0.182 | 0.856 | 0.061 | 16.371 |
| ROE | 0.315 | 0.095 | 0.315 | 3.314 | 0.001 | 0.419 | 2.384 |
| ROA | 0.438 | 0.246 | -0.438 | -1.779 | 0.078 | 0.063 | 15.981 |
| DPR | 0.659 | 0.068 | 0.659 | 9.651 | 0.000 | 0.813 | 1.230 |
| BVPS | 0.196 | 0.066 | 0.196 | 2.987 | 0.003 | 0.878 | 1.139 |
| LAT | -0.144 | 0.067 | -0.144 | -2.154 | 0.033 | 0.849 | 1.178 |
| GDP | -0.126 | 0.077 | -0.126 | -1.626 | 0.107 | 0.633 | 1.579 |
| IF | -0.165 | 0.064 | -0.165 | -2.579 | 0.011 | 0.921 | 1.086 |
**R2 =0.583, AdjR2=0.544; F = 199.139; P value=0.000**.


######################################################################################################################################################################################################
Based on the source "Market Reaction to Dividend Announcement: Evidence from Nepalese Stock Market" by Kamal Prakash Adhikari, here is the extracted information:

### **1. STUDY BASICS**
*   **Time period and number of dividend announcements studied:** The study covered **9 years** from **mid-July 2014 to mid-July 2023** (Excerpts also mention mid-July 2013 to 2022). It analyzed **141 valid dividend announcements** (out of 153 identified).
*   **Which stocks or sectors studied:** **17 commercial banks** listed on NEPSE.
*   **Type of dividends studied:** The paper considers **cash and stock (bonus)** dividends, categorizing them into: Dividend Increase (Good News), Dividend Decrease (Bad News), and No Dividend Changed (No News).
*   **Statistical method:** The **market model of the event study method** using **Ordinary Least Squares (OLS)**. It utilized **Average Abnormal Returns (AAR)** and **Cumulative Average Abnormal Returns (CAAR)**.

### **2. PRE-ANNOUNCEMENT EFFECT**
*(Based on the "Good News" / Dividend Increase sub-sample)*
*   **Average abnormal return 10 days before (t-10):** **-0.02%**.
*   **Average abnormal return 5 days before (t-5):** **0.19%**.
*   **Average abnormal return 3 days before (t-3):** **0.11%**.
*   **Average abnormal return 1 day before (t-1):** **-0.01%**.
*   **When does price movementTypically START before announcement?** Positive abnormal returns were observed starting at **day -6** (0.19%), with additional positive returns at days -5, -4, and -3.

### **3. ANNOUNCEMENT DAY AND AFTER**
*(Based on the "Good News" / Dividend Increase sub-sample)*
*   **Average abnormal return on announcement day (t=0):** **0.76%** (Significant at 5% level).
*   **Average abnormal return day +1:** **9.30%** (Highest positive return, significant at 1% level).
*   **Average abnormal return day +3:** **-0.20%**.
*   **Average abnormal return day +5:** **-0.17%**.
*   **Average abnormal return day +10:** **0.07%**.

### **4. BOOK CLOSE DATE EFFECT**
*   **Average return metrics (Week before/on/after Book Close):** **Not Reported.** The study focuses exclusively on the **Dividend Announcement Date** (Event Day 0) and does not analyze price behavior relative to the Book Close date.
*   **Buying/Selling peak relative to book close:** **Not Reported.**

### **5. DIVIDEND TYPE COMPARISON**
*   **Cash dividend vs Bonus share:** **Not Reported.** The study classifies reactions by the *direction* of the dividend change (Increase/Decrease) rather than the *form* of the dividend.
*   **High dividend vs Low dividend:** **Not Reported.**
*   **First-time dividend vs regular dividend:** **Not Reported.**

### **6. SECTOR DIFFERENCES**
*   **Banking sector vs Insurance sector:** **Not Reported.** The study only examined **commercial banks**.
*   **Strongest pre-announcement drift:** Only the banking sector was analyzed; however, the paper notes a positive pre-event drift starting 6 days before an "increase" announcement.

### **7. INFORMATION LEAKAGE**
*   **Evidence of insider trading / leakage?** **Yes.** For dividend increases (Good News), positive abnormal returns at t=-3, -4, -5, and -6 suggest "insider information among a limited group of shareholders". For "No News," a negative CAAR in the pre-event period suggests "selling pressure in the market before the no-dividend change announcement is due to information leakage".
*   **How many days before official announcement does abnormal return start?** Approximately **6 days** before for dividend increases.

### **8. PRACTICAL THRESHOLDS FOR TRADING SYSTEM**
*   **Optimal entry point:** The paper notes a positive drift starting **6 days** before the announcement.
*   **Optimal exit point:** The highest return occurs on **day +1** (9.30%), and the market begins an "adjustment period" where returns remain consistently negative from **day +2 to day +9**.
*   **Minimum dividend yield trigger:** **Not Reported.**

### **9. CONCLUSION (word for word)**
*   **Conclusion paragraph:** "This paper contributes to gaining more knowledge of market efficiency in the Nepalese stock market. It provides valuable insights for Nepalese investors, policymakers, companies and researchers. This study implies that Nepalese stock market is inefficient so investors could get an opportunity for abnormal returns by considering public information like dividend announcements, right offerings, mergers and acquisitions, monetary policy, physical policy, and government change. Furthermore, this research could be explored to expand insights into market dynamics, patterns and projections of superior market performance. Along with this, other researchers could prove or criticize the existing theories of the capital market".
*   **Table: Cumulative Average Abnormal Return of Dividend Increase (Good News)**
| Period | CAAR in percent | t-value | %+ve value | z-value |
| :--- | :--- | :--- | :--- | :--- |
| **(+2, +10)** | -0.16 | -3.17 *** | 0.09 | -19.21 *** |
| **(-5, +5)** | -0.08 | -1.53 | 0.38 | -1.89 |
| **(-3, +3)** | -0.08 | -1.85 * | 0.40 | -3.84 *** |
| **(-1, +1)** | 0.70 | 2.64 ** | 0.44 | -3.46 *** |
| **(-10, -2)** | -0.01 | -0.27 | 0.47 | 3.41 *** |
| **(-10, +10)** | -0.08 | -1.13 | 0.34 | -1.14 |

*   **Sentence with specific day counts:** "As a result, adjustments happened on subsequent day after day two (t = 2) of the event day, price adjusted up to day nine (t = +9)".
