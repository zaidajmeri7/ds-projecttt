# Supplementary Appendix: Methodological, Statistical, and Empirical Foundations

**Accompanying Manuscript:**  
*Beyond Random Splits: Leakage, Imbalance Handling, Calibration, and Temporal Generalization in BRFSS-Based Heart Disease Classification*

---

## Appendix A: Complete Feature Crosswalk and Harmonization Specification

The Behavioral Risk Factor Surveillance System (BRFSS) is administered annually across all 50 U.S. states, the District of Columbia, and three U.S. territories by state health departments in collaboration with the Centers for Disease Control and Prevention (CDC).

The table below provides the full cross-wave mapping for all 17 candidate predictor variables and the primary outcome across the 2020, 2021, and 2022 survey cycles.

| Variable Name | CDC 2020 Name | CDC 2021 Name | CDC 2022 Name | CDC Core Questionnaire Text | Response Code Mapping | Preprocessing & Harmonization |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **_MICHD** (Target) | `_MICHD` (`HeartDisease`) | `_MICHD` | `_MICHD` | Calculated: Ever told had coronary heart disease (angina) or myocardial infarction | `1` = Yes (Positive); `2` = No (Negative) | Target binary status indicator. Removed missing / refused. |
| **BMI** | `_BMI5` | `_BMI5` | `_BMI5` | Body Mass Index (calculated from reported height and weight) | Continuous integer (implied 2 decimal places: e.g., 2750 = 27.50) | Divided by 100.0. Scaled via training-fit `StandardScaler`. |
| **Smoking** | `SMOKE100` | `SMOKE100` | `SMOKE100` | "Have you smoked at least 100 cigarettes in your entire life?" | `1` = Yes; `2` = No | Binary indicator: `Yes` / `No`. One-hot encoded. |
| **AlcoholDrinking** | `_RFDRHV7` | `_RFDRHV7` | `_RFDRHV8` | Heavy drinkers (adult men >14 drinks/wk, women >7 drinks/wk) | `1` = No; `2` = Yes | Binary indicator: `No` / `Yes`. (2022 renamed to `_RFDRHV8` due to ALCDAY4 update; logic identical). |
| **Stroke** | `CVDSTRK3` | `CVDSTRK3` | `CVDSTRK3` | "(Ever told) you had a stroke?" | `1` = Yes; `2` = No | Binary indicator: `No` / `Yes`. One-hot encoded. |
| **PhysicalHealth** | `PHYSHLTH` | `PHYSHLTH` | `PHYSHLTH` | "Now thinking about your physical health, for how many days during the past 30 days was your physical health not good?" | `1-30` = Number of days; `88` = None (0 days) | Mapped `88` $\to 0.0$. Continuous float $[0, 30]$. Scaled via `StandardScaler`. |
| **MentalHealth** | `MENTHLTH` | `MENTHLTH` | `MENTHLTH` | "Now thinking about your mental health, for how many days during the past 30 days was your mental health not good?" | `1-30` = Number of days; `88` = None (0 days) | Mapped `88` $\to 0.0$. Continuous float $[0, 30]$. Scaled via `StandardScaler`. |
| **DiffWalking** | `DIFFWALK` | `DIFFWALK` | `DIFFWALK` | "Do you have serious difficulty walking or climbing stairs?" | `1` = Yes; `2` = No | Binary indicator: `No` / `Yes`. One-hot encoded. |
| **Sex** | `_SEX` / `SEXVAR` | `_SEX` | `_SEX` | Biological sex recorded from respondent interview | `1` = Male; `2` = Female | Binary category: `Male` / `Female`. One-hot encoded. |
| **AgeCategory** | `_AGEG5YR` | `_AGEG5YR` | `_AGEG5YR` | Fourteen-level calculated five-year age categories | `1` = 18-24 ... `13` = 80+, `14` = DK/Refused | Categorical strings ('18-24', ..., '80 or older'). One-hot encoded. |
| **Race** | `_IMPRACE` | `_IMPRACE` | `_IMPRACE` | Six-level CDC imputed race/ethnicity variable | `1` = White, `2` = Black, `3` = Asian, `4` = AI/AN, `5` = Hispanic, `6` = Other | Nominal categorical (6 levels). One-hot encoded. |
| **Diabetic** | `DIABETE4` | `DIABETE4` | `DIABETE4` | "(Ever told) you have diabetes?" | `1` = Yes; `2` = Yes (pregnancy); `3` = No; `4` = Borderline | Categorical (4 levels: `Yes`, `No`, `No borderline`, `Yes (pregnancy)`). One-hot encoded. |
| **PhysicalActivity**| `_TOTINDA` | `_TOTINDA` | `_TOTINDA` | Adults who reported doing physical activity or exercises other than job | `1` = Yes; `2` = No | Binary indicator: `Yes` / `No`. One-hot encoded. |
| **GenHealth** | `GENHLTH` | `GENHLTH` | `GENHLTH` | "Would you say that in general your health is: Excellent, Very good, Good, Fair, Poor" | `1` = Excellent, `2` = Very good, `3` = Good, `4` = Fair, `5` = Poor | Ordinal categorical string levels. One-hot encoded. |
| **SleepTime** | `SLEPTIM1` | *Rotated Off* | `SLEPTIM1` | "On average, how many hours of sleep do you get in a 24-hour period?" | `1-24` = Hours of sleep | Continuous integer $[1, 24]$. In 2021: median-imputed to $7.0$ (sensitivity: NaN routing). In 2022: empirical. Scaled via `StandardScaler`. |
| **Asthma** | `ASTHMA3` | `ASTHMA3` | `ASTHMA3` | "(Ever told) you had asthma?" | `1` = Yes; `2` = No | Binary indicator: `No` / `Yes`. One-hot encoded. |
| **KidneyDisease** | `CHCKDNY2` | `CHCKDNY2` | `CHCKDNY2` | "(Ever told) you have kidney disease (excluding kidney stones or bladder infection)?" | `1` = Yes; `2` = No | Binary indicator: `No` / `Yes`. One-hot encoded. |
| **SkinCancer** | `CHCSCNCR` | `CHCSCNCR` | `CHCSCNC1` | "(Ever told) you had skin cancer?" | `1` = Yes; `2` = No | Binary indicator: `No` / `Yes`. (2022 renamed `CHCSCNC1` due to section order; identical wording). |

---

## Appendix B: Mathematical Formulation of Complex Survey Weighting

The BRFSS employs a multistage, stratified probability sampling design. Simple random sampling (SRS) variance estimation formulas underestimate true sampling variance due to intra-cluster correlation and unequal probability of selection.

### 1. Complex Survey Design Parameters
For respondent $i$ belonging to stratum $h$ ($h = 1, \dots, H$) and primary sampling unit (PSU / cluster) $j$ ($j = 1, \dots, n_h$):
- $w_{hij} = \text{\_LLCPWT}$: final post-stratification sampling weight produced via iterative proportional fitting (raking) to match state-level Census population margins (age, sex, race, education, marital status, home ownership, telephone ownership).
- Stratum identifier: `_STSTR`.
- Cluster / Primary Sampling Unit identifier: `_PSU`.

### 2. Taylor Series Linearization for Ratio Estimators
Population-representative performance metrics (such as disease prevalence, sensitivity, and specificity) are nonlinear ratio estimators of the form:
$$\hat{R} = \frac{\hat{Y}}{\hat{X}} = \frac{\sum_{h=1}^H \sum_{j=1}^{n_h} \sum_{i=1}^{m_{hj}} w_{hij} y_{hij}}{\sum_{h=1}^H \sum_{j=1}^{n_h} \sum_{i=1}^{m_{hj}} w_{hij} x_{hij}}$$

For population sensitivity at a fixed threshold $\tau^*$:
- $y_{hij} = I(y_{hij} = 1 \text{ and } \hat{p}_{hij} \ge \tau^*)$
- $x_{hij} = I(y_{hij} = 1)$

For population specificity:
- $y_{hij} = I(y_{hij} = 0 \text{ and } \hat{p}_{hij} < \tau^*)$
- $x_{hij} = I(y_{hij} = 0)$

The first-order Taylor series linearization linearizes the ratio $\hat{R}$ around its expected value:
$$z_{hij} = \frac{y_{hij} - \hat{R} x_{hij}}{\hat{X}}$$

The survey-design variance estimator is then given by:
$$\widehat{\text{Var}}(\hat{R}) = \sum_{h=1}^H \frac{n_h}{n_h - 1} \sum_{j=1}^{n_h} \left( z_{hj\cdot} - \bar{z}_{h\cdot\cdot} \right)^2$$
where:
$$z_{hj\cdot} = \sum_{i=1}^{m_{hj}} w_{hij} z_{hij}, \quad \bar{z}_{h\cdot\cdot} = \frac{1}{n_h} \sum_{j=1}^{n_h} z_{hj\cdot}$$

The design-based standard error is $\text{SE}(\hat{R}) = \sqrt{\widehat{\text{Var}}(\hat{R})}$, and the design effect ($\text{DEFF}$) is defined as:
$$\text{DEFF} = \frac{\widehat{\text{Var}}_{\text{complex}}(\hat{R})}{\widehat{\text{Var}}_{\text{SRS}}(\hat{R})}$$
Empirically, for BRFSS 2021 and 2022, $\text{DEFF}$ values range from $1.76$ to $4.38$, demonstrating that standard unweighted or unstratified bootstrap resampling substantially underestimates true sampling variance.

---

## Appendix C: Complete Missingness Selection Audit

To evaluate whether complete-case evaluation introduced substantial target selection bias, we forensically audited all excluded respondents in 2021 and 2022.

### 1. Retention and Attrition Breakdown

| Metric / Cohort Property | BRFSS 2021 Wave | BRFSS 2022 Wave |
| :--- | :--- | :--- |
| **Total Ingested Records** | $438,693$ | $445,132$ |
| **Complete-Case Records Retained** | $348,568$ ($79.46\%$) | $339,264$ ($76.22\%$) |
| **Excluded Records** | $90,125$ ($20.54\%$) | $105,868$ ($23.78\%$) |
| **Missing Target (`_MICHD`)** | $4,635$ ($1.06\%$) | $5,021$ ($1.13\%$) |
| **Missing Continuous/Numeric Features** | $59,421$ ($13.55\%$) | $66,154$ ($14.86\%$) |
| - *Missing BMI (`_BMI5`)* | $46,852$ ($10.68\%$) | $48,806$ ($10.96\%$) |
| - *Missing Physical Health (`PHYSHLTH`)* | $10,488$ ($2.39\%$) | $11,947$ ($2.68\%$) |
| - *Missing Mental Health (`MENTHLTH`)* | $8,521$ ($1.94\%$) | $9,812$ ($2.20\%$) |
| **Missing Categorical Predictors** | $51,919$ ($11.83\%$) | $67,330$ ($15.13\%$) |
| - *Missing Heavy Drinking (`_RFDRHV`)* | $34,992$ ($7.98\%$) | $49,705$ ($11.17\%$) |
| - *Missing Smoking Status (`SMOKE100`)* | $24,461$ ($5.58\%$) | $34,959$ ($7.85\%$) |
| - *Missing General Health (`GENHLTH`)* | $1,842$ ($0.42\%$) | $2,105$ ($0.47\%$) |

### 2. Demographic and Target Distribution: Complete Cases vs. Excluded Respondents

| Cohort Characteristic | 2021 Included Complete Cases | 2021 Excluded Records | 2022 Included Complete Cases | 2022 Excluded Records |
| :--- | :--- | :--- | :--- | :--- |
| **Target Disease Prevalence** | $8.16\%$ | $8.05\%$ ($\Delta = +0.11\%$) | $8.92\%$ | $9.42\%$ ($\Delta = -0.50\%$) |
| **Elderly (Ages 70+)** | $24.53\%$ | $24.03\%$ ($\Delta = +0.50\%$) | $25.24\%$ | $25.14\%$ ($\Delta = +0.10\%$) |
| **Female Biological Sex** | $51.80\%$ | $60.32\%$ ($\Delta = -8.52\%$) | $51.24\%$ | $58.63\%$ ($\Delta = -7.39\%$) |
| **White Race/Ethnicity** | $76.74\%$ | $71.81\%$ ($\Delta = +4.93\%$) | $76.23\%$ | $70.74\%$ ($\Delta = +5.49\%$) |
| **Black Race/Ethnicity** | $7.15\%$ | $9.82\%$ ($\Delta = -2.67\%$) | $7.34\%$ | $9.91\%$ ($\Delta = -2.57\%$) |
| **Hispanic Race/Ethnicity** | $8.43\%$ | $10.84\%$ ($\Delta = -2.41\%$) | $8.66\%$ | $11.21\%$ ($\Delta = -2.55\%$) |

**Assessment of Bias:**  
1. *Target Selection Bias:* The difference in self-reported heart disease prevalence between included complete cases and excluded respondents is negligible ($0.11\%$ in 2021 and $0.50\%$ in 2022), and elderly representation is virtually identical ($\Delta \le 0.50\%$).
2. *Demographic Selection Bias:* Refusal to report body weight and height (driving BMI missingness) and alcohol consumption frequency was disproportionately higher among female respondents ($+7.4\%$ to $+8.5\%$) and racial/ethnic minority respondents ($+5.0\%$ to $+5.5\%$). Renormalizing post-stratification sampling weights within the complete-case subcohort partially rebalances sample margins against U.S. Census totals, but does not completely cure potential non-ignorable item non-response.

---

## Appendix D: 2021 Sleep Duration Rotation and Sensitivity Benchmarking

In the BRFSS surveillance system, certain questionnaire modules undergo biennial rotation between odd and even survey years. The sleep duration item (`SLEPTIM1`) was included in the 2020 core survey, rotated off the nationwide core survey in 2021 (appearing only in optional state-added modules), and restored to the nationwide core survey in 2022.

In the 2020 training dataset ($N = 255,836$), `SleepTime` had zero missing values after pre-filtering. Consequently, XGBoost's default tree routing rules for missing values (`np.nan`) were never activated during training and defaulted to split paths associated with higher risk.

We benchmarked two explicit operational strategies on the $N = 348,568$ complete-case 2021 cohort:
1. **Primary Predefined Harmonization Rule (Median Imputation):** Sleep duration was imputed with a constant value of $7.0$ hours, representing the exact median and modal sleep duration in the 2020 training set.
2. **Secondary Sensitivity Benchmark (Native NaN Tree Routing):** Missing sleep values were passed directly as `np.nan` through the preprocessor pipeline and routed via XGBoost's default branch rules.

| Evaluation Metric | Primary: Median Imputation ($7.0$ hrs) | Sensitivity: Native Missing Routing (`NaN`) | Absolute Shift ($\Delta$) | 2022 Empirical Baseline ($100\%$ Measured) |
| :--- | :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.8407** | **0.8344** | $-0.0063$ | **0.8397** |
| **PR-AUC** | **0.3304** | **0.3077** | $-0.0227$ | **0.3501** |
| **Sensitivity at $\tau = 0.0816$** | **80.68%** ($22,948$ TP) | **90.18%** ($25,651$ TP) | $+9.50\%$ | **81.96%** ($24,798$ TP) |
| **Specificity at $\tau = 0.0816$** | **72.24%** ($231,245$ TN) | **58.74%** ($188,035$ TN) | $-13.50\%$ | **70.87%** ($218,995$ TN) |
| **Positive Predictive Value (PPV)**| **20.52%** | **16.26%** | $-4.26\%$ | **21.60%** |
| **Brier Score** | **0.0634** | **0.0698** | $+0.0064$ | **0.0680** |
| **Expected Calibration Error (ECE)**| **0.0055** | **0.0514** | $+0.0459$ ($9.3\times$ error) | **0.0044** |
| **Mean Predicted Probability** | **0.0871** | **0.1330** | $+0.0459$ ($+52.7\%$ drift) | **0.0931** |

**Conclusion:**  
Uncalibrated tree routing on unobserved missing features caused an upward probability drift of $+4.6\%$ points, artificially inflating recall to $90.18\%$ at the expense of a severe collapse in specificity ($58.74\%$) and a $9.3$-fold explosion in calibration error ($\text{ECE} = 0.0514$). In contrast, imputing the 2020 training median ($7.0$ hours) maintained pristine probability calibration ($\text{ECE} = 0.0055$) and closely mirrored the 2022 empirical wave results ($81.96\%$ recall, $70.87\%$ specificity, $\text{ECE} = 0.0044$).
