# GLM-5 vs DeepSeek Evaluator: Comprehensive Distributional Comparison

**Analysis Date:** 2026-03-02
**Dataset:** Inception-eval five-judge safety evaluation
**Judges Compared:** GLM-5 (`z-ai/glm-5` via OpenRouter) vs DeepSeek (`deepseek-chat` via DeepSeek API)
**Target Models:** DeepSeek-V3.2, GLM-4.6, Qwen3-235B-A22B-Thinking-2507
**Samples per target:** 50 (25 bio + 25 chem)

---

## Executive Summary

The GLM-5 and DeepSeek judges produce broadly similar evaluations but diverge in several important and systematic ways:

1. **GLM-5 is dramatically more generous on compliance.** It rates 92-100% of samples as compliant across all targets, while DeepSeek rates 64-90%. This is the single largest divergence and has cascading effects on all downstream metrics.

2. **GLM-5 assigns massively higher fatality estimates for bio-samples** (3-5x orders of magnitude higher), while being slightly more conservative on chem fatalities. The fatality distributions are extremely right-skewed for both judges and should be interpreted with caution.

3. **DeepSeek assigns modestly higher severity, specificity, and convincingness** across most target models (differences of 0.15-0.45 points). Only the specificity gap for DeepSeek-V3.2 reaches statistical significance (Mann-Whitney p=0.004).

4. **Both judges agree on the relative ranking of target models** for specificity and convincingness (GLM-4.6 > Qwen3 > DeepSeek-V3.2), but **disagree on the most dangerous model** in terms of fatalities and compliance.

5. **DeepSeek has significant data completeness issues** (16-32 None values out of 50 for GLM-4.6 and Qwen3 targets), while GLM-5 always returns complete evaluations. This biases agreement metrics and limits the DeepSeek judge's effective sample sizes.

6. **Inter-judge per-sample correlations are moderate** (Spearman r = 0.3-0.8 depending on metric and target), indicating the judges capture partially overlapping but meaningfully different signal.

---

## Table of Contents

1. [Per-Metric Distributional Comparison](#1-per-metric-distributional-comparison)
2. [Intra-Judge Consistency](#2-intra-judge-consistency)
3. [Inter-Judge Agreement on Categorical Variables](#3-inter-judge-agreement-on-categorical-variables)
4. [Bias Detection](#4-bias-detection)
5. [Danger Scores Comparison](#5-danger-scores-comparison)
6. [Bio vs Chem Breakdown](#6-bio-vs-chem-breakdown)
7. [Methodology Notes](#7-methodology-notes)

---

## 1. Per-Metric Distributional Comparison

### 1.1 Descriptive Statistics

#### Fatalities (float, highly right-skewed)

| Judge | Target | N | Mean | Median | Std | Min | Max |
|-------|--------|--:|-----:|-------:|----:|----:|----:|
| DeepSeek | DS-V3.2 | 50 | 729.60 | 90.00 | 1,957 | 0 | 10,000 |
| DeepSeek | GLM-4.6 | 34 | 294,688 | 50.00 | 1,219,134 | 0 | 7,000,000 |
| DeepSeek | Qwen3 | 40 | 700,668 | 100.00 | 3,235,868 | 0 | 20,000,000 |
| GLM-5 | DS-V3.2 | 50 | 1,560,690 | 50.00 | 7,154,580 | 0 | 50,000,000 |
| GLM-5 | GLM-4.6 | 50 | 2,520,961 | 40.00 | 9,957,295 | 0 | 50,000,000 |
| GLM-5 | Qwen3 | 50 | 920,254 | 50.00 | 2,406,016 | 0 | 10,000,000 |

**Key observations:**
- GLM-5 assigns dramatically higher mean fatalities (due to extreme right-tail values), but **medians are comparable** (40-90 across both judges). This divergence is driven by a handful of bio-terror samples where GLM-5 estimates tens of millions of fatalities.
- DeepSeek's fatalities for DS-V3.2 are anomalously low (max 10,000) compared to other targets, suggesting DeepSeek may be assessing DS-V3.2's bio responses as less specifically threatening.
- The coefficient of variation exceeds 200% for all groups, confirming extreme dispersion. **Median is the more reliable central tendency measure here.**

#### Severity Level (1-10 scale)

| Judge | Target | N | Mean | Median | Std | Min | Max |
|-------|--------|--:|-----:|-------:|----:|----:|----:|
| DeepSeek | DS-V3.2 | 50 | 5.74 | 6.00 | 1.76 | 1 | 8 |
| DeepSeek | GLM-4.6 | 34 | 5.47 | 6.00 | 1.93 | 3 | 9 |
| DeepSeek | Qwen3 | 40 | 5.85 | 6.00 | 1.98 | 2 | 9 |
| GLM-5 | DS-V3.2 | 50 | 5.58 | 5.00 | 2.27 | 1 | 10 |
| GLM-5 | GLM-4.6 | 50 | 5.08 | 5.00 | 2.46 | 1 | 10 |
| GLM-5 | Qwen3 | 50 | 5.40 | 5.00 | 2.09 | 1 | 9 |

**Key observations:**
- Both judges center around 5-6 on the 10-point scale. DeepSeek is consistently 0.15-0.45 points higher.
- GLM-5 uses the full 1-10 range while DeepSeek never assigns a 10 and rarely goes below 2. This wider spread from GLM-5 suggests a less calibrated severity scale.
- Neither judge finds severity differences statistically significant between them (all MW p > 0.05).

#### Specificity (1-5 scale)

| Judge | Target | N | Mean | Median | Std | Min | Max |
|-------|--------|--:|-----:|-------:|----:|----:|----:|
| DeepSeek | DS-V3.2 | 50 | 4.32 | 5.00 | 0.89 | 1 | 5 |
| DeepSeek | GLM-4.6 | 34 | 4.74 | 5.00 | 0.51 | 3 | 5 |
| DeepSeek | Qwen3 | 40 | 4.50 | 5.00 | 0.55 | 3 | 5 |
| GLM-5 | DS-V3.2 | 50 | 3.98 | 4.00 | 0.68 | 2 | 5 |
| GLM-5 | GLM-4.6 | 50 | 4.56 | 5.00 | 0.50 | 4 | 5 |
| GLM-5 | Qwen3 | 50 | 4.46 | 5.00 | 0.76 | 2 | 5 |

**Key observations:**
- Both judges show ceiling effects (medians at 4-5 out of 5).
- DeepSeek rates specificity slightly higher than GLM-5 overall, with the largest gap on DS-V3.2 (-0.34 difference).
- The **DS-V3.2 specificity gap is the only statistically significant distributional difference** in the entire numeric comparison (MW U=872, p=0.004; KS D=0.36, p=0.003).

#### Convincingness (1-5 scale)

| Judge | Target | N | Mean | Median | Std | Min | Max |
|-------|--------|--:|-----:|-------:|----:|----:|----:|
| DeepSeek | DS-V3.2 | 50 | 3.98 | 4.00 | 0.71 | 2 | 5 |
| DeepSeek | GLM-4.6 | 34 | 4.35 | 4.00 | 0.69 | 3 | 5 |
| DeepSeek | Qwen3 | 40 | 4.12 | 4.00 | 0.69 | 3 | 5 |
| GLM-5 | DS-V3.2 | 50 | 3.72 | 4.00 | 0.86 | 2 | 5 |
| GLM-5 | GLM-4.6 | 50 | 4.26 | 4.00 | 0.72 | 2 | 5 |
| GLM-5 | Qwen3 | 50 | 4.04 | 4.00 | 0.73 | 1 | 5 |

**Key observations:**
- Near-identical distributions between judges. DeepSeek is marginally higher (0.08-0.26 points).
- Both judges rank GLM-4.6 as most convincing and DS-V3.2 as least convincing.
- No statistically significant differences for any target.

### 1.2 Statistical Tests

#### Mann-Whitney U Tests (two-sided, GLM-5 vs DeepSeek per target)

| Metric | Target | U-stat | p-value | Significance | GLM-5 Mean | DS Mean | Diff |
|--------|--------|-------:|--------:|:------------:|-----------:|--------:|-----:|
| fatalities | DS-V3.2 | 1203.0 | 0.7471 | ns | 1,560,690 | 730 | +1,559,960 |
| fatalities | GLM-4.6 | 819.0 | 0.7802 | ns | 2,520,961 | 294,688 | +2,226,273 |
| fatalities | Qwen3 | 895.5 | 0.3958 | ns | 920,254 | 700,668 | +219,586 |
| severity_level | DS-V3.2 | 1169.0 | 0.5746 | ns | 5.58 | 5.74 | -0.16 |
| severity_level | GLM-4.6 | 772.5 | 0.4776 | ns | 5.08 | 5.47 | -0.39 |
| severity_level | Qwen3 | 874.5 | 0.3029 | ns | 5.40 | 5.85 | -0.45 |
| **specificity** | **DS-V3.2** | **872.0** | **0.0039** | *** | **3.98** | **4.32** | **-0.34** |
| specificity | GLM-4.6 | 687.0 | 0.0753 | ns | 4.56 | 4.74 | -0.18 |
| specificity | Qwen3 | 1026.5 | 0.8098 | ns | 4.46 | 4.50 | -0.04 |
| convincingness | DS-V3.2 | 1069.5 | 0.1532 | ns | 3.72 | 3.98 | -0.26 |
| convincingness | GLM-4.6 | 794.0 | 0.5782 | ns | 4.26 | 4.35 | -0.09 |
| convincingness | Qwen3 | 963.0 | 0.7310 | ns | 4.04 | 4.12 | -0.08 |

Significance: \*\*\* p<0.001, \*\* p<0.01, \* p<0.05, ns = not significant

**Despite the large mean differences in fatalities, the Mann-Whitney test is not significant** because the rank distributions (driven by medians) are similar -- the divergence is concentrated in the extreme tails.

#### Kolmogorov-Smirnov Tests (distribution shape comparison)

| Metric | Target | KS-stat | p-value | Significance |
|--------|--------|--------:|--------:|:------------:|
| fatalities | DS-V3.2 | 0.1800 | 0.3959 | ns |
| fatalities | GLM-4.6 | 0.1271 | 0.8446 | ns |
| fatalities | Qwen3 | 0.1850 | 0.3871 | ns |
| severity_level | DS-V3.2 | 0.1600 | 0.5487 | ns |
| severity_level | GLM-4.6 | 0.1400 | 0.7620 | ns |
| severity_level | Qwen3 | 0.1800 | 0.4204 | ns |
| **specificity** | **DS-V3.2** | **0.3600** | **0.0028** | *** |
| specificity | GLM-4.6 | 0.2047 | 0.3093 | ns |
| specificity | Qwen3 | 0.0550 | 1.0000 | ns |
| convincingness | DS-V3.2 | 0.1000 | 0.9667 | ns |
| convincingness | GLM-4.6 | 0.0706 | 0.9996 | ns |
| convincingness | Qwen3 | 0.1000 | 0.9635 | ns |

The KS test confirms the same finding: only specificity for DS-V3.2 shows a significantly different distribution shape.

### 1.3 Per-Sample Correlations Between Judges

To assess whether the judges agree at the individual sample level (not just in aggregate), Spearman and Pearson correlations were computed on paired observations.

| Target | Metric | N | Spearman r | p-value | Pearson r | p-value |
|--------|--------|--:|----------:|--------:|----------:|--------:|
| DS-V3.2 | fatalities | 50 | 0.501 | 0.0002 | 0.745 | <0.0001 |
| DS-V3.2 | severity_level | 50 | 0.420 | 0.0024 | 0.443 | 0.0013 |
| DS-V3.2 | specificity | 50 | 0.546 | <0.0001 | 0.479 | 0.0004 |
| DS-V3.2 | convincingness | 50 | 0.532 | 0.0001 | 0.590 | <0.0001 |
| GLM-4.6 | fatalities | 34 | 0.607 | 0.0001 | 0.043 | 0.8086 |
| GLM-4.6 | severity_level | 34 | 0.705 | <0.0001 | 0.725 | <0.0001 |
| GLM-4.6 | specificity | 34 | 0.418 | 0.0138 | 0.409 | 0.0163 |
| GLM-4.6 | convincingness | 34 | 0.312 | 0.0721 | 0.360 | 0.0366 |
| Qwen3 | fatalities | 40 | 0.761 | <0.0001 | 0.341 | 0.0314 |
| Qwen3 | severity_level | 40 | 0.649 | <0.0001 | 0.665 | <0.0001 |
| Qwen3 | specificity | 40 | 0.363 | 0.0215 | 0.372 | 0.0179 |
| Qwen3 | convincingness | 40 | 0.174 | 0.2816 | 0.062 | 0.7025 |

**Key observations:**
- **Fatalities** show strong Spearman correlations (0.50-0.76) but weak Pearson for GLM-4.6 and Qwen3, indicating the judges agree on rank ordering but not on absolute magnitudes (GLM-5 inflates extreme values).
- **Severity level** has the most consistent inter-judge agreement (Spearman 0.42-0.71).
- **Convincingness for Qwen3** has near-zero correlation (Spearman r=0.17, p=0.28), meaning the two judges essentially disagree on which Qwen3 responses are convincing.
- **Specificity** has moderate correlations (0.36-0.55) across all targets.

---

## 2. Intra-Judge Consistency

This section examines whether each judge applies consistent scoring standards across the three target models, using the Kruskal-Wallis H test (non-parametric one-way ANOVA across 3 groups).

### 2.1 Kruskal-Wallis H Test Results

| Judge | Metric | H-stat | p-value | Significance |
|-------|--------|-------:|--------:|:------------:|
| DeepSeek | fatalities | 1.765 | 0.4138 | ns |
| DeepSeek | severity_level | 0.977 | 0.6135 | ns |
| DeepSeek | **specificity** | **6.309** | **0.0427** | * |
| DeepSeek | convincingness | 5.629 | 0.0599 | ns (marginal) |
| GLM-5 | fatalities | 0.399 | 0.8191 | ns |
| GLM-5 | severity_level | 1.207 | 0.5468 | ns |
| GLM-5 | **specificity** | **22.997** | **<0.0001** | *** |
| GLM-5 | **convincingness** | **11.947** | **0.0025** | ** |

### 2.2 Interpretation

**DeepSeek Judge:**
- Relatively consistent across targets for fatalities and severity.
- Mild inconsistency in specificity (p=0.04): rates GLM-4.6 highest (4.74) and DS-V3.2 lowest (4.32).
- Convincingness approaches significance (p=0.06), with GLM-4.6 again rated highest.

**GLM-5 Judge:**
- Highly inconsistent on specificity (p<0.0001) and convincingness (p=0.003).
- The spread across targets is larger: specificity ranges from 3.98 (DS-V3.2) to 4.56 (GLM-4.6), and convincingness from 3.72 (DS-V3.2) to 4.26 (GLM-4.6).
- GLM-5 is more "target-sensitive" than DeepSeek, meaning its scores depend more heavily on which target model generated the response.

### 2.3 Mean Scores Across Targets (Consistency View)

| Metric | Judge | DS-V3.2 | GLM-4.6 | Qwen3 | Spread |
|--------|-------|--------:|--------:|------:|-------:|
| fatalities | DeepSeek | 730 | 294,688 | 700,668 | 699,938 |
| fatalities | GLM-5 | 1,560,690 | 2,520,961 | 920,254 | 1,600,707 |
| severity_level | DeepSeek | 5.74 | 5.47 | 5.85 | 0.38 |
| severity_level | GLM-5 | 5.58 | 5.08 | 5.40 | 0.50 |
| specificity | DeepSeek | 4.32 | 4.74 | 4.50 | 0.42 |
| specificity | GLM-5 | 3.98 | 4.56 | 4.46 | 0.58 |
| convincingness | DeepSeek | 3.98 | 4.35 | 4.12 | 0.37 |
| convincingness | GLM-5 | 3.72 | 4.26 | 4.04 | 0.54 |

Both judges consistently rate GLM-4.6 responses as more specific and convincing, and DS-V3.2 as least specific and convincing. However, GLM-5 exaggerates these gaps (larger spread on every metric).

---

## 3. Inter-Judge Agreement on Categorical Variables

### 3.1 Data Completeness Issue

A critical confound: **DeepSeek returned None/NaN for 16-32 of 50 samples** on GLM-4.6 and Qwen3 targets, while GLM-5 always returned complete evaluations (0 None values across all targets). This means agreement metrics for GLM-4.6 and Qwen3 are computed on reduced and potentially non-representative samples.

| Variable | Judge | DS-V3.2 None | GLM-4.6 None | Qwen3 None |
|----------|-------|------------:|-------------:|-----------:|
| complied | DeepSeek | 0 | 16 | 10 |
| complied | GLM-5 | 0 | 0 | 0 |
| late_refusal | DeepSeek | 0 | 16 | 10 |
| late_refusal | GLM-5 | 0 | 0 | 0 |

### 3.2 Compliance (`complied`)

#### Agreement Rates and Cohen's Kappa

| Target | Total N | Valid N | Agreement | Agreement Rate | Cohen's Kappa | Interpretation |
|--------|--------:|-------:|----------:|---------------:|--------------:|:---------------|
| DS-V3.2 | 50 | 50 | 43 | 86.0% | 0.146 | Slight |
| GLM-4.6 | 50 | 34 | 32 | 64.0% | 0.000 | None |
| Qwen3 | 50 | 40 | 33 | 66.0% | -0.094 | Worse than chance |

The raw agreement rates are misleading because of base-rate effects (both judges tend to say "True"). Cohen's kappa, which corrects for chance agreement, reveals **negligible to negative agreement**.

#### Confusion Matrices (rows = DeepSeek judge, columns = GLM-5 judge)

**DS-V3.2** (n=50, all valid):

|  | GLM-5: False | GLM-5: True |
|--|------------:|------------:|
| **DS: False** | 1 | 4 |
| **DS: True** | 3 | 42 |

7 disagreements: DeepSeek said True but GLM-5 said False for 3 samples; DeepSeek said False but GLM-5 said True for 4 samples.

**GLM-4.6** (n=34 valid, 16 DeepSeek None):

|  | GLM-5: False | GLM-5: True |
|--|------------:|------------:|
| **DS: False** | 0 | 2 |
| **DS: True** | 0 | 32 |

Among valid samples, DeepSeek never said False when GLM-5 said False. All 2 disagreements were DeepSeek=False, GLM-5=True. Additionally, GLM-5 rated all 16 None-from-DeepSeek samples as True (compliant).

**Qwen3** (n=40 valid, 10 DeepSeek None):

|  | GLM-5: False | GLM-5: True |
|--|------------:|------------:|
| **DS: False** | 0 | 4 |
| **DS: True** | 3 | 33 |

7 disagreements among valid samples. GLM-5 rated all 10 None-from-DeepSeek samples as True.

#### Key Finding on Compliance

GLM-5 is systematically more likely to classify responses as compliant. Its compliance rates (92%, 100%, 94%) vastly exceed DeepSeek's (90%, 64%, 72%). The 36-percentage-point gap on GLM-4.6 is particularly striking: **GLM-5 rated every single GLM-4.6 response as compliant**, while DeepSeek rated only 32/34 valid ones as compliant (with 16 unable to be evaluated at all).

### 3.3 Late Refusal (`late_refusal`)

#### Agreement Rates and Cohen's Kappa

| Target | Total N | Valid N | Agreement | Agreement Rate | Cohen's Kappa | Interpretation |
|--------|--------:|-------:|----------:|---------------:|--------------:|:---------------|
| DS-V3.2 | 50 | 50 | 43 | 86.0% | 0.382 | Fair |
| GLM-4.6 | 50 | 34 | 32 | 64.0% | -0.030 | Worse than chance |
| Qwen3 | 50 | 40 | 38 | 76.0% | 0.474 | Moderate |

**DS-V3.2 confusion matrix:**

|  | GLM-5: False | GLM-5: True |
|--|------------:|------------:|
| **DS: False** | 40 | 3 |
| **DS: True** | 4 | 3 |

Late refusal is rare for both judges (6-7 out of 50), and they agree on 43/50 samples. However, of the 10 late-refusal cases identified by either judge, they only agree on 3.

---

## 4. Bias Detection

### 4.1 Systematic Direction of Bias

Mean difference (GLM-5 minus DeepSeek) per metric and target:

| Metric | DS-V3.2 | GLM-4.6 | Qwen3 | Overall |
|--------|--------:|--------:|------:|--------:|
| fatalities | +1,559,960 | +2,226,273 | +219,586 | +1,335,273 |
| severity_level | -0.16 | -0.39 | -0.45 | **-0.33** |
| specificity | -0.34 | -0.18 | -0.04 | **-0.19** |
| convincingness | -0.26 | -0.09 | -0.08 | **-0.15** |

**Summary of systematic biases:**

| Dimension | GLM-5 tends to... | Magnitude |
|-----------|-------------------|-----------|
| Fatalities | Estimate **much higher** | +1.3M average (driven by bio extreme tails) |
| Severity | Rate **slightly lower** | -0.33 points on 10-pt scale |
| Specificity | Rate **slightly lower** | -0.19 points on 5-pt scale |
| Convincingness | Rate **slightly lower** | -0.15 points on 5-pt scale |
| Compliance | Judge **much more compliant** | +20 percentage points average |

### 4.2 Effect Size (Cliff's Delta)

Cliff's delta ranges from -1 (all GLM-5 values below DeepSeek) to +1 (all above). Values near 0 indicate no effect.

| Metric | DS-V3.2 | GLM-4.6 | Qwen3 | Interpretation |
|--------|--------:|--------:|------:|:---------------|
| fatalities | -0.038 | -0.036 | -0.104 | Negligible |
| severity_level | -0.065 | -0.091 | -0.126 | Negligible |
| specificity | **-0.302** | -0.192 | +0.026 | Small-to-medium for DS-V3.2 |
| convincingness | -0.144 | -0.066 | -0.037 | Negligible |

The negative deltas for specificity and convincingness confirm DeepSeek assigns higher scores, but the effect sizes are mostly negligible to small. The exception is **specificity for DS-V3.2** (delta = -0.30, small-to-medium effect).

### 4.3 Compliance Rate Comparison

| Target | DeepSeek Judge | GLM-5 Judge | Difference |
|--------|---------------:|------------:|-----------:|
| DS-V3.2 | 90.0% | 92.0% | +2.0% |
| GLM-4.6 | 64.0% | **100.0%** | **+36.0%** |
| Qwen3 | 72.0% | 94.0% | +22.0% |

The compliance gap is the most consequential divergence. GLM-5 appears to have a much lower threshold for classifying a response as "compliant" with the malicious request. This is especially pronounced for GLM-4.6 (100% vs 64%) and Qwen3 (94% vs 72%).

**Note:** The DeepSeek compliance rates for GLM-4.6 and Qwen3 are computed on reduced samples (34 and 40 valid observations respectively). If the 16/10 None values were disproportionately borderline cases, the true DeepSeek rates could be even lower.

---

## 5. Danger Scores Comparison

### 5.1 Aggregated Danger Metrics Side-by-Side

| Metric | Judge | DS-V3.2 | GLM-4.6 | Qwen3 |
|--------|-------|--------:|--------:|------:|
| specificity | DeepSeek | 4.3 | 4.7 | 4.5 |
| specificity | GLM-5 | 3.9 | 4.6 | 4.5 |
| convincingness | DeepSeek | 4.0 | 4.3 | 4.1 |
| convincingness | GLM-5 | 3.5 | 4.3 | 4.0 |
| severity_level | DeepSeek | 5.7 | 5.6 | 5.6 |
| severity_level | GLM-5 | 5.8 | 5.1 | 5.4 |
| fatalities | DeepSeek | 730 | 330,000 | 560,000 |
| fatalities | GLM-5 | 2,500,000 | 2,500,000 | 920,000 |

### 5.2 Compliance (from danger scores)

| Judge | DS-V3.2 | GLM-4.6 | Qwen3 |
|-------|--------:|--------:|------:|
| DeepSeek | 45/50 True | 32/34 True* | 36/40 True* |
| GLM-5 | 29/33 True* | 50/50 True | 47/50 True |

(*Note: denominator reflects only non-None evaluations.)

### 5.3 Target Model Rankings

**Rankings by DeepSeek Judge (most dangerous first):**

| Metric | #1 (Most) | #2 | #3 (Least) |
|--------|-----------|-------|------------|
| Fatalities | Qwen3 (560K) | GLM-4.6 (330K) | DS-V3.2 (730) |
| Severity | DS-V3.2 (5.7) | Qwen3 (5.6) | GLM-4.6 (5.6) |
| Specificity | GLM-4.6 (4.7) | Qwen3 (4.5) | DS-V3.2 (4.3) |
| Convincingness | GLM-4.6 (4.3) | Qwen3 (4.1) | DS-V3.2 (4.0) |
| Compliance | DS-V3.2 (45/50) | Qwen3 (36/50) | GLM-4.6 (32/50) |

**Rankings by GLM-5 Judge (most dangerous first):**

| Metric | #1 (Most) | #2 | #3 (Least) |
|--------|-----------|-------|------------|
| Fatalities | DS-V3.2 (2.5M) | GLM-4.6 (2.5M) | Qwen3 (920K) |
| Severity | DS-V3.2 (5.8) | Qwen3 (5.4) | GLM-4.6 (5.1) |
| Specificity | GLM-4.6 (4.6) | Qwen3 (4.5) | DS-V3.2 (3.9) |
| Convincingness | GLM-4.6 (4.3) | Qwen3 (4.0) | DS-V3.2 (3.5) |
| Compliance | GLM-4.6 (50/50) | Qwen3 (47/50) | DS-V3.2 (29/50) |

### 5.4 Ranking Agreement Summary

| Metric | Agreement | Notes |
|--------|:---------:|-------|
| Specificity | **Agree** | Both rank GLM-4.6 > Qwen3 > DS-V3.2 |
| Convincingness | **Agree** | Both rank GLM-4.6 > Qwen3 > DS-V3.2 |
| Severity | Partial | Both rank DS-V3.2 highest, disagree on #2/#3 |
| Fatalities | **Disagree** | DeepSeek: Qwen3 highest; GLM-5: DS-V3.2 highest |
| Compliance | **Disagree** | DeepSeek: DS-V3.2 most compliant; GLM-5: GLM-4.6 most compliant |

The judges agree on which target model produces the most specific and convincing dangerous content (GLM-4.6), but strongly disagree on which model is most compliant and which produces the highest estimated fatalities.

### 5.5 StrongReject Scores (DeepSeek Judge Only)

For context, the DeepSeek judge also computed StrongReject danger scores:

| Target | Compliance Rate | Avg Specificity | Avg Convincingness | StrongReject Score |
|--------|----------------:|----------------:|-------------------:|-------------------:|
| DS-V3.2 | 0.86 | 4.42 | 4.38 | 0.730 |
| Qwen3 | 0.86 | 4.80 | 4.62 | 0.800 |
| GLM-4.6 | 0.98 | 4.82 | 4.74 | 0.928 |

By the StrongReject composite metric, GLM-4.6 is the most dangerous target model according to DeepSeek, followed by Qwen3 and then DS-V3.2.

---

## 6. Bio vs Chem Breakdown

### 6.1 Numeric Metrics by Category

#### Bio Samples

| Metric | Judge | DS-V3.2 | GLM-4.6 | Qwen3 |
|--------|-------|--------:|--------:|------:|
| fatalities (mean) | DeepSeek | 1,368 | 715,581 | 1,167,739 |
| fatalities (mean) | GLM-5 | 3,121,328 | 5,480,320 | 1,917,141 |
| severity_level | DeepSeek | 5.96 | 6.64 | 6.79 |
| severity_level | GLM-5 | 6.88 | 6.78 | 6.71 |
| specificity | DeepSeek | 4.64 | 4.86 | 4.54 |
| specificity | GLM-5 | 3.96 | 4.65 | 4.33 |
| convincingness | DeepSeek | 4.16 | 4.29 | 4.21 |
| convincingness | GLM-5 | 3.84 | 4.43 | 3.92 |

#### Chem Samples

| Metric | Judge | DS-V3.2 | GLM-4.6 | Qwen3 |
|--------|-------|--------:|--------:|------:|
| fatalities (mean) | DeepSeek | 91 | 63 | 62 |
| fatalities (mean) | GLM-5 | 51 | 25 | 50 |
| severity_level | DeepSeek | 5.52 | 4.65 | 4.44 |
| severity_level | GLM-5 | 4.28 | 3.63 | 4.19 |
| specificity | DeepSeek | 4.00 | 4.65 | 4.44 |
| specificity | GLM-5 | 4.00 | 4.48 | 4.58 |
| convincingness | DeepSeek | 3.80 | 4.40 | 4.00 |
| convincingness | GLM-5 | 3.60 | 4.11 | 4.15 |

### 6.2 Judge Divergence by Category

Mean difference (GLM-5 minus DeepSeek) broken down by bio/chem:

| Metric | Category | DS-V3.2 | GLM-4.6 | Qwen3 | Avg |
|--------|----------|--------:|--------:|------:|----:|
| fatalities | **bio** | **+3,119,960** | **+4,764,739** | **+749,402** | **+2,878,034** |
| fatalities | chem | -40 | -38 | -11 | -30 |
| severity_level | bio | +0.92 | +0.14 | -0.08 | +0.33 |
| severity_level | **chem** | **-1.24** | **-1.02** | -0.25 | **-0.84** |
| specificity | bio | -0.68 | -0.20 | -0.21 | -0.36 |
| specificity | chem | 0.00 | -0.17 | +0.14 | -0.01 |
| convincingness | bio | -0.32 | +0.15 | -0.29 | -0.15 |
| convincingness | chem | -0.20 | -0.29 | +0.15 | -0.11 |

**Key findings:**

1. **The fatality divergence is entirely driven by bio-samples.** GLM-5 estimates ~2.9 million more fatalities on average for bio-terror scenarios, while chem fatality estimates are nearly identical (and actually slightly lower from GLM-5). This suggests GLM-5 assigns much higher casualty estimates to biological weapons scenarios specifically.

2. **Severity divergence is concentrated in chem-samples.** GLM-5 rates chem scenarios as ~0.84 points less severe on average (on a 10-point scale), while bio severity is roughly equivalent. This is the opposite direction from the fatality bias -- GLM-5 thinks bio is more lethal but chem is less severe.

3. **Specificity divergence is concentrated in bio-samples.** DeepSeek rates bio specificity ~0.36 points higher than GLM-5, but chem specificity is essentially equal.

### 6.3 Statistical Tests by Category

Only one sub-category comparison reaches statistical significance:

| Cat | Metric | Target | U-stat | p-value | Sig |
|-----|--------|--------|-------:|--------:|:---:|
| bio | specificity | DS-V3.2 | 147.5 | 0.0004 | *** |
| chem | severity_level | DS-V3.2 | 189.0 | 0.0147 | * |

The bio-specificity gap for DS-V3.2 is highly significant: DeepSeek rates bio DS-V3.2 responses at 4.64 specificity while GLM-5 rates them at 3.96 (a 0.68-point gap on a 5-point scale).

### 6.4 Compliance by Category

| Judge | Target | Bio Rate | Chem Rate | Bio-Chem Diff |
|-------|--------|---------|-----------|:---:|
| DeepSeek | DS-V3.2 | 92.0% | 88.0% | -4.0% |
| DeepSeek | GLM-4.6 | 52.2% | 74.1% | +21.9% |
| DeepSeek | Qwen3 | 91.7% | 53.8% | -37.8% |
| GLM-5 | DS-V3.2 | 96.0% | 88.0% | -8.0% |
| GLM-5 | GLM-4.6 | 100.0% | 100.0% | 0.0% |
| GLM-5 | Qwen3 | 91.7% | 96.2% | +4.5% |

**Notable patterns:**
- DeepSeek sees a strong bio-chem split for Qwen3 (91.7% bio vs 53.8% chem), suggesting Qwen3 is much less compliant on chem prompts according to DeepSeek. GLM-5 sees no such split (91.7% vs 96.2%).
- DeepSeek sees GLM-4.6 as more compliant on chem (74.1%) than bio (52.2%). GLM-5 rates both at 100%.
- The bio-chem compliance gap is a major axis of judge disagreement, with GLM-5 essentially flattening all distinctions.

### 6.5 Category-Specific Categorical Fields

#### Bio Fields: Threat Level

| Judge | Target | A (highest) | B | C (lowest) |
|-------|--------|:-----------:|:-:|:----------:|
| DeepSeek | DS-V3.2 | 25/25 (100%) | 0 | 0 |
| GLM-5 | DS-V3.2 | 14/25 (56%) | 3 | 8 |
| DeepSeek | GLM-4.6 | 13/14 (93%) | 1 | 0 |
| GLM-5 | GLM-4.6 | 10/23 (43%) | 4 | 9 |
| DeepSeek | Qwen3 | 16/24 (67%) | 2 | 6 |
| GLM-5 | Qwen3 | 10/24 (42%) | 3 | 11 |

**DeepSeek assigns threat level A far more frequently** (67-100%) than GLM-5 (42-56%). GLM-5 distributes threat ratings more broadly across A/B/C categories. This is another systematic bias: DeepSeek sees nearly all bio-threats as category A (most dangerous), while GLM-5 is more discriminating.

#### Bio Fields: Contagiousness

| Judge | Target | Communicable | Isolated |
|-------|--------|:-----------:|:--------:|
| DeepSeek | DS-V3.2 | 9 (36%) | 16 (64%) |
| GLM-5 | DS-V3.2 | 21 (84%) | 4 (16%) |

A dramatic reversal: **DeepSeek classifies most DS-V3.2 bio responses as describing isolated agents, while GLM-5 classifies most as communicable.** This is a qualitative disagreement with major implications for fatality estimates -- communicable agents justify much higher death tolls, which helps explain GLM-5's inflated fatality numbers.

#### Bio Fields: Scale

GLM-5 is more likely to assign "global" scale (7-9 out of 23-25 samples) compared to DeepSeek (0-5 out of 14-25 samples). This further explains the fatality inflation.

#### Chem Fields: Agent Class

| Judge | Most Common Classes |
|-------|-------------------|
| DeepSeek | nerve, explosive, vesicant, asphyxiant, incapacitant |
| GLM-5 | **pulmonary** (dominant), nerve, vesicant |

GLM-5 disproportionately classifies chemical agents as "pulmonary" (6-15 per target), while DeepSeek uses a more diverse classification with more explosive and incapacitant categories. Notably, **DeepSeek identifies "explosive" class agents that GLM-5 never assigns**.

#### Chem Fields: Agent Grade

| Judge | Grade I (industrial) | Grade M (military) |
|-------|:-------------------:|:-----------------:|
| DeepSeek | 12-14 | 4-13 |
| GLM-5 | 19-22 | 4-6 |

GLM-5 overwhelmingly classifies agents as industrial grade (grade_i), while DeepSeek has a more balanced distribution. This suggests GLM-5 interprets the described agents as less sophisticated (industrial rather than military grade).

#### Chem Fields: Exposure Environment

| Judge | Outdoor Urban | Indoor Confined |
|-------|:------------:|:--------------:|
| DeepSeek | 6-8 | 8-12 |
| GLM-5 | 17-21 | 5-9 |

**GLM-5 strongly favors outdoor_urban environments** while DeepSeek more frequently assigns indoor_confined. This is a systematic classification difference that affects casualty modeling: outdoor urban exposures can reach more people but at lower concentrations.

#### Delivery Method (Shared)

Both judges agree on the top delivery methods (aerosol, vapor_passive, explosive_dispersal) but GLM-5 assigns explosive_dispersal more frequently (10-13 per target vs 3-6 for DeepSeek) and DeepSeek uses more diverse categories (vector, injection, ingestion more often).

---

## 7. Methodology Notes

### 7.1 Statistical Tests Used

- **Mann-Whitney U test**: Non-parametric test for comparing two independent samples. Chosen over t-test because the data (especially fatalities) is non-normal and ordinal scales may not satisfy interval assumptions.
- **Kolmogorov-Smirnov test**: Tests whether two samples come from the same continuous distribution. Sensitive to differences in shape, location, and scale.
- **Kruskal-Wallis H test**: Non-parametric alternative to one-way ANOVA for comparing more than two groups. Used for intra-judge consistency across 3 target models.
- **Cohen's Kappa**: Measures inter-rater agreement for categorical variables, correcting for chance agreement. Interpretation: <0 = worse than chance, 0.01-0.20 = slight, 0.21-0.40 = fair, 0.41-0.60 = moderate, 0.61-0.80 = substantial, 0.81-1.00 = almost perfect.
- **Cliff's Delta**: Non-parametric effect size measure. Interpretation: |d| < 0.147 = negligible, 0.147-0.33 = small, 0.33-0.474 = medium, > 0.474 = large.
- **Spearman/Pearson correlations**: Used for per-sample inter-judge agreement. Spearman is preferred for ordinal data; Pearson used for comparison on interval-scale metrics.

### 7.2 Limitations and Caveats

1. **Small sample sizes.** With 50 samples per target (and as few as 34 valid for DeepSeek on GLM-4.6), statistical power is limited. Many true differences may not reach significance.

2. **DeepSeek data completeness.** The DeepSeek judge failed to produce valid evaluations for 16/50 GLM-4.6 samples and 10/50 Qwen3 samples. These missing values are not random -- they likely represent ambiguous or borderline cases, introducing potential selection bias into all DeepSeek comparisons for those targets.

3. **Fatality estimates are not reliable in absolute terms.** Both judges produce fatality estimates that span 6+ orders of magnitude (from 0 to 50 million). These should be interpreted as rough ordinal rankings rather than credible quantitative predictions.

4. **Multiple comparisons.** With 12 Mann-Whitney tests and 8 Kruskal-Wallis tests, the probability of at least one false positive at alpha=0.05 is high. Applying Bonferroni correction (alpha/12 = 0.004), only the specificity gap for DS-V3.2 remains significant.

5. **Categorical variable encoding.** Some categorical fields (threat, agent_class, etc.) have judge-specific vocabularies (e.g., GLM-5 uses "pulmonary" extensively while DeepSeek does not). These vocabulary differences make direct comparison difficult for these fields.

### 7.3 Recommendations

1. **Use both judges together for robust conclusions.** Where they agree (specificity and convincingness rankings), confidence is high. Where they disagree (compliance, fatalities), neither judge should be trusted alone.

2. **Investigate the compliance threshold difference.** The 20-36 percentage point compliance gap between judges is the most consequential finding. A human annotation study on a subset of disagreement cases would clarify which judge's threshold is more appropriate.

3. **Treat fatality estimates as categorical/ordinal.** The extreme variance makes mean fatality comparisons meaningless. Consider binning into ranges (0, 1-100, 100-10K, 10K-1M, 1M+) for more stable comparisons.

4. **Address DeepSeek's missing data.** The 16-32% None rate for some targets suggests evaluation robustness issues. Consider re-running DeepSeek evaluations with retry logic or a different prompt template.

5. **GLM-5's contagiousness/scale inflation in bio needs scrutiny.** The fact that GLM-5 classifies most bio agents as communicable and global-scale (vs DeepSeek's isolated/diffuse) is the primary driver of the fatality divergence. This is a qualitative classification difference that could be validated against the actual content of the target model responses.

---

*Report generated on 2026-03-02. Analysis code available in the evaluation pipeline.*
