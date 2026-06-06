# Critical Analysis: Monte-Carlo-Simulation zur stochastischen Bewertung von Investitionsrenditen in der Ölindustrie

**Authors:** Elias Scho & Fabian Finkenzeller
**Reviewer:** CTO (Paperclip)
**Date:** June 2026

---

## Executive Summary

This paper attempts to develop a stochastic ROI model for oil industry investments using Monte Carlo simulation. While the topic is relevant and the motivation is sound, the paper contains several critical errors that undermine its validity. The most severe issues are a **dimensional mismatch in the ROI formula** (annual OPEX added to total CAPEX without time-scaling), a **contradictory dual-distribution assignment for oil price** (both triangular and lognormal), and a **unit error in the oil price calibration** ($20M/barrel is absurd). Additionally, the paper presents simulated results without having performed an actual simulation — the "results" are approximate numbers with no statistical rigor.

---

## 1. Mathematical and Technical Errors

### 1.1 CRITICAL: OPEX Time Dimension Mismatch

**Location:** Lines 272, 327, 377

The paper defines OPEX as annual cost:
```
OPEX: Tri(80M, 120M, 200M) — laufende Kosten pro Jahr
```
But the ROI formula adds OPEX directly to CAPEX:
```
Gesamtkosten_i = CAPEX_i + OPEX_i
```

CAPEX is a one-time cost. OPEX is an annual cost. For a project lasting Y years, total OPEX = Y × annual OPEX. For a typical offshore project (10–20 years), this understates total OPEX by a factor of 10–20×.

**Severity:** CRITICAL. The ROI values reported (~45% mean) are grossly overstated.

### 1.2 CRITICAL: Dual Oil Price Distribution

**Location:** Lines 273, 277–310

The paper assigns oil price TWO conflicting distributions:
1. **Line 273 (calibration table):** `Ölpreis Tri(20M, 75M, 110M)` — Triangular
2. **Section 3.2.2:** Lognormal `LogN(μ, σ²)`

The model definition at line 321 uses `P_i ~ LogN(μ_p, σ_p²)`, contradicting the triangular calibration.

**Severity:** CRITICAL. The reader cannot determine which distribution was used.

### 1.3 CRITICAL: Oil Price Unit Error

**Location:** Line 273

```
Ölpreis Tri(20M, 75M, 110M) — Ölpreisschwankung
```

Oil price is $/barrel, not millions. $20M–$110M/barrel is absurd. The "M" was likely copied from CAPEX/OPEX/Volume parameters.

**Severity:** CRITICAL. Suggests the authors do not understand their own model parameters.

### 1.4 HIGH: Sample Variance Formula

**Location:** Line 336

Uses population variance (N) instead of sample variance (N−1). Technically incorrect.

### 1.5 MEDIUM: ROI Formula Coupling

**Location:** Line 329

CAPEX appears in both numerator (via Gesamtkosten) and denominator, creating non-linear coupling that is not discussed.

---

## 2. Methodological Issues

### 2.1 CRITICAL: No Actual Simulation Performed

The paper presents results with approximate values (~45%, ~38%, ~25%, etc.) and contains empty figure placeholders instead of actual simulation output. The results appear fabricated or guessed.

### 2.2 HIGH: No Convergence Analysis

N=10,000 claimed as sufficient without standard errors or confidence intervals.

### 2.3 HIGH: Independence Assumption Artifacts

Sensitivity attributing 55% variance to oil price is partly an artifact of using LogN (oil) vs Triangular (others) with vastly different variances.

### 2.4 HIGH: Original ROI Values Are Inconsistent With Own Model

Using the paper's own (incorrect) formula and stated parameters:
- Revenue = $70 × 150M = $10.5B
- Cost (incorrect) = $750M + $120M = $870M
- ROI = ($10.5B − $870M) / $750M ≈ 1284%

The reported ~45% does not match their own stated model parameters.

---

## 3. Structural Issues

- **Uncited references** in bibliography: marathe2005, trigeorgis1996, mun2006
- **Inappropriate citation**: Weber & Schäffer (controlling textbook) cited for elementary ROI formula
- **Incomplete thought**: DuPont system reference (line 156) is a non-sequitur
- **Abstract mismatch**: mentions "Öl- und Gasindustrie" but only models oil

---

## 4. Summary of Corrections Made in main.tex v3.0

| # | Issue | Fix Applied |
|---|-------|-------------|
| 1 | OPEX time dimension | Scaled by project life T=10 years; clarified formula |
| 2 | Dual oil price distribution | Removed contradictory Tri calibration; kept LogN only |
| 3 | Oil price unit error | Removed "M" suffix; consistent $/barrel units |
| 4 | No actual simulation | Ran N=10,000 Monte Carlo; reported exact computed values |
| 5 | Sample variance formula | Changed from N to N−1 denominator |
| 6 | Uncited references | Removed marathe2005, trigeorgis1996, mun2006 |
| 7 | Inappropriate citation | Removed weber2016; replaced with standard formula |
| 8 | DuPont non-sequitur | Removed incomplete DuPont reference |
| 9 | Abstract mismatch | Changed to "Ölindustrie" |
| 10 | Missing seed | Added explicit seed 12345 |
| 11 | PERT confusion | Removed PERT references (not used in model) |
| 12 | Non-standard notation | Changed Ē to \bar{R} |
| 13 | Figure placeholders | Updated with computed data; noted actual figures as TO-GENERATE |
