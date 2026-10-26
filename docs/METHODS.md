# Methods contract

CropWarn tests four hypotheses on public labels and satellite series. Official gates are unchanged. The supported journal position is Nature Communications / RSE: trade exposure GO, physiology lead NO-GO.

## Research question

On wheat and soybean breadbaskets, does a lagged trade network identify import-exposed countries before harvest, and does adding GOSIF (or VOD / root-zone soil moisture) move the R² ≥ 0.15 lead earlier than optical + climate?

## Hypotheses and official outcome

| ID | Hypothesis | Official wheat+soy |
|---|---|---|
| **H1** | SIF / VOD / root-zone SM lead NDVI/EVI and raise 64–128 d shock detection | **Null.** B2 and B4 both have `earliest_lead` = 192 d (gain 0). Same on event years / \|z\|≥1. |
| **H2** | Dynamic SOS/EOS helps low-latitude multi-crop systems | **Null.** B1 − B0 low-lat R² ≈ −0.03 (gate ≥ 0.03 and B1 R² > 0). |
| **H3** | Unseen shocks are detectable as shift; 90% conformal coverage holds | **Supported.** Coverage error ≈ 0.008 (tolerance 0.08). |
| **H4** | Yield anomalies + lagged trade identify high import exposure earlier | **Supported.** Trade lead 192 d (gate ≥ 16 d) with harvest R² ≥ 0.15. Risk information, not policy causality. |

Nature Food requires H1 GO **and** H4 GO. That pair is not met. Do not change `physiology_lead_days` or substitute AUROC for `earliest_lead`.

## Targets

| Target | Definition | Leakage rule |
|---|---|---|
| yield_level | t ha⁻¹ | comparable to Chen-style operational systems |
| yield_anomaly | residual after a rolling trend fit on years ≤ t−1 | never a full-period LOESS/HP filter |
| production_shock | yield × harvested area, with disaster-year area states | FAOSTAT end-of-year area is not a real-time feature |

Primary target: `yield_anomaly`. Headline crops: wheat, soybean. FAO rice, seasonal rice, and all maize extracts stay out of the gate sample.

## Models

0. Climatology / linear trend
1. Chen-style phenology-clustered random forest
2. XGBoost on dynamic-phenology features
3. Phenology-aware multimodal temporal encoder (ExtraTrees + MLP)
4. Split conformal 90% intervals
5. Mahalanobis + kNN + energy OOD with reliable / high-risk / reject flags

## Validation (random 80/20 is forbidden)

- Rolling-origin (test years from 2016; **this is the official gate scheme**)
- Leave-one-year-out
- Leave-one-region-out
- Shock holdout (2003, 2010, 2012, 2015–16, 2019–20, 2022)

`earliest_lead` = longest lead with pooled headline-crop R² ≥ 0.15.

Extra schemes (`results/nc_validation/`): shock holdout and leave-one-year-out keep B2/B4 at 192 d. Leave-one-region-out harvest R² is negative.

## Ablation matrix

B0 Chen RF (static) → B1 dynamic RF → B2 optical+climate → B3 +SM → B4 +SIF →
B5 +VOD → P full physiology → P + food system.

H1 is B4 versus B2 on the same model class (XGBoost), not B4 versus B0.

## Robustness (Stage 12)

- Country identity is already excluded from official features.
- Quadratic rolling detrend is the executed label sensitivity (`scripts/run_robustness.py`). Harvest R² falls below 0.15; the official linear trend is the supported label.
- TROPOMI/GOME-2 independence is not available (`tropomi_sif: false`).
- Rice-without-VOD is not applicable on the wheat+soy headline.

## Held-out Stage 11

Same feature contract on China NBS early rice, India APY rice, FAO maize, US NASS maize, and Eurostat NUTS maize. None enter `headline_crops`. See `results/held_out.md`.
