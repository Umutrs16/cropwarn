# Anti-leakage rules

These rules are implemented as runtime assertions in
`cropwarn.validation.leakage` and covered by `tests/test_leakage.py`.

1. Rolling trend, z-scores and production anomalies for year *t* use only years ≤ *t*−1.
2. Standardization, PCA, feature selection and conformal calibration are fold-internal.
3. Real-time phenology never uses post-cutoff observations to set EOS, and does
   not use same-year MCD12Q2 senescence as crop EOS. Dynamic SOS is used only
   after greenup has already been observed.
4. Every lead-time feature slice ends at `harvest − lead`.
5. Trade weights come from the previous complete trade year.
6. AlphaEarth annual embeddings are lagged or excluded from early warning.
7. WorldCereal 2021 maps are not copied across 2001–2024 as truth.
8. Country identity and technology level are not required features; dropping them is a robustness check.
9. Oracle columns (`shock_true`, onset times, raw yield) cannot enter `X`.
10. SMAP / L-band VOD features are masked before 2015.
