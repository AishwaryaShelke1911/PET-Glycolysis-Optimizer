# Data Dictionary

## Sheet: Glycolysis_Dataset (main ML dataset)

| Column | Type | Description |
|---|---|---|
| Paper Citation | string | Source paper (author, journal, year) |
| Catalyst | categorical | Catalyst name/type used in the reaction |
| Catalyst Dosage (wt% or other) | string/numeric | Catalyst loading (wt%, g, or mol%) |
| Temp (°C) | numeric | Reaction temperature in Celsius |
| Time (min) | numeric | Reaction time in minutes |
| EG:PET Ratio / EG dosage | string | Ethylene glycol to PET ratio (molar, mass, or volume basis — see notes) |
| PET Conversion (%) | numeric | % of PET converted (depolymerized) |
| BHET Yield (%) | numeric | % yield of BHET monomer product |
| Activation Energy (kJ/mol) | numeric | Reported activation energy, if available |
| Notes | string | Context, caveats, optimal-condition flags |

**Target variables for ML**: `PET Conversion (%)`, `BHET Yield (%)` (BHET Yield is primary target)

**Key features for ML**: Catalyst (encoded), Catalyst Dosage (parsed to numeric wt%),
Temp, Time, EG:PET ratio (parsed to numeric)

## Bonus Sheets (not used in main ML model, available for extensions)

- **Activation_Energy_Comparison**: Cross-paper activation energy comparison table
- **BHET_Purity_Quality**: Product purity (mol%), Mn, color (L*), Zn content data
- **Enzymatic_TPA_Conversion**: Downstream enzymatic BHET→TPA conversion data
- **Heavy_Metal_Removal**: Metal contaminant removal via resin purification
- **Enzymatic_PETase_Degradation**: Pure enzymatic (PETase) PET degradation data — separate process, not glycolysis
- **Fe3O4_Metal_Impurity_Removal**: Metal impurity data specific to Fe3O4-catalyzed BHET

## Literature Sources

1. Zhang, He, Shen, Li, Ye — Ind. Eng. Chem. Res. 2024, 63, 10903–10913 (ionic liquids)
2. Swapna, Singh, Patowary, Bharali, Madras, Sudarsanam — Catal. Sci. Technol. 2024, 14, 5574–5587 (MnOx nanocatalyst)
3. Amundarain et al. — Polymers 2024, 16, 142 (bio-based glycol, real waste streams)
4. Gabrič, Lavrič, Schwiderski, Marc, Temmel, Grilc, Likozar — Polymers 2025, 17, 2246 (kinetic modeling)
5. Mehjabeen, Tian, Rui, Yan, Hu, Wang — Applied Catalysis A General 2026, 711, 120730 (MgO/Y2O3)
6. Ha, Rashid, Ha, Yoo, Jeon, Jeong, Kim — Chemosphere 2024, 349, 140781 (microwave + DES)
7. Pan, Qi, You, Gao, Zhou, Jiang, Wang, Su, Qi — Biochemical Engineering Journal 2024, 207, 109328 (NaHCO3, enzymatic upgrade)
8. Hwang, Han — Chemical Engineering Science 2026, 322, 123144 (kinetic rate constants)
9. Aer, Jiang, Gul, Qi, Feng, Tang — Environmental Research 2022, 212, 113472 (PETase expression/kinetics)
10. Zhong-Johnson, Voigt, Sinskey — Scientific Reports 2021, 11, 928 (PETase thermostability)
11. Jo, Kim, Kim, An — Green Chemistry 2023, 25, 8160–8171 (Fe3O4/CoFe2O4 nanoparticles)

## Known Data Limitations

- Some figures report ranges or graph-only values that could not be extracted as exact
  numbers from text (flagged in the Notes column) — these rows have blank yield/conversion
  cells and should be excluded from training, or the graphs should be manually digitized
  for more precision.
- EG:PET ratio is reported in mixed units across papers (molar, mass, volume) — must be
  standardized during preprocessing.
- Catalyst dosage units vary (wt%, g, mol%) — must be parsed/normalized per catalyst type.
