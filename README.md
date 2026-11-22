# PET Glycolysis ML Optimizer

A data-driven optimization tool for catalytic PET (polyethylene terephthalate) glycolysis,
built from a curated literature dataset (11 papers, 100+ experimental data points).

Instead of running physical experiments, this project mines published research on
catalytic PET glycolysis (zinc acetate, metal oxides, ionic liquids, deep eutectic
solvents, MgO/Y2O3, Fe3O4 nanoparticles, etc.) and uses machine learning to:

1. **Predict** PET conversion % and BHET monomer yield % from reaction conditions
   (catalyst type, dosage, temperature, time, EG:PET ratio)
2. **Recommend** optimal reaction conditions for a chosen catalyst to maximize yield
3. **Visualize** how each process variable affects yield, based on real literature data

## Project Motivation

Catalytic glycolysis of PET bottles is a key green-recycling pathway (PET → BHET
monomer → recycled PET). Published studies report wildly different optimal conditions
depending on the catalyst system used. This project consolidates that scattered
literature data into a single structured dataset and builds a predictive/optimization
layer on top — useful for quickly narrowing down promising experimental conditions
before running physical trials.

## Tech Stack

- **Data**: pandas, openpyxl (literature-mined dataset)
- **ML**: scikit-learn, XGBoost (Random Forest / Gradient Boosting regression)
- **Optimization**: scikit-optimize / grid search for condition recommendation
- **App**: Streamlit (interactive prediction + optimization dashboard)

## Folder Structure

```
data/raw/            Original literature dataset (Excel, multi-sheet)
data/processed/       Cleaned CSV ready for ML
notebooks/            EDA, feature engineering, model training notebooks
src/                   Reusable Python modules (data loading, preprocessing, training, prediction)
models/                Saved trained models + encoders
app/                   Streamlit dashboard
reports/figures/       Saved plots for the final report
tests/                 Unit tests
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Clean and prepare data
```bash
python src/data_loader.py
```

### 2. Train the model
```bash
python src/train_model.py
```

### 3. Run the dashboard
```bash
streamlit run app/app.py
```

## Dataset

See `data/data_dictionary.md` for column definitions and full literature citations.
Dataset spans catalyst types including: zinc acetate, ionic liquids ([P4444]Ace),
metal oxides (MnOx, MgO/Y2O3, TiO2, Nb2O5, CeO2), deep eutectic solvents (ChCl-ZnCl2),
sodium bicarbonate, and Fe/Co-based ferrite nanoparticles (Fe3O4, CoFe2O4).

## Results

(Fill in after training: best model, R² score, RMSE, top features by importance)

## Future Work

- Expand dataset with more literature data points
- Add a "product quality" sub-model using the BHET purity dataset
- Explore physics-informed features using Arrhenius kinetic parameters
- Deploy as a public web app
