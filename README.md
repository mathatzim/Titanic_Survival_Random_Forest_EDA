# Titanic Survival Prediction (Random Forest) + EDA

This project applies **supervised machine learning** to predict whether a passenger survived the Titanic disaster, using the classic Kaggle *Titanic Survival Predictions* dataset (891 passengers).

It includes:
- A **Random Forest** classification pipeline (feature cleaning + encoding + evaluation)
- An **EDA notebook** (Google Colab / Jupyter) with survival patterns by sex, class, age, and fare
- The original coursework presentation (`docs/Presentation1.pptx`) and a Colab link

## Dataset
- Source: Kaggle Titanic `TitanicSurvival.xlsx` (891 rows, 12 columns)
- Target: `Survived` (1 = survived, 0 = not survived)

### Features used for modeling
- **Kept:** `Age`, `Sex`, `Pclass`, `SibSp`, `Parch`, `Fare`, `Embarked`
- **Dropped:** `Name`, `Ticket`, `Cabin`, `PassengerId`

### Preprocessing
- Missing `Age` filled with median
- Missing `Embarked` filled with mode
- `Sex` encoded as binary (female=1, male=0)
- `Embarked` one-hot encoded

## Results (Random Forest)
Model: `RandomForestClassifier(n_estimators=100, class_weight="balanced")`  
Train/Test split: 90% / 10% (random_state=42)

- **Accuracy:** **90%**
- Confusion matrix is saved to: `outputs/figures/confusion_matrix.png`

## How to run

### Install
```bash
pip install -r requirements.txt
```

### Train + evaluate
```bash
python -m src.modeling.train_random_forest
```

Outputs:
- `outputs/metrics.json`
- `outputs/figures/confusion_matrix.png`

## EDA notebook
- Notebook: `notebooks/titanic_eda.ipynb`
- Colab link: `docs/colab_link.txt`

## Repo structure
- `src/modeling/train_random_forest.py` — training + evaluation + plot export
- `data/raw/` — dataset files (`TitanicSurvival.xlsx` used in the coursework + Kaggle-style `train.csv`)
- `notebooks/` — EDA notebook
- `docs/` — presentation and Colab link
