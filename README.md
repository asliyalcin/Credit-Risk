# 💳 Credit Risk Analysis

This project aims to build a machine learning pipeline that predicts **credit default risk** and identifies high-risk customers early.  
It includes **EDA**, **data preprocessing**, **feature engineering**, **sampling methods**, and **model training** (Random Forest & XGBoost).

---
## 📊 Metrics & Business Focus

The target default is highly imbalanced (~3% positives).
Business focus: missing a default (FN) is more costly than a false alarm (FP).
Therefore:
Recall and G-Mean are prioritized.
Threshold is optimized instead of using the default 0.5.
Each run’s detailed metrics and confusion matrix are stored under results/run_*/.

## 📂 Project Structure
```
credit_risk/
│
├── data/ # Raw and processed datasets
├── eda_plots/ # Visual EDA outputs
├── results/ # Model outputs for each run
├── utils/ # Modular Python scripts (cleaning, sampling, modeling)
│
├── EDA.ipynb # Exploratory Data Analysis
├── modelling.ipynb # Model training & evaluation
├── requirements.txt
```


---

## 🔍 EDA & Preprocessing

Main steps in `EDA.ipynb`:

- Load config from `utils/config.yaml`
- Read the raw credit risk dataset from `data/`
- Analyze missing values and handle them with  
  `clean_missing_values()` from `utils.missing_outliar_values`
- Check skewness and apply log transform for selected variables
- Create new features with `create_features()` from `utils.feature_eng`
- Detect potential leakage columns and exclude them from modelling
- Save the processed dataset to the path defined in `config.yaml`


---

## 🤖 Modelling

Main steps in `modelling.ipynb`:

- Load processed data using `paths_model.read_file` from `config.yaml`
- Split into train/test (`train_test_split` with `stratify=y`)
- Scale features with `StandardScaler`
- Apply sampling on the **training set** using  
  `apply_sampling()` from `utils.sampling`  
- Train and tune:

  1. **XGBoost (XGBClassifier)**  
     - Hyperparameter tuning with **Optuna**  
     - Optional feature selection via Optuna feature flags  
     - Best model is evaluated on the test set  

  2. **Logistic Regression**  
     - Hyperparameter tuning with **Optuna**  
     - Trained on resampled & scaled data as a strong baseline

- For each model:
  - Predicted probabilities are passed to `evaluate_with_gmean()`  
  - Best threshold is selected using **G-Mean**  
  - Metrics (recall, precision, f1, g-mean, etc.) are stored

- All results are registered via `ResultsRegistry` from `utils.results` and saved with:

  ```python
  run_dir = registry.save_all(base_dir="results", save_models=True)
