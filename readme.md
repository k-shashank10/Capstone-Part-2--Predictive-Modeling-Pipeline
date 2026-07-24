# Part 2: Predictive Modeling Pipeline

This part of the project builds a self-contained machine learning pipeline using the IPL matches dataset. The goal is to predict which team will win a match using only pre-match conditions (before ball one is bowled).

---

## Task 1: Problem Framing

### Business Problem
We want to answer a straightforward question: **Can we predict the winner of an IPL match using only pre-match conditions?** 

Predicting match outcomes ahead of time is useful for sports analytics, team strategy, and media broadcasting. To keep the model realistic, we only use information available before the match starts.

### Target & Features
* **Target (`y`):** `winner` — A multi-class categorical column representing the winning team.
* **Features (`X`):**
  * `team1` — First listed team (home team)
  * `team2` — Second listed team (away team)
  * `toss_winner` — Team that won the coin toss
  * `toss_decision` — Decision to bat or field first (`bat` or `field`)
  * `city` — Location where the match took place
  * `season_year` — The year of the tournament

To avoid data leakage, all in-game performance metrics like `result_margin`, `player_of_match`, `target_runs`, and ball-by-ball delivery stats were dropped from `X`.

---

## Data Cleaning (Self-Contained)

Part 2 runs independently and does not import processed files from Part 1. It loads directly from the raw `matches.csv` file and applies basic cleaning:
1. Removed matches where `winner` was marked as `'No Result'` (rained out or abandoned).
2. Mapped older franchise names to their current team names (e.g., *Delhi Daredevils* to *Delhi Capitals*, *Kings XI Punjab* to *Punjab Kings*).

---
 ### Multi-Dataset Ingestion & Feature Engineering
To enrich pre-match feature vectors, historical delivery records from `deliveries.csv` were aggregated alongside `matches.csv`:
* **`avg_city_1st_inn_runs` (Numeric Feature):** Calculated as the historical average 1st-innings total score recorded in a given host city across all prior matches.
* **Leakage Avoidance:** In-match delivery stats for the active match being evaluated were strictly excluded.
---

## Task 2: Feature Preparation & Leakage Prevention

### 1. Train-Test Split First
To prevent test set statistics from leaking into our preprocessing steps, the data was split before applying any encoders or scalers:
* **Split Ratio:** 80% Training (`X_train`, `y_train`), 20% Testing (`X_test`, `y_test`)
* **Stratification:** Stratified on `y` to preserve team win distributions across splits.
* **Random State:** Set to `42` for exact reproducibility.

### 2. Categorical Encoding
* **Technique:** `OneHotEncoder(handle_unknown='ignore')`
* **Reasoning:** Features like `team1`, `team2`, `toss_winner`, `toss_decision`, and `city` are nominal categorical variables with no inherent rank or order. One-hot encoding avoids introducing artificial numeric relationships between teams or venues.
* **Leakage Control:** The encoder was fit strictly on `X_train` inside a Scikit-Learn `ColumnTransformer` and used only to transform `X_test`.

### 3. Feature Scaling
* **Technique:** `StandardScaler()`
* **Reasoning:** Applied to `season_year` so distance- and gradient-based models (like Logistic Regression) don't weight recent years heavier simply due to raw numeric scale.
* **Leakage Control:** Fit exclusively on `X_train` and applied to `X_test` in transform mode.

---

## Task 3: Model Training & Evaluation

### Primary Metric Definition
Since `winner` is a multi-class target with 10+ teams, **Macro-Averaged F1-Score (`macro-F1`)** was selected as the primary metric for all model comparisons, cross-validation, and tuning. 

Macro-F1 calculates the F1-score for each team independently and averages them equally. This ensures teams with fewer total matches carry equal weight in evaluation, preventing dominant teams (like Chennai Super Kings or Mumbai Indians) from skewing overall performance.

---

### Baseline Model Performance (Evaluated on Holdout Test Set)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 (Primary) | Weighted F1 |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.526 | 0.481 | 0.492 | **0.485** | 0.518 |
| **Decision Tree Classifier** | 0.468 | 0.435 | 0.441 | **0.437** | 0.465 |
| **Random Forest Classifier** | 0.552 | 0.512 | 0.520 | **0.514** | 0.546 |
| **XGBoost Classifier** | 0.538 | 0.498 | 0.505 | **0.501** | 0.531 |

Random Forest performed best among baseline models with a holdout Macro-F1 score of **0.514**.

---

## Task 4: Pipeline Integration & Hyperparameter Tuning

## Task 4: Pipeline Integration, Cross-Validation & Hyperparameter Tuning

### 1. Data Leakage-Free Pipeline Construction
To ensure that all preprocessing operations (encoding categorical variables and scaling numeric variables) are computed fresh on every single cross-validation fold, the preprocessing steps and classifier were encapsulated inside a unified Scikit-Learn `Pipeline`.

By wrapping the preprocessor and estimator together, OneHotEncoder and StandardScaler fit exclusively on the training split of each individual fold during cross-validation, guaranteeing zero test-fold information leaks into feature transformations.

### 2. Stratified Cross-Validation
Because match wins across multi-class team targets can be slightly imbalanced, we validated the baseline pipeline using 5-Fold Stratified Cross-Validation (StratifiedKFold(n_splits=5, shuffle=True, random_state=42)).

Primary Metric: Macro-Averaged F1-Score (macro-F1)

Baseline Pipeline Mean CV Score: 0.508

Standard Deviation Across Folds: ± 0.023

### 3. Hyperparameter Tuning (GridSearchCV)
Using the unified pipeline as the estimator, we executed a cross-validated grid search (GridSearchCV) across two key tree-building parameters in the Random Forest model:

Hyperparameter Grid Searched:

classifier__n_estimators: [100, 200, 300] (Number of trees in forest)

classifier__max_depth: [None, 10, 20] (Maximum tree depth to prevent overfitting)

classifier__min_samples_split: [2, 5, 10] (Minimum samples required to split an internal node)

Best Hyperparameter Combination Found:

n_estimators: 200

max_depth: 10

min_samples_split: 5

Tuned Model Cross-Validated Macro-F1 Score: 0.529 (an improvement of +0.021 over the default baseline pipeline).

## Task 5: Model Ranking & Recommendation

### Model Comparison Table

All trained models were evaluated and ranked by our primary metric, **Macro-Averaged F1-Score (`macro-F1`)**. Using macro-averaging treats every IPL franchise equally regardless of total historical matches played, preventing high-frequency winning teams from biasing the overall score.

| Rank | Model / Configuration | Primary Metric (Macro-F1) | Holdout Accuracy | Validation Method |
|:---:|---|:---:|:---:|---|
| **1** | **Tuned Random Forest (GridSearch)** | **0.529** | **0.565** | 5-Fold Stratified CV |
| **2** | Baseline Random Forest | **0.514** | 0.552 | Holdout Test Set |
| **3** | XGBoost Classifier | **0.501** | 0.538 | Holdout Test Set |
| **4** | Logistic Regression | **0.485** | 0.526 | Holdout Test Set |
| **5** | Decision Tree Classifier | **0.437** | 0.468 | Holdout Test Set |

---

### Deployment Recommendation

I recommend deploying the **Tuned Random Forest Pipeline (`n_estimators=200`, `max_depth=10`, `min_samples_split=5`)**. 

Predicting cricket match winners purely from pre-match conditions is inherently noisy because crucial real-time factors—such as pitch deterioration, weather shifts, and individual player form—are unobserved in pre-game metadata. Setting `max_depth=10` keeps the decision trees from memorizing venue-specific noise while achieving the highest overall macro-averaged F1 score (**0.529**). Furthermore, exporting the entire preprocessing and model pipeline as a single `joblib` artifact (`best_ipl_winner_pipeline.pkl`) ensures future pre-match data can be processed and predicted without risking preprocessing mismatch or data leakage.