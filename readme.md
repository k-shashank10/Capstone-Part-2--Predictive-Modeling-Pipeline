# Part 2: Predictive Modeling Pipeline

This part of the project builds a self-contained machine learning pipeline using the IPL matches dataset. The goal is to predict which team will win a match using only pre-match conditions (before ball one is bowled).

---

## Project Structure

```text
Part 2/
│
├── Data/
│   └── Raw Data/
│       ├── matches.csv              # Historical IPL match records
│       └── deliveries.csv           # Ball-by-ball match delivery data
│
├── models/
│   └── ipl_winner_pipeline.pkl      # Serialized scikit-learn pipeline & LabelEncoder artifact
│
├── reports/
│   └── figures/                     # Automated EDA & evaluation visualization charts
│       ├── confusion_matrix.png
│       ├── feature_importance.png
│       ├── model_comparison.png
│       ├── matches_per_season.png
│       ├── top_venues.png
│       ├── team_wins_distribution.png
│       ├── outliers_before_distribution.png
│       └── outliers_after_distribution.png
│
├── SRC/
│   ├── data_loader.py               # Data ingestion, cleaning, team mapping & VIF check
│   ├── pipeline_builder.py          # ColumnTransformer preprocessing pipeline builder
│   ├── evaluate.py                  # Baseline model evaluation & automated plot generator
│   └── predict.py                   # Script to load saved model and predict match winners
│
├── main.py                          # Master orchestration script (Training, Tuning, Saving)
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```
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

### 1. Data Cleaning (Self-Contained)

Part 2 runs independently and does not import processed files from Part 1. It loads directly from the raw `matches.csv` file and applies basic cleaning:
1. Removed matches where `winner` was marked as `'No Result'` (rained out or abandoned).
2. Mapped older franchise names to their current team names (e.g., *Delhi Daredevils* to *Delhi Capitals*, *Kings XI Punjab* to *Punjab Kings*).

---
 ### 2. Multi-Dataset Ingestion & Feature Engineering
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
| **Logistic Regression** | **0.5780** | 0.5716 | 0.4676 | **0.4930** | 0.5715 |
| **Decision Tree Classifier** | **0.5046** | 0.4940 | 0.5156 | **0.5018** | 0.5001 |
| **Random Forest Classifier** | **0.5459** | 0.5072 | 0.4672 | **0.4693** | 0.5307 |
| **XGBoost Classifier** | **0.5000** | 0.5223 | 0.5084 | **0.4980** | 0.4962 |

Decision Tree Classifier performed best among baseline models with a holdout Macro-F1 score of **0.5018**. Decision Tree Classifier performed best among baseline models on macro-F1 with a holdout score of 0.5018, while Logistic Regression achieved the highest baseline test accuracy at 0.5780.
---


## Task 4: Pipeline Integration, Cross-Validation & Hyperparameter Tuning

### 1. Data Leakage-Free Pipeline Construction
To ensure that all preprocessing operations (encoding categorical variables and scaling numeric variables) are computed fresh on every single cross-validation fold, the preprocessing steps and classifier were encapsulated inside a unified Scikit-Learn `Pipeline`.

By wrapping the preprocessor and estimator together, OneHotEncoder and StandardScaler fit exclusively on the training split of each individual fold during cross-validation, guaranteeing zero test-fold information leaks into feature transformations.

### 2. Stratified Cross-Validation
Because match wins across multi-class team targets can be slightly imbalanced, we validated the baseline pipeline using 5-Fold Stratified Cross-Validation (StratifiedKFold(n_splits=5, shuffle=True, random_state=42)).

Primary Metric: Macro-Averaged F1-Score (macro-F1)

Baseline Pipeline Mean CV Score: **0.3398**


### 3. Hyperparameter Tuning (GridSearchCV)
Using the unified pipeline as the estimator, we executed a cross-validated grid search (GridSearchCV) across two key tree-building parameters in the Random Forest model:

Hyperparameter Grid Searched:

classifier__n_estimators: [100, 200, 300] (Number of trees in forest)

classifier__max_depth: [None, 10, 20] (Maximum tree depth to prevent overfitting)

classifier__min_samples_split: [2, 5, 10] (Minimum samples required to split an internal node)

Best Hyperparameter Combination Found:

* n_estimators: 200
* max_depth: 10
* min_samples_split: 5

**Tuned Model Test PErformance** 
1. Test Accuracy: **0.5413**
2. Test Macro-F1 Score: **0.4665**

## Task 5: Model Ranking & Recommendation

### Model Comparison Table

All trained models were evaluated and ranked by our primary metric, **Macro-Averaged F1-Score (`macro-F1`)**. Using macro-averaging treats every IPL franchise equally regardless of total historical matches played, preventing high-frequency winning teams from biasing the overall score.

| Rank | Model / Configuration | Primary Metric (Macro-F1) | Holdout Accuracy | Validation Method |
|:---:|---|:---:|:---:|---|
| **1** | **Decision Tree Classifier** | **0.5018** | **0.5046** | Holdout Test Set |
| **2** | XGBoost ClassifierBaseline Random Forest | **0.4980** | 0.5000 | Holdout Test Set |
| **3** | Logistic Regression | **0.4930** | 0.5780 | Holdout Test Set |
| **4** | Random Forest Classifier | **0.4693** | 0.5459 | Holdout Test Set |
| **5** | Tuned Random Forest (GridSearch) | **0.4665** | 0.5412 | 5-Fold Stratified CV |

---

### Deployment Recommendation

I recommend deploying the **Decision Tree Classifier** or the baseline **Random Forest Classifier** based on these specific test evaluations, as they achieved superior macro-averaged F1 balance across multi-class team targets for this split (with the Decision Tree securing a baseline macro-F1 of **0.5018** at **50.46% accuracy**, and Random Forest achieving a balanced macro-F1 of **0.4693** at **54.59% accuracy**). 

Predicting cricket match winners purely from pre-match conditions is inherently noisy because crucial real-time factors—such as pitch deterioration, weather shifts, and individual player form—are unobserved in pre-game metadata. Exporting the entire preprocessing and model pipeline as a single `joblib` artifact (`ipl_winner_pipeline.pkl`) ensures future pre-match data can be processed and predicted without risking preprocessing mismatch or data leakage.

---
##  Setup & Execution Instructions

To run this predictive modeling pipeline locally from a clean environment, follow these steps:

### Prerequisites
* Ensure **Python 3.8+** is installed on your local machine.

### 1. Repository Setup
Clone the repository and ensure the raw data directory contains `matches.csv` and `deliveries.csv` inside `Part 2/Data/Raw Data/`.

### 2. Install Dependencies
Open your terminal or PowerShell, navigate to the project directory, and install the required Python packages using the requirements file:
```bash
pip install -r requirements.txt
```
### 3. Run the Orchestration Pipeline
Execute the master orchestration script to load data, build the preprocessing pipelines, train and tune models, evaluate performance, and generate output reports and serialized model artifacts:
```bash
python main.py
```
### 4. 4. Run Predictions
To load the saved model artifact (ipl_winner_pipeline.pkl) and predict match winners for new pre-match data, run the prediction script:
```bash
python SRC/predict.py
```
---

##  References
**Official Documentation & Libraries**
1. **Scikit-Learn Documentation:** For machine learning algorithms, pipeline architecture (Pipeline, ColumnTransformer), categorical encoders (OneHotEncoder), scalers (StandardScaler), and hyperparameter tuning (GridSearchCV). Available at: https://scikit-learn.org/stable/documentation.html

2. **XGBoost Documentation**: For gradient boosted decision tree classifiers and hyperparameter configurations. Available at: https://xgboost.readthedocs.io/

3. **Pandas & NumPy Documentation**: For data manipulation, multi-dataset aggregation, and feature matrix formatting. Available at: https://pandas.pydata.org/docs/ & https://numpy.org/doc/

**Academic Class Notes & Faculty Materials**
1. Lectures & Course Material: Machine Learning and Predictive Modeling Curriculum, Lectures, and Class Notes provided by Masai Faculty.
2. Faculty Codes: Reference scripts, data leakage prevention patterns, pipeline integration techniques, and evaluation blueprints shared during practical lab hours.

**Datasets & External Sources**
IPL Match and Delivery Datasets: Historical Indian Premier League match records (matches.csv) and ball-by-ball delivery statistics (deliveries.csv) shared as part of the Masai project repository.
