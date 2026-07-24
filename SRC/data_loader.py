import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_data(data_dir="Data/Raw Data", figures_dir="reports/figures"):
    matches_path = os.path.join(data_dir, "matches.csv")
    deliveries_path = os.path.join(data_dir, "deliveries.csv")

    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)

    # Remove matches with no result
    matches = matches[matches['winner'].notnull() & (matches['winner'] != 'No Result')].copy()

    # Fix old team names so they match current names
    team_map = {
        'Rising Pune Supergiant': 'Rising Pune Supergiants',
        'Delhi Daredevils': 'Delhi Capitals',
        'Kings XI Punjab': 'Punjab Kings',
        'Deccan Chargers': 'Sunrisers Hyderabad',
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru'
    }

    matches['winner'] = matches['winner'].replace(team_map)
    matches['team1'] = matches['team1'].replace(team_map)
    matches['team2'] = matches['team2'].replace(team_map)
    matches['toss_winner'] = matches['toss_winner'].replace(team_map)

    matches['city'] = matches['city'].fillna(matches['city'].mode()[0])
    matches['season_year'] = matches['season'].astype(str).str[:4].astype(int)

    # Calculate average first-innings runs for each city
    first_innings = deliveries[deliveries['inning'] == 1]
    match_totals = first_innings.groupby('match_id')['total_runs'].sum().reset_index()

    matches_temp = matches.merge(match_totals, left_on='id', right_on='match_id', how='left')
    city_runs = matches_temp.groupby('city')['total_runs'].mean().reset_index()
    city_runs.rename(columns={'total_runs': 'avg_city_1st_inn_runs'}, inplace=True)

    matches = matches.merge(city_runs, on='city', how='left')
    matches['avg_city_1st_inn_runs'] = matches['avg_city_1st_inn_runs'].fillna(
        matches['avg_city_1st_inn_runs'].mean()
    )

    feature_cols = [
        'team1',
        'team2',
        'toss_winner',
        'toss_decision',
        'city',
        'season_year',
        'avg_city_1st_inn_runs'
    ]

    X = matches[feature_cols].copy()

    # ==========================================
    # 1. OUTLIER CHECK & DENSITY HISTOGRAMS
    # ==========================================
    os.makedirs(figures_dir, exist_ok=True)
    target_col = 'avg_city_1st_inn_runs'
    
    # Draw histogram BEFORE fixing outliers (with a smooth density curve)
    plt.figure(figsize=(8, 5))
    sns.histplot(X[target_col], kde=True, color='salmon', stat="density", bins=20)
    plt.title("Distribution BEFORE Outlier Handling", fontsize=12, fontweight='bold')
    plt.xlabel("Average City 1st Innings Runs")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "outliers_before_distribution.png"), dpi=300)
    plt.close()

    # Calculate IQR to cap extreme outliers smoothly
    Q1 = X[target_col].quantile(0.25)
    Q3 = X[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Cap the values so extreme spikes don't confuse the AI model
    X[target_col] = np.clip(X[target_col], lower_bound, upper_bound)

    # Draw histogram AFTER fixing outliers
    plt.figure(figsize=(8, 5))
    sns.histplot(X[target_col], kde=True, color='skyblue', stat="density", bins=20)
    plt.title("Distribution AFTER Outlier Handling (Capped)", fontsize=12, fontweight='bold')
    plt.xlabel("Average City 1st Innings Runs")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "outliers_after_distribution.png"), dpi=300)
    plt.close()

    print(f"-> Outlier density histograms saved inside '{figures_dir}/'")

    # ==========================================
    # 2. VIF & MULTICOLLINEARITY TEST
    # ==========================================
    print("\n--- Running VIF & Multicollinearity Test ---")
    X_encoded = pd.get_dummies(X, drop_first=True, dtype=float)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_encoded.columns
    vif_data["VIF"] = [variance_inflation_factor(X_encoded.values, i) for i in range(X_encoded.shape[1])]
    vif_data = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)
    
    print("Top VIF Results (Checking for column overlap):")
    print(vif_data.head(5).to_string())
    print("--------------------------------------------\n")

    # Turn winning team names into integer codes for XGBoost / Random Forest
    le = LabelEncoder()
    y = le.fit_transform(matches['winner'])

    return X, y, le