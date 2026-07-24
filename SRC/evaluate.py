import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

try:
    from SRC.pipeline_builder import build_pipeline
except ModuleNotFoundError:
    from pipeline_builder import build_pipeline


def evaluate_model(pipeline, X_train, y_train, X_test, y_test, model_name="Model"):
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    macro_prec = precision_score(y_test, preds, average='macro', zero_division=0)
    macro_rec = recall_score(y_test, preds, average='macro', zero_division=0)
    macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
    
    return {
        'Model': model_name,
        'Accuracy': round(acc, 4),
        'Macro Precision': round(macro_prec, 4),
        'Macro Recall': round(macro_rec, 4),
        'Macro F1 (Primary)': round(macro_f1, 4),
        'Weighted F1': round(weighted_f1, 4)
    }


def compare_models(X_train, y_train, X_test, y_test):
    candidate_models = {
        'XGBoost Classifier': XGBClassifier(random_state=42, eval_metric='mlogloss'),
        'Decision Tree Classifier': DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest Classifier': RandomForestClassifier(random_state=42)
    }
    
    results = []
    for name, model in candidate_models.items():
        pipe = build_pipeline(model=model)
        metrics = evaluate_model(pipe, X_train, y_train, X_test, y_test, model_name=name)
        results.append(metrics)
        
    results_df = pd.DataFrame(results).sort_values(by='Macro F1 (Primary)', ascending=False).reset_index(drop=True)
    return results_df


def save_evaluation_plots(best_pipeline, X_test, y_test, results_df, output_dir="reports/figures"):
    """Generates and saves comprehensive data EDA and model evaluation plots."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Model Comparison Chart
    plt.figure(figsize=(9, 5))
    df_melted = results_df.melt(id_vars=['Model'], value_vars=['Accuracy', 'Macro F1 (Primary)'], 
                                var_name='Metric', value_name='Score')
    sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric', palette='Blues_d')
    plt.title("Baseline Model Comparison (Test Set)", fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300)
    plt.close()

    # 2. Confusion Matrix Heatmap
    y_pred = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix (Tuned Model)", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # 3. Feature Importance Plot (if available in the classifier)
    try:
        classifier = best_pipeline.named_steps['classifier']
        preprocessor = best_pipeline.named_steps['preprocessor']
        
        # Get feature names from column transformer
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        encoded_cat_cols = list(cat_encoder.get_feature_names_out(['team1', 'team2', 'toss_winner', 'toss_decision', 'city']))
        all_features = ['season_year', 'avg_city_1st_inn_runs'] + encoded_cat_cols
        
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            if len(importances) == len(all_features):
                feat_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
                feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
                
                plt.figure(figsize=(10, 6))
                sns.barplot(data=feat_df, x='Importance', y='Feature', palette='crest')
                plt.title("Top 10 Feature Importances (Tuned Model)", fontsize=14, fontweight='bold')
                plt.xlabel("Importance Score")
                plt.ylabel("Feature")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300)
                plt.close()
    except Exception as e:
        print(f"Note: Could not generate feature importance plot automatically ({e})")

    # 4. Dataset Exploratory Graphs (Matches per Season & Top Venues)
    raw_data_path = os.path.join("Data", "Raw Data", "matches.csv")
    if os.path.exists(raw_data_path):
        raw_matches = pd.read_csv(raw_data_path)
        raw_matches['season_year'] = raw_matches['season'].astype(str).str[:4].astype(int)

        # Matches per Season Bar Chart
        plt.figure(figsize=(10, 5))
        sns.countplot(x='season_year', data=raw_matches, palette='viridis')
        plt.title("IPL Matches Played Per Season", fontsize=14, fontweight='bold')
        plt.xlabel("Season Year")
        plt.ylabel("Match Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "matches_per_season.png"), dpi=300)
        plt.close()

        # Top Host Cities Horizontal Bar Chart
        plt.figure(figsize=(10, 6))
        top_cities = raw_matches['city'].value_counts().head(10).reset_index()
        top_cities.columns = ['City', 'Count']
        sns.barplot(data=top_cities, x='Count', y='City', palette='magma')
        plt.title("Top 10 IPL Host Cities/Venues", fontsize=14, fontweight='bold')
        plt.xlabel("Number of Matches")
        plt.ylabel("City")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_venues.png"), dpi=300)
        plt.close()

        # Team Wins Distribution
        plt.figure(figsize=(10, 6))
        top_winners = raw_matches['winner'].value_counts().reset_index()
        top_winners.columns = ['Team', 'Wins']
        sns.barplot(data=top_winners, x='Wins', y='Team', palette='cubehelix')
        plt.title("Total Historical Match Wins Per Team", fontsize=14, fontweight='bold')
        plt.xlabel("Total Wins")
        plt.ylabel("Team")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "team_wins_distribution.png"), dpi=300)
        plt.close()

    print(f"\nAll comprehensive evaluation and EDA plots successfully saved to '{output_dir}/' folder!")