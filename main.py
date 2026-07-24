import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Ensure SRC directory is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'SRC'))

from data_loader import load_data
from pipeline_builder import build_pipeline
from evaluate import compare_models, evaluate_model, save_evaluation_plots


def main():
    print("Loading data...")
    X, y, le = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nComparing baseline models...")
    baseline_results = compare_models(X_train, y_train, X_test, y_test)
    print(baseline_results.to_string())

    print("\nRunning cross validation...")
    base_pipeline = build_pipeline()
    cv_scores = cross_val_score(base_pipeline, X_train, y_train, cv=5, scoring='f1_macro')
    print(f"Base Random Forest CV Macro-F1: {cv_scores.mean():.4f}")

    print("\nTuning parameters with GridSearchCV...")
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    print("\nBest parameters:")
    print(grid_search.best_params_)

    best_pipeline = grid_search.best_estimator_

    # Evaluate tuned model on holdout test set
    tuned_eval = evaluate_model(best_pipeline, X_train, y_train, X_test, y_test, model_name="Tuned RF")
    print(f"\nTuned Model Test Accuracy: {tuned_eval['Accuracy']:.4f}")
    print(f"Tuned Model Test Macro-F1: {tuned_eval['Macro F1 (Primary)']:.4f}")

    # Save evaluation figures for report
    save_evaluation_plots(best_pipeline, X_test, y_test, baseline_results)

    # Save artifact (Pipeline + LabelEncoder)
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "ipl_winner_pipeline.pkl")

    artifact = {
        'pipeline': best_pipeline,
        'label_encoder': le
    }

    joblib.dump(artifact, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()