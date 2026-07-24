from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def build_pipeline(model=None):
    # If no model is given, default to Random Forest
    if model is None:
        model = RandomForestClassifier(random_state=42)

    # Separate our numbers from our text/categories
    numeric_features = ['season_year', 'avg_city_1st_inn_runs']
    categorical_features = ['team1', 'team2', 'toss_winner', 'toss_decision', 'city']

    # For numbers: fill missing blanks with the median value, then scale them
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # For text/categories: fill blanks with the most common word, then turn words into numbers
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine both rules into one single preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Bundle preprocessing and the AI model together
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    return pipeline