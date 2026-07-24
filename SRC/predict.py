import os
import joblib
import pandas as pd

# Find the saved model file in the models folder
current_folder = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(current_folder, "..", "models", "ipl_winner_pipeline.pkl")

print("Loading the trained model...")
saved_data = joblib.load(model_file)

# Extract the pipeline and label encoder from the saved file
model_pipeline = saved_data['pipeline']
label_encoder = saved_data['label_encoder']

# Define a sample match you want to predict
# Change these team names or city to test different matches
match_details = {
    'team1': 'Mumbai Indians',
    'team2': 'Chennai Super Kings',
    'toss_winner': 'Mumbai Indians',
    'toss_decision': 'field',
    'city': 'Mumbai',
    'season_year': 2026,
    'avg_city_1st_inn_runs': 175.0
}

# Put the match details into a table format
input_table = pd.DataFrame([match_details])

# Get the win probabilities from the model
probabilities = model_pipeline.predict_proba(input_table)[0]
available_classes = model_pipeline.named_steps['classifier'].classes_

team_a = match_details['team1']
team_b = match_details['team2']

# Find the correct numbers for each team using the label encoder
team_a_code = label_encoder.transform([team_a])[0]
team_b_code = label_encoder.transform([team_b])[0]

# Get the probability score for each team
score_a = probabilities[list(available_classes).index(team_a_code)]
score_b = probabilities[list(available_classes).index(team_b_code)]

# Convert scores into percentages
total = score_a + score_b
percent_a = (score_a / total) * 100
percent_b = (score_b / total) * 100

# Figure out who is predicted to win
if percent_a >= percent_b:
    winner = team_a
else:
    winner = team_b

# Print the final result clearly
print("\n--- IPL Match Prediction ---")
print(f"Match: {team_a} vs {team_b}")
print(f"Venue: {match_details['city']}")
print(f"Winner: {winner}")
print(f"Chances:")
print(f" - {team_a}: {percent_a:.1f}%")
print(f" - {team_b}: {percent_b:.1f}%")
print("----------------------------\n")