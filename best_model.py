import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import GridSearchCV, train_test_split

# Step 1: Load datasets
ratings_file = 'data_movie_lens_100k/ratings_all_development_set.csv'
leaderboard_file = 'data_movie_lens_100k/ratings_masked_leaderboard_set.csv'
movie_info_file = 'data_movie_lens_100k/movie_info.csv'
user_info_file = 'data_movie_lens_100k/user_info.csv'

ratings_data = pd.read_csv(ratings_file)
leaderboard_data = pd.read_csv(leaderboard_file)
movie_info = pd.read_csv(movie_info_file)
user_info = pd.read_csv(user_info_file)

# Display the first few rows of the datasets
print("Ratings Data:\n", ratings_data.head())
print("Leaderboard Data:\n", leaderboard_data.head())
print("Movie Info:\n", movie_info.head())
print("User Info:\n", user_info.head())

# Step 2: Prepare ratings data for the surprise library
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_data[['user_id', 'item_id', 'rating']], reader)

# Step 3: Split data into training and validation sets
trainset, validset = train_test_split(data, test_size=0.2)

# Step 4: Define parameter grid for hyperparameter tuning
param_grid = {
    'n_factors': [10, 20, 50, 100],          # Latent factors
    'n_epochs': [20, 50, 100, 500],          # Number of epochs
    'lr_all': [0.002, 0.005, 0.01],     # Learning rate
    'reg_all': [0.02, 0.1, 0.4]         # Regularization term
}

# Step 5: Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(SVD, param_grid, measures=['mae'], cv=3)
grid_search.fit(data)

# Best parameters and model
best_params = grid_search.best_params['mae']
print("Best parameters:", best_params)

best_model = grid_search.best_estimator['mae']

# Step 6: Train the best model on the full training dataset
trainset = data.build_full_trainset()
best_model.fit(trainset)

# Step 7: Evaluate the model on the validation set
predictions = best_model.test(validset)
mae = accuracy.mae(predictions, verbose=True)
print("Validation MAE:", mae)


# Step 8: Make predictions for the leaderboard set
# Generate predictions for leaderboard user-item pairs
#leaderboard_data['predicted_rating'] = leaderboard_data.apply(
    #lambda row: best_model.predict(row['user_id'], row['item_id']).est, axis=1
#)

# Save predictions to file for submission
#leaderboard_data[['user_id', 'item_id', 'predicted_rating']].to_csv(
    #'predictions_leaderboard.csv', index=False
#)

print("Predictions saved to 'predictions_leaderboard.csv'.")
