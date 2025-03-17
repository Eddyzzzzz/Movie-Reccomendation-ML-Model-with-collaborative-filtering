import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, KNNWithMeans
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class AdvancedHybridRecommender:
    def __init__(self, ratings_file, user_info_file, movie_info_file):
        # Load datasets
        self.ratings_data = pd.read_csv(ratings_file)
        self.user_info = pd.read_csv(user_info_file)
        self.movie_info = pd.read_csv(movie_info_file)
        
        # Prepare comprehensive features
        self.prepare_comprehensive_features()
        
    def prepare_comprehensive_features(self):
        # Enhanced user feature engineering
        self.user_info['age_group'] = pd.cut(
            self.user_info['age'], 
            bins=[0, 18, 25, 35, 45, 55, 100], 
            labels=['Under 18', '18-25', '26-35', '36-45', '46-55', 'Over 55']
        )
        
        # Movie feature enrichment
        # Normalize year and add more derived features
        self.movie_info['movie_age'] = 2024 - self.movie_info['release_year']
        
        # Create decade feature
        self.movie_info['decade'] = (self.movie_info['release_year'] // 10) * 10
        
        # Compute user-level statistics
        self.user_rating_stats = self.ratings_data.groupby('user_id').agg({
            'rating': ['count', 'mean', 'std']
        }).reset_index()
        self.user_rating_stats.columns = ['user_id', 'user_rating_count', 'user_avg_rating', 'user_rating_std']
        
        # Compute movie-level statistics
        self.movie_rating_stats = self.ratings_data.groupby('item_id').agg({
            'rating': ['count', 'mean', 'std']
        }).reset_index()
        self.movie_rating_stats.columns = ['item_id', 'movie_rating_count', 'movie_avg_rating', 'movie_rating_std']
        
        # Merge statistical features
        self.user_feature_matrix = self.user_info.merge(
            self.user_rating_stats, on='user_id', how='left'
        )
        
        self.movie_feature_matrix = self.movie_info.merge(
            self.movie_rating_stats, on='item_id', how='left'
        )
    
    def prepare_surprise_dataset(self):
        # Merge features with ratings
        merged_data = self.ratings_data.merge(
            self.user_feature_matrix, 
            on='user_id', 
            how='left'
        ).merge(
            self.movie_feature_matrix, 
            on='item_id', 
            how='left'
        )
        
        # Prepare Surprise dataset
        reader = Reader(rating_scale=(1, 5))
        
        # Prepare additional features
        feature_columns = [
            # User features
            'is_male', 'age', 
            'user_rating_count', 'user_avg_rating', 'user_rating_std',
            
            # Movie features
            'release_year', 'movie_age', 'decade', 
            'movie_rating_count', 'movie_avg_rating', 'movie_rating_std'
        ]
        
        # Fill NaN with 0 for features
        merged_data[feature_columns] = merged_data[feature_columns].fillna(0)
        
        # Impute and scale features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), 
                 ['age', 'user_rating_count', 'user_avg_rating', 'user_rating_std', 
                  'release_year', 'movie_age', 'movie_rating_count', 
                  'movie_avg_rating', 'movie_rating_std']),
                ('cat', OneHotEncoder(handle_unknown='ignore'), 
                 ['is_male', 'decade'])
            ])
        
        # Fit preprocessor
        preprocessed_features = preprocessor.fit_transform(
            merged_data[feature_columns]
        )
        
        # Create Surprise dataset
        data = Dataset.load_from_df(
            merged_data[['user_id', 'item_id', 'rating']], 
            reader
        )
        
        return data, preprocessed_features
    
    def train_ensemble_model(self):
        # Prepare dataset with features
        data, features = self.prepare_surprise_dataset()
        
        # Corrected KNN parameter grid with supported similarity metrics
        knn_param_grid = {
            # More extensive neighbor count options
            'k': [10, 20, 30, 50, 75, 100, 150, 200],
            
            # Enhanced similarity options
            'sim_options': {
                # Supported similarity metrics
                'name': [
                    'pearson_baseline',  # Pearson correlation with baseline adjustment
                    'cosine',            # Cosine similarity
                    'msd',               # Mean Squared Difference
                    'pearson'            # Standard Pearson correlation
                ],
                
                # Exploration of user-based vs item-based
                'user_based': [True, False],
                
                # Additional similarity configuration
                'min_support': [1, 3, 5],     # Minimum number of common items/users
                'shrinkage': [0, 10, 50, 100] # Pearson baseline shrinkage
            },
            
            # Weight calculation strategies
            'weight_options': {
                'strategy': [
                    'uniform',    # Equal weights
                    'and_ratio'   # Weighted by number of common ratings
                ]
            }
        }
        
        # Improved grid search with multiple metrics and more comprehensive cross-validation
        knn_grid_search = GridSearchCV(
            KNNWithMeans, 
            knn_param_grid, 
            measures=['mae', 'rmse'],  # Multiple evaluation metrics
            cv=5,                      # Increased cross-validation folds
            n_jobs=-1,                 # Use all available CPU cores
        )
        
        try:
            # Fit the grid search on the entire dataset
            knn_grid_search.fit(data)
            
            # Get the best model based on MAE
            best_knn_model = knn_grid_search.best_estimator['mae']
            
            # Print detailed information about the best model
            print("\nBest KNN Model Details:")
            print("Best Parameters (MAE):", knn_grid_search.best_params['mae'])
            print("Best MAE Score:", knn_grid_search.best_score['mae'])
            print("Best RMSE Score:", knn_grid_search.best_score['rmse'])
            
            # Train on full dataset
            full_trainset = data.build_full_trainset()
            best_knn_model.fit(full_trainset)
            
            # Optional: Create a validation set for final evaluation
            _, validset = train_test_split(data, test_size=0.2)
            
            # Evaluate KNN model
            knn_predictions = best_knn_model.test(validset)
            
            # Calculate and print detailed accuracy metrics
            mae_score = accuracy.mae(knn_predictions, verbose=False)
            rmse_score = accuracy.rmse(knn_predictions, verbose=False)
            
            print("\nFinal Model Performance:")
            print(f"MAE: {mae_score}")
            print(f"RMSE: {rmse_score}")
            
            # Optional: Add some additional logging
            print("\nModel Configuration:")
            print(f"Number of Neighbors (k): {best_knn_model.k}")
            print(f"Similarity Metric: {best_knn_model.sim_options['name']}")
            print(f"User-based: {best_knn_model.sim_options['user_based']}")
            
            # Return the best KNN model
            return best_knn_model
        
        except Exception as e:
            print(f"An error occurred during grid search: {e}")
            raise

# Prediction function (kept from previous implementation)
def predict_leaderboard_txt(model, leaderboard_file, output_file='predicted_ratings_leaderboard.txt'):
    # Load leaderboard data
    leaderboard_data = pd.read_csv(leaderboard_file)
    
    # Generate predictions as a numpy array
    predictions = np.array([
        model.predict(row['user_id'], row['item_id']).est 
        for _, row in leaderboard_data.iterrows()
    ])
    
    # Save predictions to a plain text file
    np.savetxt(output_file, predictions)
    
    # Verify the output
    loaded_predictions = np.loadtxt(output_file)
    print(f"Predictions saved to {output_file}")
    print(f"Predictions shape: {loaded_predictions.shape}")
    print(f"First 5 predictions: {loaded_predictions[:5]}")
    
    return predictions

# Usage remains the same
recommender = AdvancedHybridRecommender(
    ratings_file='data_movie_lens_100k/ratings_all_development_set.csv',
    user_info_file='data_movie_lens_100k/user_info.csv',
    movie_info_file='data_movie_lens_100k/movie_info.csv'
)

# Train KNN model
best_model = recommender.train_ensemble_model()

# Save predictions to a plain text file
predictions = predict_leaderboard_txt(
    best_model, 
    'data_movie_lens_100k/ratings_masked_leaderboard_set.csv',
    'predicted_ratings_leaderboard.txt'
)
