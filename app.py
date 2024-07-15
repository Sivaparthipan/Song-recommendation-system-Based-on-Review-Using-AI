import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from flask import Flask, request, jsonify

# Load and preprocess the data
file_path = 'reviews.csv'
ratings = pd.read_csv(file_path)

# Display the first few rows to understand the structure
print(ratings.head())

# Check for missing values and drop them if any
print(ratings.isnull().sum())
ratings = ratings.dropna()

# Ensure the 'Rating' column is numerical
ratings['Rating'] = pd.to_numeric(ratings['Rating'], errors='coerce')

# Check for any NaN values after conversion
print(ratings.isnull().sum())
ratings = ratings.dropna()

# Generate synthetic user_id and song_id
ratings['user_id'] = ratings.index
ratings['song_id'] = ratings.index

# Verify the data structure
print(ratings.head())

# Create the user-item matrix
try:
    user_item_matrix = ratings.pivot_table(index='user_id', columns='song_id', values='Rating')
    user_item_matrix.fillna(0, inplace=True)
    user_item_matrix_np = user_item_matrix.values
    print("User-item matrix created successfully.")
except Exception as e:
    print(f"Error creating user-item matrix: {e}")

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Train the k-NN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_item_matrix_np)

# Simple Collaborative Filtering Model
class SimpleCollaborativeFiltering:
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix

    def fit(self):
        pass

    def predict(self, user_id, song_id):
        user_ratings = self.user_item_matrix.loc[user_id]
        song_ratings = self.user_item_matrix[song_id]
        user_mean_rating = user_ratings[user_ratings > 0].mean()
        song_mean_rating = song_ratings[song_ratings > 0].mean()
        return (user_mean_rating + song_mean_rating) / 2

simple_cf = SimpleCollaborativeFiltering(user_item_matrix)
simple_cf.fit()

# Function to evaluate the models
def evaluate_model(model, test_data, model_type='knn'):
    if model_type == 'knn':
        predictions = []
        for _, row in test_data.iterrows():
            user_id = row['user_id']
            song_id = row['song_id']
            user_index = user_item_matrix.index.get_loc(user_id)
            distances, indices = model.kneighbors(user_item_matrix_np[user_index, :].reshape(1, -1), n_neighbors=6)
            neighbor_ratings = user_item_matrix_np[indices.flatten()]
            avg_ratings = neighbor_ratings.mean(axis=0)
            sorted_indices = np.argsort(avg_ratings)[::-1]
            predicted_rating = avg_ratings[song_id] if song_id in sorted_indices else np.mean(avg_ratings)
            predictions.append(predicted_rating)
    elif model_type == 'simple_cf':
        predictions = [model.predict(row['user_id'], row['song_id']) for _, row in test_data.iterrows()]
    
    true_ratings = test_data['Rating'].values
    mse = mean_squared_error(true_ratings, predictions)
    return mse

# Evaluate k-NN model
knn_mse = evaluate_model(knn, test_data, model_type='knn')
print(f"k-NN MSE: {knn_mse}")

# Evaluate Simple Collaborative Filtering model
simple_cf_mse = evaluate_model(simple_cf, test_data, model_type='simple_cf')
print(f"Simple CF MSE: {simple_cf_mse}")

# Function to get top N recommendations
def get_top_n_recommendations(user_id, n=10, model=None, model_type='knn'):
    if model_type == 'knn':
        user_index = user_item_matrix.index.get_loc(user_id)
        distances, indices = model.kneighbors(user_item_matrix_np[user_index, :].reshape(1, -1), n_neighbors=n+1)
        neighbor_ratings = user_item_matrix_np[indices.flatten()]
        avg_ratings = neighbor_ratings.mean(axis=0)
        sorted_indices = np.argsort(avg_ratings)[::-1]
        top_n_recommendations = [(user_item_matrix.columns[i], avg_ratings[i]) for i in sorted_indices[:n]]
    elif model_type == 'simple_cf':
        all_songs = user_item_matrix.columns
        predictions = [model.predict(user_id, song_id) for song_id in all_songs]
        top_n_indices = np.argsort(predictions)[-n:][::-1]
        top_n_recommendations = [(all_songs[i], predictions[i]) for i in top_n_indices]
    
    return top_n_recommendations

# Deploy with Flask
app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400
    
    try:
        user_id = int(user_id)
    except ValueError:
        return jsonify({'error': 'Invalid User ID'}), 400
    
    recommendations = get_top_n_recommendations(user_id, model=knn, model_type='knn')
    return jsonify({'user_id': user_id, 'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)