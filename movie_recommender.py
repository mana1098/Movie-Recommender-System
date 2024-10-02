import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error
from flask import Flask, render_template, request

# Load and preprocess data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Create user-item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Compute user similarity matrix
user_similarity = cosine_similarity(user_item_matrix)

# Collaborative Filtering Recommender
def get_recommendations(user_id, n_recommendations=5):
    user_ratings = user_item_matrix.loc[user_id]
    similar_users = user_similarity[user_id]
    
    # Get top similar users (excluding the user itself)
    similar_users_indices = similar_users.argsort()[::-1][1:11]
    
    # Get movies that the user hasn't rated
    unrated_movies = user_ratings[user_ratings == 0].index
    
    # Predict ratings for unrated movies
    predicted_ratings = {}
    for movie in unrated_movies:
        weighted_sum = 0
        similarity_sum = 0
        for similar_user in similar_users_indices:
            if user_item_matrix.iloc[similar_user][movie] > 0:
                weighted_sum += user_similarity[user_id][similar_user] * user_item_matrix.iloc[similar_user][movie]
                similarity_sum += user_similarity[user_id][similar_user]
        if similarity_sum > 0:
            predicted_ratings[movie] = weighted_sum / similarity_sum
    
    # Sort predicted ratings and get top n recommendations
    sorted_predictions = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)
    top_recommendations = sorted_predictions[:n_recommendations]
    
    return [(movies.loc[movies['movieId'] == movie_id, 'title'].iloc[0], rating) for movie_id, rating in top_recommendations]

# Evaluate the model
def evaluate_model():
    # Split the data into training and testing sets
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    
    # Create user-item matrices for training and testing
    train_matrix = train.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    test_matrix = test.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    # Compute user similarity matrix based on training data
    train_user_similarity = cosine_similarity(train_matrix)
    
    # Make predictions for test set
    predictions = []
    actual_ratings = []
    
    for user_id in test['userId'].unique():
        user_ratings = test_matrix.loc[user_id]
        similar_users = train_user_similarity[user_id]
        similar_users_indices = similar_users.argsort()[::-1][1:11]
        
        for movie_id in user_ratings[user_ratings > 0].index:
            weighted_sum = 0
            similarity_sum = 0
            for similar_user in similar_users_indices:
                if train_matrix.iloc[similar_user][movie_id] > 0:
                    weighted_sum += train_user_similarity[user_id][similar_user] * train_matrix.iloc[similar_user][movie_id]
                    similarity_sum += train_user_similarity[user_id][similar_user]
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                predictions.append(predicted_rating)
                actual_ratings.append(user_ratings[movie_id])
    
    # Calculate Mean Absolute Error
    mae = mean_absolute_error(actual_ratings, predictions)
    return mae

# Flask web application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    recommendations = get_recommendations(user_id)
    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    print(f"Model MAE: {evaluate_model()}")
    app.run(debug=True)
