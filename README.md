# fake-articel-detection


Step 1: Create a Dummy Dataset

import numpy as np
import pandas as pd

# Generate dummy dataset
np.random.seed(42)

# Genuine reviews
genuine_reviews = ["The product is excellent", "I love this product", "Highly recommend", "Very satisfied", "Will buy again"] * 100

# Fake reviews (anomalous)
fake_reviews = ["Buy this product now", "This is a scam", "Don't waste your money", "Worst product ever", "Fake review"] * 20

# Create a DataFrame
reviews = pd.DataFrame({
    'review': genuine_reviews + fake_reviews,
    'label': [0] * len(genuine_reviews) + [1] * len(fake_reviews)  # 0: genuine, 1: fake
})

# Shuffle the dataset
reviews = reviews.sample(frac=1).reset_index(drop=True)
print(reviews.head())


Step 2: Text Preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=100)  # Limit to 100 features for simplicity

# Fit and transform the reviews
X = vectorizer.fit_transform(reviews['review']).toarray()
print(X.shape)

Step 3: Build and Train the Autoencoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Define the autoencoder
input_dim = X.shape[1]
encoding_dim = 32

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder on genuine reviews only
genuine_X = X[reviews['label'] == 0]
autoencoder.fit(genuine_X, genuine_X, epochs=50, batch_size=16, shuffle=True)

Step 4: Detect Fake Reviews
# Compute reconstruction error for all reviews
reconstructions = autoencoder.predict(X)
mse = np.mean(np.power(X - reconstructions, 2), axis=1)

# Set a threshold for the reconstruction error
threshold = np.percentile(mse, 95)  # 95th percentile of the reconstruction error on genuine reviews

# Predict fake reviews
predicted_labels = (mse > threshold).astype(int)

# Compare with actual labels
results = pd.DataFrame({'review': reviews['review'], 'actual': reviews['label'], 'predicted': predicted_labels, 'mse': mse})
print(results.head(20))
