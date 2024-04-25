import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import matplotlib.pyplot as plt

np.random.seed(42) # Set random seed for reproducibility

data = pd.read_csv('Updated_Cleaned_Games_Dataset.csv')
data['Preprocessed_Summary'] = data['Preprocessed_Summary'].astype(str)
X = data['Preprocessed_Summary']
y = data['Global_Sales']

vectorizer = TfidfVectorizer(lowercase=False)
X = vectorizer.fit_transform(X).toarray()

# Training/test split (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential()
model.add(Dense(1, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=regularizers.l2(0.001)))  # Input layer
model.add(Dense(2, activation='relu', kernel_regularizer=regularizers.l2(0.001)))  # Hidden layer
model.add(Dense(1, activation='linear'))  # Output layer

model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, y_train, epochs=40, batch_size=16, validation_data=(X_test, y_test), verbose=0)

# Plotting and error calculation
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

mse = model.evaluate(X_test, y_test)
print("Mean Squared Error on Test Set:", mse)
y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual Values")
plt.show()

residuals = y_test - y_pred.squeeze()

plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
