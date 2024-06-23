from ucimlrepo import fetch_ucirepo 
# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Fetch dataset
car_evaluation = fetch_ucirepo(id=19) 
car_evaluation_df = car_evaluation.data['original'].drop('persons', axis=1)

# Select relevant columns
features = ['maint', 'doors', 'lug_boot', 'safety', 'class']
X = car_evaluation_df[features]
y = car_evaluation_df['buying']

# Perform one-hot encoding
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X).toarray()

# Encode the target variable
y_encoded = pd.get_dummies(y).values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Build the neural network
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=6, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

