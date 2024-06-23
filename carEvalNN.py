from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Fetch dataset 
car_evaluation = fetch_ucirepo(id=19) 
car_evaluation_df = car_evaluation.data['original'].drop('persons', axis=1)

# Select relevant columns
features = ['maint', 'doors', 'lug_boot', 'safety', 'class']
X = car_evaluation_df[features]
y = car_evaluation_df['buying']

# Initialize label encoder
label_encoder = LabelEncoder()

# Encode each categorical variable in X
X_encoded = X.apply(label_encoder.fit_transform)
X_encoded['class_normalized'] = X_encoded['class'] / X_encoded['class'].max()

# Encode Target Variable (y)
y_encoded = label_encoder.fit_transform(y)

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Determine number of unique categories for each feature
num_unique_maint = len(car_evaluation_df['maint'].unique())
num_unique_doors = len(car_evaluation_df['doors'].unique())
num_unique_lug_boot = len(car_evaluation_df['lug_boot'].unique())
num_unique_safety = len(car_evaluation_df['safety'].unique())
num_unique_class = len(car_evaluation_df['class'].unique())

# Build Neural Network Model with Embedding Layers
tf.random.set_seed(8)

# Define input shapes for each categorical variable
maint_input = tf.keras.Input(shape=(1,), name='maint')
doors_input = tf.keras.Input(shape=(1,), name='doors')
lug_boot_input = tf.keras.Input(shape=(1,), name='lug_boot')
safety_input = tf.keras.Input(shape=(1,), name='safety')
class_input = tf.keras.Input(shape=(1,), name='class')

# Embedding layers
# for each categorical variable
embedding_size = 8
embedding_maint = tf.keras.layers.Embedding(input_dim=num_unique_maint, output_dim=embedding_size)(maint_input)
embedding_doors = tf.keras.layers.Embedding(input_dim=num_unique_doors, output_dim=embedding_size)(doors_input)
embedding_lug_boot = tf.keras.layers.Embedding(input_dim=num_unique_lug_boot, output_dim=embedding_size)(lug_boot_input)
embedding_safety = tf.keras.layers.Embedding(input_dim=num_unique_safety, output_dim=embedding_size)(safety_input)
embedding_class = tf.keras.layers.Embedding(input_dim=num_unique_class, output_dim=embedding_size)(class_input)

# Flatten embeddings and concatenate into a single vector
flatten_maint = tf.keras.layers.Flatten()(embedding_maint)
flatten_doors = tf.keras.layers.Flatten()(embedding_doors)
flatten_lug_boot = tf.keras.layers.Flatten()(embedding_lug_boot)
flatten_safety = tf.keras.layers.Flatten()(embedding_safety)
flatten_class = tf.keras.layers.Flatten()(embedding_class)

# Concatenate all embeddings into a single vector
concatenated = tf.keras.layers.Concatenate()([flatten_maint, flatten_doors, flatten_lug_boot, flatten_safety, flatten_class])

# Dense layers for further processing
dense1 = tf.keras.layers.Dense(4, activation='relu')(concatenated)
dense2 = tf.keras.layers.Dense(12, activation='sigmoid')(dense1)
dense3 = tf.keras.layers.Dense(4, activation='relu')(dense2)

# Output layer
output = tf.keras.layers.Dense(4, activation='softmax')(dense3)

# Define model with multiple inputs
model = tf.keras.Model(inputs=[maint_input, doors_input, lug_boot_input, safety_input, class_input], outputs=output)

# Compile the model
model.compile(optimizer='RMSprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model
model.fit({'maint': X_train['maint'],
           'doors': X_train['doors'],
           'lug_boot': X_train['lug_boot'],
           'safety': X_train['safety'],
           'class': X_train['class']},
          y_train,
          epochs=50,
          batch_size=8,
          validation_data=({'maint': X_test['maint'],
                            'doors': X_test['doors'],
                            'lug_boot': X_test['lug_boot'],
                            'safety': X_test['safety'],
                            'class': X_test['class']},
                           y_test))

# Evaluate the model
loss, accuracy = model.evaluate({'maint': X_test['maint'],
                                 'doors': X_test['doors'],
                                 'lug_boot': X_test['lug_boot'],
                                 'safety': X_test['safety'],
                                 'class': X_test['class']},
                                y_test)
print(f'Model Loss on Test Set: {loss}')
print(f'Model Accuracy on Test Set: {accuracy * 100:.2f}%')
