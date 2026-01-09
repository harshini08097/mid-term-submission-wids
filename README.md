import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset from Seaborn
df = sns.load_dataset('mpg')

# Quick look at the data
print(df.head())

# Cleaning: Remove rows with missing values
df = df.dropna()

# We want to predict 'mpg' using these features:
features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']
X = df[features]
y = df['mpg']
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# 1. Split the data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Scaling (Normalization)
scaler = StandardScaler()

# Fit the scaler on training data and transform both sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Data Pre-processed!")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset from Seaborn
df = sns.load_dataset('mpg')

# Quick look at the data
print(df.head())

# Cleaning: Remove rows with missing values
df = df.dropna()

# We want to predict 'mpg' using these features:
features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']
X = df[features]
y = df['mpg']

print("\n✅ Data Loaded and Cleaned!")
# TODO: Define the Sequential model
model = Sequential([
    # Layer 1: Try 64 neurons with 'relu' activation
    Dense(64, activation='relu', input_shape=(len(features),)),

    # Layer 2: Try 32 neurons with 'relu' activation
    Dense(32, activation='relu'),
    # Output Layer: 1 neuron

   Dense(1)
])

# TODO: Compile the model
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mae'])

print("✅ Model Constructed!")
model.summary()
# TODO: Train the model using .fit()
# Remember to use the SCALED features (X_train_scaled)
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    validation_split=0.2,
    verbose=1
)

print("✅ Training Complete!")
# TODO: Calculate Prediction, R2 Score and MAE

from sklearn.metrics import r2_score, mean_absolute_error
predictions = model.predict(X_test_scaled).flatten()
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)


print(f"Final R2 Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f} MPG")

# Visualization: Predicted vs Actual
plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title('How well did we do?')
plt.show()

print("\n✅ Data Loaded and Cleaned!")
