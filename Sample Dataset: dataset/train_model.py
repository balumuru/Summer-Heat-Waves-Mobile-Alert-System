import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Load the dataset
df = pd.read_csv('dataset/heatwave_data.csv')

# Features and label
X = df[['temperature', 'humidity', 'wind_speed']]
y = df['heatwave']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/heatwave_model.pkl')

print("Model trained and saved successfully.")
