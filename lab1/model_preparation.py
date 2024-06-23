import pandas as pd
from sklearn.linear_model import LinearRegression

# Load processed
train_data = pd.read_csv('data/train/preprocessed_train_data.csv')

# Preparing
X_train = train_data['scaled_temperature'].values.reshape(-1, 1)
y_train = train_data['Temperature'].values

# Regression 
model = LinearRegression()
model.fit(X_train, y_train)

# Saving model
import joblib
joblib.dump(model, 'trained_model.pkl')