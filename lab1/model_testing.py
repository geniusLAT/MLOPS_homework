import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# LAODING data
test_data = pd.read_csv('data/test/temperature_test.csv')

# LOADING model
model = joblib.load('trained_model.pkl')

# preprocessing StandardScaler
scaler = StandardScaler()
scaled_temperature_test = scaler.fit_transform(test_data['Temperature'].values.reshape(-1, 1))

# forecasting
predictions = model.predict(scaled_temperature_test)

# output
print(predictions)