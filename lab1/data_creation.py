import os
import numpy as np
import pandas as pd

data_dir = 'data'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

for directory in [train_dir, test_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

np.random.seed(10)

data_train = pd.date_range(start='1/1/2022', periods=100)
temperature_train = np.sin(np.arange(0, 10, 0.1)) * 20 + np.random.normal(0, 2, 100)
temperature_train[10:15] += 10  # Добавляем аномалии
df_train = pd.DataFrame({'Date': data_train, 'Temperature': temperature_train})
df_train.to_csv(os.path.join(train_dir, 'temperature_train.csv'), index=False)

data_test = pd.date_range(start='1/1/2022', periods=50)
temperature_test = np.sin(np.arange(0, 5, 0.1)) * 20 + np.random.normal(0, 2, 50)
temperature_test[10:15] += 10  # Добавляем аномалии
df_test = pd.DataFrame({'Date': data_test, 'Temperature': temperature_test})
df_test.to_csv(os.path.join(test_dir, 'temperature_test.csv'), index=False)

print("Данные успешно созданы и сохранены в папке 'data'.")
