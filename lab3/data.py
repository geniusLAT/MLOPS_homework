import pandas as pd
import os


csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
cur_path = "./iris.csv"
col_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
def download_and_save_data():
    iris = pd.read_csv(csv_url, names=col_names)
    iris.to_csv(cur_path, index=False)
    if os.path.exists(cur_path):
        print("Successful downloading")
    else:
        raise FileNotFoundError("Something wrong with IRIS dataset downloading")