import pandas as pd
import pickle

from entity import IrisRequest, IrisPredictEntity, IrisPredictResponse

# FIle path
model_path = './model.pkl'


# function for file loading
def get_model():
    return pickle.load(open(model_path, 'rb'))

# function for predicting
def get_predict(model, data: IrisRequest):
    response_list = list()
    for el in data.iris_entities:
        df = pd.DataFrame([[
            el['sepal_len'],
            el['sepal_wid'],
            el['petal_len'],
            el['petal_wid']]])
        prediction = model.predict(df)

        response_row = IrisPredictEntity(
            el['sepal_len'],
            el['sepal_wid'],
            el['petal_len'],
            el['petal_wid'],
            prediction[0]
        )
        response_list.append(response_row)
    return IrisPredictResponse(response_list)