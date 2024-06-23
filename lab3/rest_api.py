from fastapi import FastAPI

from data import download_and_save_data
from test_model import get_model, get_predict
from entity import IrisRequest
from model import processing, check_result

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
   
app = FastAPI()
download_and_save_data() 
processing() 
check_result()
model = get_model()


@app.post("/predict")
def get_history(iris_request: IrisRequest):
    result = get_predict(model, iris_request) 
    return result
