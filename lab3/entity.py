from pydantic import BaseModel
class IrisRequest(BaseModel):
    iris_entities: list

class IrisPredictEntity:
    def __init__(self,
                 sepal_len: float,
                 sepal_wid: float,
                 petal_len: float,
                 petal_wid: float,
                 iris_type: str):
        self.sepal_len = sepal_len  
        self.sepal_wid = sepal_wid 
        self.petal_len = petal_len  
        self.petal_wid = petal_wid 
        self.iris_type = iris_type 


class IrisPredictResponse:
    def __init__(self, predictions: list):
        self.predictions = predictions