import pickle
import numpy as np
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, conlist

app = FastAPI(title="Predicting Wine Class")

# Represents a particular wine (or data-point)

class Wine(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


class WineBatch(BaseModel):
    """
    The Class represents a batch of Wines
    """
    batches: List[conlist(item_type=float, min_length=13, max_length=13)]

@app.on_event("startup")
def load_model(): 
    """
    Load classifier from pickle file
    """
    with open("/app/wine_dtc_model.pkl", "rb") as file:
        global model
        try:
            model = pickle.load(file)
        except ValueError as w:
            print("We got problem with Pickle. Use a compatible Pickle protocol")

@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://localhost:80/docs"
    
    
@app.post("/predict")
def predict(wine: Wine):
    data_point = np.array(
        [
            [
                wine.alcohol,
                wine.malic_acid,
                wine.ash, 
                wine.alcalinity_of_ash,
                wine.magnesium,
                wine.total_phenols,
                wine.flavanoids,
                wine.nonflavanoid_phenols,
                wine.proanthocyanins,
                wine.color_intensity,
                wine.hue,
                wine.od280_od315_of_diluted_wines,
                wine.proline,
            ]
        ]
    )

    pred = model.predict(data_point).tolist()
    pred = pred[0]
    print(pred)
    return {"Prediction": pred}

@app.post("/predict_batch")
def predict(wine: WineBatch):
    """
    predict wine types for a batch of wine data
    """
    batches = wine.batches
    np_batches = np.array(batches)
    pred = model.predict(np_batches).tolist()
    return {"Prediction": pred}
