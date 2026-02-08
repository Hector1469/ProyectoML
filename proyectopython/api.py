from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from typing import Annotated
import numpy as np
import joblib
import os

-----------------------------
1️⃣ Cargar modelo LOCAL (.pkl)
-----------------------------

MODEL_PATH = os.path.join(os.path.dirname(file), "gb_model.pkl")

try:
model = joblib.load(MODEL_PATH)
print("✅ Modelo cargado correctamente desde archivo local")
except Exception as e:
model = None
print(f"❌ Error cargando modelo local: {e}")

-----------------------------
2️⃣ Inicializar FastAPI
-----------------------------

app = FastAPI(
title="API de Predicción - Gradient Boosting",
description="API para predecir usando modelo Gradient Boosting",
version="1.0"
)

-----------------------------
3️⃣ Validación de entrada
-----------------------------

class PredictRequest(BaseModel):
features: Annotated[list[float], conlist(item_type=float, min_length=10, max_length=10)]

-----------------------------
4️⃣ Endpoint /health
-----------------------------

@app.get("/health")
def health_check():
if model:
try:
_ = model.predict(np.zeros((1, 10)))
return {"status": "ok"}
except Exception as e:
return {"status": "ko", "error": str(e)}
return {"status": "ko", "error": "Modelo no cargado"}

-----------------------------
5️⃣ Endpoint /predict
-----------------------------

@app.post("/predict")
def predict(request: PredictRequest):
if not model:
raise HTTPException(status_code=500, detail="Modelo no cargado")

try:
    x_input = np.array([request.features])
    prediction = model.predict(x_input)
    return {"prediction": float(prediction[0])}
except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))
