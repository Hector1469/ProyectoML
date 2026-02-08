# api.py# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from typing import Annotated
import numpy as np
import mlflow.sklearn
import os

# -----------------------------
# 1️⃣ Configuración del modelo usando Model Registry
# -----------------------------
# En lugar de RUN_ID, usamos un nombre de modelo y versión
MODEL_NAME = os.getenv("MODEL_NAME", "gb_model")  # nombre del modelo en MLflow
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")   # versión, puede ser "1" o "latest"

MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    print("✅ Modelo cargado desde MLflow (Model Registry)")
except Exception as e:
    model = None
    print(f"❌ Error cargando modelo: {e}")

# -----------------------------
# 2️⃣ Inicializar FastAPI
# -----------------------------
app = FastAPI(title="API de Predicción - Gradient Boosting")

# -----------------------------
# 3️⃣ Validación de entrada con Pydantic v2
# -----------------------------
class PredictRequest(BaseModel):
    features: Annotated[list[float], conlist(item_type=float, min_length=10, max_length=10)]

# -----------------------------
# 4️⃣ Endpoint /health
# -----------------------------
@app.get("/health")
def health_check():
    if model:
        try:
            _ = model.predict(np.zeros((1, 10)))
            return {"status": "ok"}
        except Exception as e:
            return {"status": "ko", "error": str(e)}
    else:
        return {"status": "ko", "error": "Modelo no cargado"}

# -----------------------------
# 5️⃣ Endpoint /predict
# -----------------------------
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


