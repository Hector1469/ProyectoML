# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from typing import Annotated
import numpy as np
import mlflow.sklearn

# -----------------------------
# 1️⃣ Configuración del modelo
# -----------------------------
RUN_ID = "f71f6ae8feb341ad972efaa5af0ffc4c"  # reemplaza con tu run_id
MODEL_URI = f"runs:/{RUN_ID}/model"

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    print(" Modelo cargado desde MLflow")
except Exception as e:
    model = None
    print(f" Error cargando modelo: {e}")

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
            # prueba rápida: 1 fila con 10 ceros
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
