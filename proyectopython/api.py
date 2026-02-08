# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from typing import Annotated
import numpy as np
import mlflow.sklearn
import os

# -----------------------------
# 1️⃣ Configuración del modelo usando MLflow + Databricks
# -----------------------------
# Variables de entorno (configúralas en Render)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "gb_model")  # nombre del modelo en MLflow
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")   # versión del modelo, "1" o "latest"

if MLFLOW_TRACKING_URI:
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
if DATABRICKS_TOKEN:
    os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

# -----------------------------
# 2️⃣ Intentar cargar el modelo
# -----------------------------
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    print("✅ Modelo cargado desde Databricks MLflow (Model Registry)")
except Exception as e:
    model = None
    print(f"❌ Error cargando modelo: {e}")

# -----------------------------
# 3️⃣ Inicializar FastAPI
# -----------------------------
app = FastAPI(
    title="API de Predicción - Gradient Boosting",
    description="API para predecir usando modelo Gradient Boosting registrado en Databricks MLflow",
    version="1.0"
)

# -----------------------------
# 4️⃣ Validación de entrada con Pydantic
# -----------------------------
class PredictRequest(BaseModel):
    features: Annotated[list[float], conlist(item_type=float, min_length=10, max_length=10)]

# -----------------------------
# 5️⃣ Endpoint /health
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
# 6️⃣ Endpoint /predict
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
