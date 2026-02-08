import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

# -----------------------------
# 1️⃣ Cargar modelo entrenado localmente (.pkl desde tu PC)
# -----------------------------
local_model_path = "gb_model.pkl"
model = joblib.load(local_model_path)

# -----------------------------
# 2️⃣ Registrar modelo en MLflow con Model Registry
# -----------------------------
mlflow.set_experiment("/Users/hectorgozade@gmail.com/gradient_boost_project")

with mlflow.start_run(run_name="GB_pretrained_model") as run:
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="gb_model",  # 🔑 Nombre para el Model Registry
        input_example=pd.DataFrame([[0]*10], columns=[f"f{i}" for i in range(10)])
    )
    mlflow.log_param("note", "Modelo entrenado localmente y registrado en MLflow")
    print(f"✅ Modelo registrado en MLflow, run_id: {run.info.run_id}")
