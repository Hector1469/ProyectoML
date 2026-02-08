import joblib
import mlflow
import mlflow.sklearn
import pandas as pd


# -----------------------------
# 1. Cargar modelo entrenado localmente (.pkl desde tu PC)
# -----------------------------
local_model_path = "gb_model.pkl"  # tu archivo en la misma carpeta que este script
model = joblib.load(local_model_path)

# -----------------------------
# 2. Subir modelo a DBFS
# -----------------------------
dbfs_path = "dbfs:/FileStore/models/gb_model.pkl"

# En Databricks Community, subimos usando dbutils
try:
    
    dbutils = DBUtils(spark)
    dbutils.fs.cp(f"file:/databricks/driver/{local_model_path}", dbfs_path)
    print(f" Modelo subido a DBFS: {dbfs_path}")
except Exception as e:
    print(f" No se pudo subir a DBFS: {e}. Esto es normal si se ejecuta fuera de Databricks.")

# -----------------------------
# 3. Registrar modelo en MLflow
# -----------------------------
# Crear un input_example de 10 features para MLflow
input_example = pd.DataFrame([[0]*10], columns=[f"f{i}" for i in range(10)])

mlflow.set_experiment("/Users/hectorgozade@gmail.com/gradient_boost_project")  
with mlflow.start_run(run_name="GB_pretrained_model") as run:
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
    mlflow.log_param("note", "Modelo entrenado localmente y registrado en MLflow")
    print(f"✅ Modelo registrado en MLflow, run_id: {run.info.run_id}")
