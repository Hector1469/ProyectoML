import gradio as gr
import requests

# 🔹 Endpoint correcto
API_URL = "https://proyectoml-1-yfso.onrender.com/predict"

def predict(*features):
    try:
        data = {"features": list(features)}
        response = requests.post(API_URL, json=data, timeout=10)

        if response.status_code == 200:
            result = response.json()["prediction"]
            return f"Predicción: {result}"
        else:
            return f"Error API: {response.text}"

    except Exception as e:
        return f"Error de conexión: {e}"

# 10 inputs como espera tu modelo
inputs = [gr.Number(label=f"Feature {i+1}") for i in range(10)]

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="text",
    title="Predicción Modelo Gradient Boosting",
    description="Introduce las 10 variables para obtener la predicción desde la API en Render"
)

# 🔹 share=True para generar link público
demo.launch(share=True)
