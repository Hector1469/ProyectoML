import gradio as gr
import requests

API_URL = "https://proyectoml-1-yfso.onrender.com/predict"

def predict(*features):
    try:
        data = {"features": list(features)}
        r = requests.post(API_URL, json=data, timeout=10)
        if r.status_code == 200:
            return f"Predicción: {r.json()['prediction']}"
        return f"Error API: {r.text}"
    except Exception as e:
        return f"Error conexión API: {e}"

inputs = [gr.Number(label=f"Feature {i+1}") for i in range(10)]

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="text",
    title="Predicción Salarial Municipal",
    description="Modelo Gradient Boosting desplegado en la nube"
)

demo.launch(server_name="0.0.0.0", server_port=10000)


