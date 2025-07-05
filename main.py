from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="API de Previsão de Vibração Alta")

# CORS (caso vá consumir de outro domínio)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajuste se precisar restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rota para teste
@app.get("/")
def status():
    return {"mensagem": "API de previsão está online!"}

# Carregando modelo e threshold
modelo = joblib.load('modelo_vibracao_final.pkl')
with open('threshold.txt', 'r') as f:
    threshold = float(f.read())

class DadosEntrada(BaseModel):
    rpm: float
    pressao_bar: float
    corrente_A: float
    vazao_m3_min: float
    potencia_kw: float

@app.post("/prever/")
def prever_vibracao(dados: DadosEntrada):
    entrada = np.array([[dados.rpm, dados.pressao_bar, dados.corrente_A,
                          dados.vazao_m3_min, dados.potencia_kw]])

    proba = modelo.predict_proba(entrada)[:, 1][0]
    previsao = int(proba >= threshold)

    return {
        "probabilidade_vibracao_alta": round(proba, 3),
        "previsao": previsao,
        "threshold_usado": threshold
    }
