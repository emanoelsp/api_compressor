from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Inicializa a API
app = FastAPI(title="API de Previsão de Vibração Alta")

# Carrega o modelo treinado
modelo = joblib.load('modelo_vibracao_final.pkl')

# Carrega o threshold
with open('threshold.txt', 'r') as f:
    threshold = float(f.read())

# Define o formato esperado dos dados (substitua pelos nomes reais das features)
class DadosEntrada(BaseModel):
    rpm: float
    pressao_bar: float
    corrente_A: float
    vazao_m3_min: float
    potencia_kw: float

@app.post("/prever/")
def prever_vibracao(dados: DadosEntrada):
    # Criar vetor de entrada para o modelo
    entrada = np.array([[dados.rpm, dados.pressao_bar, dados.corrente_A,
                          dados.vazao_m3_min, dados.potencia_kw]])

    # Obter probabilidade da classe 1
    proba = modelo.predict_proba(entrada)[:, 1][0]

    # Aplicar threshold ajustado
    previsao = int(proba >= threshold)

    return {
        "probabilidade_vibracao_alta": round(proba, 3),
        "previsao": previsao,  # 1 = vibração alta, 0 = normal
        "threshold_usado": threshold
    }
