from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from twilio.rest import Client
import os

app = FastAPI()

# Configura las credenciales de Twilio
ACCOUNT_SID = "AC44953c9da5e171c3410431a55186ee6b"
AUTH_TOKEN = "9efdd1de38ce681196c9f1f9ab8dd41e"
FROM_NUMBER = "+17627016116"
TO_NUMBER = "+59160501893"  # Tu número boliviano

# Inicializa Twilio
twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)

# Cargar modelo y tokenizador
modelo_path = "modelo_distilbert_clasificacion_relatosnuevos_3"
tokenizer = AutoTokenizer.from_pretrained(modelo_path)
model = AutoModelForSequenceClassification.from_pretrained(modelo_path)

# Modelo de entrada
class TextoEntrada(BaseModel):
    texto: str

@app.get("/")
def home():
    return {"mensaje": "API de clasificación de texto con DistilBERT"}

@app.post("/clasificar")
async def clasificar(entrada: TextoEntrada):
    texto = entrada.texto
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        if(pred == 1):
            predic = "es acoso"
            # Enviar SMS con Twilio
            #try:
            #    message = twilio_client.messages.create(
            #        body=f"Alerta: se detectó posible indicio de acoso escolar.\nTexto: {texto}",
            #        from_=FROM_NUMBER,
            #        to=TO_NUMBER
            #    )
            #    print("Mensaje enviado:", message.sid)
            #except Exception as e:
            #    print("Error al enviar SMS:", e)
        else: 
            predic = "no es acoso"

    return {"texto": texto, "prediccion": predic}
