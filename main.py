
import os
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO

app = FastAPI(title="Smart Farming Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

@app.on_event("startup")
def load_model():
    global model

    MODEL_PATH = hf_hub_download(
        repo_id="divya-bharathi-2609/smart-farming-tomato-model",
        filename="tomato_disease_model.keras"
    )

    model = tf.keras.models.load_model(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "Smart Farming API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        return {"error": "Please upload an image"}

    image_bytes = await file.read()

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)

    classes = ["Early Blight", "Healthy"]

    index = int(np.argmax(predictions[0]))

    return {
        "disease": classes[index],
        "confidence": float(predictions[0][index])
    }
