from fastapi import FastAPI, File, UploadFile
from predictor import predict_image
from risk_scoring import get_risk_level
from PIL import Image
import io
#uvicorn main:app --reload
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    probability = predict_image(image)

    score, level, message = get_risk_level(probability)

    return {
        "probability": float(probability),
        "risk_score": score,
        "risk_level": level,
        "message": message
    }