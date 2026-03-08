from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import inference
import io

app = FastAPI(title="Weather Recognition Resnet API")

#learned this from gemini
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "ResNet18", "version": "1.0.0"}
# REST API endpoint
allowed_extensions = {"png", "jpg", "jpeg"}
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1].lower()
    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Supported types: {allowed_extensions}"
        )
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        probability, prediction = inference.predict(image)
        
        return {
            "filename": file.filename,
            "prediction": prediction,
            "probability": f"{probability*100:.2f}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")