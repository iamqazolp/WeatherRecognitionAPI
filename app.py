from fastapi import FastAPI, File, UploadFile
from PIL import Image
import inference
import io
app = FastAPI(title="Weather Recognition Resnet API")

# REST API endpoint
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    probability, prediction=inference.predict(image)
        
    return {"filename": file.filename, "prediction": prediction, "probablity": f"{probability*100:.2f}"}