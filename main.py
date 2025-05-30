from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional
import os
import shutil
import json
from uuid import uuid4

from crop import crop_bottom_middle
from sc import process_image_noise

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",  
    "*",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("qrs", exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "FastAPI Server Running"}

@app.post("/noiseqr/noise/")
async def receive_image(
    input_image: UploadFile = File(...),
    text_input: str = Form(...),
    print_type: str = Form(...),
    fingerprint_id: str = Form(...),
    location: Optional[str] = Form(None),
    blur_values_history: Optional[str] = Form(None),
):
    
    try:
        # Save the input image
        filename = f"{uuid4().hex}_{input_image.filename}"
        image_path = os.path.join("qrs", filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(input_image.file, buffer)
        print("Hello0")
        # Optionally, read other fields
        #qr_text = await text_input.read()  # if you want to process it as bytes/text
        loc_data = json.loads(location) if location else None
        blur_history = json.loads(blur_values_history) if blur_values_history else None
        print("Hello1")
        # Process image here later
        # results = your_processing_function(image_path, qr_text, ...)
        cropped_path = crop_bottom_middle(image_path)
        print(cropped_path)
        ssim = process_image_noise(cropped_path, "qrs/RP30.png", "qrs") 
        result = 0
        if ssim > 0.6:
            result = 1
            
        return {
            "message": "Image processed",
            "ensemble_score": result,            
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True) 
