from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
import os
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from .utils import allowed_file, model_predict

origins = ["*"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR , 'updated_model.h5'))

#app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )

@app.post('/predict/')
async def predict(file: UploadFile = File(...)):

    if file and allowed_file(file.filename):
        try:
            class_result , prob_result = model_predict(file , model)

            predictions = {
                    "class1":class_result[0],
                    "class2":class_result[1],
                    "class3":class_result[2],
                    "class4":class_result[3],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                    "prob4": prob_result[3],
            }
            return JSONResponse(status_code=200,content={
                "filename": file.filename,
                "predictions": predictions,
                "message": "File uploaded successfully"
            })
        
        except Exception as e:
            return JSONResponse(status_code=500, content={"message": f"Prediction failed: {str(e)}"})

    else:
        return JSONResponse(
            status_code=400,
            content={"message": "Please upload images of jpg, jpeg, or png extension only"}
        )
