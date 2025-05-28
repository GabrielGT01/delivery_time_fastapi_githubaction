import os
import sys
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import Response, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from uvicorn import run as app_run
from pydantic import BaseModel
from typing import Optional
from src.logger import logging
from src.utils import load_object, DeliveryModel

# Pydantic model for live prediction input
class DeliveryPredictionInput(BaseModel):
    Distance_km: float
    Weather: str
    Traffic_Level: str
    Time_of_Day: str
    Vehicle_Type: str
    Preparation_Time_min: float
    Courier_Experience_yrs: float
    
    class Config:
        schema_extra = {
            "example": {
                "Distance_km": 7.93,
                "Weather": "Windy",
                "Traffic_Level": "Low",
                "Time_of_Day": "Afternoon",
                "Vehicle_Type": "Scooter",
                "Preparation_Time_min": 12,
                "Courier_Experience_yrs": 1.0
            }
        }

# FastAPI app initialization
app = FastAPI(title="Delivery Time Prediction API", description="Predict delivery times using ML models")

# Allow all origins for CORS (change for production)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jinja2 template directory for HTML rendering
templates = Jinja2Templates(directory="./templates")

# Mount static files (for CSS, JS)
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse, tags=["web interface"])
async def home(request: Request):
    """Main page with prediction form"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/batch", response_class=HTMLResponse, tags=["web interface"])
async def batch_upload_page(request: Request):
    """Batch prediction upload page"""
    return templates.TemplateResponse("batch.html", {"request": request})

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    """
    Expects a CSV file, runs prediction with the saved model,
    returns an HTML table with predictions.
    """
    try:
        # Read uploaded CSV as dataframe
        df = pd.read_csv(file.file)
        logging.info(f"Received dataframe for prediction: shape={df.shape}")
        
        # Load preprocessor and model
        preprocessor = load_object("final_model/preprocessor.pkl")
        model = load_object("final_model/model.pkl")
        delivery_model = DeliveryModel(preprocessor=preprocessor, model=model)
        
        # Make predictions
        y_pred = delivery_model.predict(df)
        df['predicted_delivery_time_min'] = y_pred
        
        # Save output CSV for record (optional)
        os.makedirs('prediction_output', exist_ok=True)
        df.to_csv('prediction_output/output.csv', index=False)
        
        # Render as HTML table
        table_html = df.to_html(classes='table table-striped table-hover', index=False)
        return templates.TemplateResponse("results.html", {
            "request": request, 
            "table": table_html,
            "total_predictions": len(df)
        })
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": str(e)
        })

@app.post("/predict/live")
async def live_predict_route(input_data: DeliveryPredictionInput):
    """
    API endpoint for live prediction - returns JSON
    """
    try:
        input_dict = input_data.dict()
        logging.info(f"Received live prediction request: {input_dict}")
        
        df = pd.DataFrame([input_dict])
        
        preprocessor = load_object("final_model/preprocessor.pkl")
        model = load_object("final_model/model.pkl")
        delivery_model = DeliveryModel(preprocessor=preprocessor, model=model)
        
        y_pred = delivery_model.predict(df)
        predicted_delivery_time = float(y_pred[0])
        
        result = {
            "input_data": input_dict,
            "predicted_delivery_time_min": round(predicted_delivery_time, 2),
            "status": "success"
        }
        
        logging.info(f"Live prediction result: {predicted_delivery_time:.2f} minutes")
        return JSONResponse(content=result)
        
    except Exception as e:
        logging.error(f"Live prediction error: {e}")
        error_response = {
            "error": str(e),
            "status": "error"
        }
        return JSONResponse(content=error_response, status_code=500)

@app.post("/predict/form")
async def form_predict_route(
    request: Request,
    distance_km: float = Form(...),
    weather: str = Form(...),
    traffic_level: str = Form(...),
    time_of_day: str = Form(...),
    vehicle_type: str = Form(...),
    preparation_time_min: float = Form(...),
    courier_experience_yrs: float = Form(...)
):
    """
    Web form submission endpoint - returns HTML page with result
    """
    try:
        # Create input dictionary
        input_dict = {
            "Distance_km": distance_km,
            "Weather": weather,
            "Traffic_Level": traffic_level,
            "Time_of_Day": time_of_day,
            "Vehicle_Type": vehicle_type,
            "Preparation_Time_min": preparation_time_min,
            "Courier_Experience_yrs": courier_experience_yrs
        }
        
        logging.info(f"Received form prediction request: {input_dict}")
        
        # Create DataFrame from single record
        df = pd.DataFrame([input_dict])
        
        # Load preprocessor and model
        preprocessor = load_object("final_model/preprocessor.pkl")
        model = load_object("final_model/model.pkl")
        delivery_model = DeliveryModel(preprocessor=preprocessor, model=model)
        
        # Make prediction
        y_pred = delivery_model.predict(df)
        predicted_delivery_time = float(y_pred[0])
        
        logging.info(f"Form prediction result: {predicted_delivery_time:.2f} minutes")
        
        return templates.TemplateResponse("prediction_result.html", {
            "request": request,
            "input_data": input_dict,
            "predicted_time": round(predicted_delivery_time, 2)
        })
        
    except Exception as e:
        logging.error(f"Form prediction error: {e}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": str(e)
        })

if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
