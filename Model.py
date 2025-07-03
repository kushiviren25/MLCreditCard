from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi import HTTPException
import traceback


load_model = joblib.load("C:/Users/admin/Desktop/xgb_fraud_model.pkl")
load_scaler = joblib.load("C:/Users/admin/Desktop/fraud_scaler.pkl")
load_feature = joblib.load("C:/Users/admin/Desktop/feature_order.pkl")

print("Model :",load_model)
print("Scaler :",load_scaler)
print("Feature Order :",load_feature)


app = FastAPI(debug=True)

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.post("/predict_fraud")

def pred_fraud(data : Transaction):
    try:
        input_df = pd.DataFrame([data.model_dump()])
        input_df = input_df[load_feature]
        
        input_scale = load_scaler.transform(input_df)
        
        probability = load_model.predict_proba(input_scale)[0][1]
        predict = load_model.predict(input_scale)[0]

    
        if probability >0.95:
            risk = "High Risk"
            action = "Transaction Blocked"
        elif probability > 0.5 :
            risk = "Moderate Risk"
            action = "OTP verification"
        else:
            risk = "Low Risk"
            action = "Proceed Transaction"
            
        return {
            "Fraud Probability" : round(float(probability),4),
            "Risks " : risk,
            "Fraud Prediction" : int(predict),
            "Recommended Actions" : action
        }
    
    except Exception as e:
        print("ERROR:", traceback.format_exc())  # 👈 Shows error in terminal
        raise HTTPException(status_code=500, detail="Prediction failed")

    
# python -m uvicorn Model:app 
# cd C:\Users\admin\Desktop
