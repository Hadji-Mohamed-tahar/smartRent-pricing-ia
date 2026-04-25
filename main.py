from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List

# تحميل النموذج
try:
    artifacts = joblib.load('model_artifacts.joblib')
    model = artifacts['model']
    mlb = artifacts['mlb']
    feature_columns = artifacts['feature_columns']
except:
    exit("Model not found!")

app = FastAPI(title="Smart Rent AI V2")

class ApartmentFeatures(BaseModel):
    area: float
    rooms: int
    bathrooms: int
    wilaya: str
    municipality: str
    property_type: str # Apartment, Villa, Studio
    floor: int         # 0 للارضي، 1، 2...
    is_furnished: int  # 1 للمؤثثة، 0 للفارغة
    amenities: List[str]

@app.post("/predict")
def predict_rent(features: ApartmentFeatures):
    try:
        # 1. الميزات الرقمية
        input_data = {
            'area': [features.area],
            'rooms': [features.rooms],
            'bathrooms': [features.bathrooms],
            'floor': [features.floor],
            'is_furnished': [features.is_furnished]
        }
        
        # 2. معالجة One-Hot Encoding (الولاية + نوع السكن)
        # تصفير كل الأعمدة الفئوية أولاً
        cat_cols = [col for col in feature_columns if col.startswith(('wilaya_', 'municipality_', 'property_type_'))]
        for col in cat_cols:
            input_data[col] = [0]
        
        # تفعيل الاختيارات الحالية
        keys = [f"wilaya_{features.wilaya}", f"municipality_{features.municipality}", f"property_type_{features.property_type}"]
        for key in keys:
            if key in feature_columns:
                input_data[key] = [1]
            
        # 3. معالجة المرافق
        amenities_encoded = mlb.transform([features.amenities])
        amenities_df = pd.DataFrame(amenities_encoded, columns=mlb.classes_)
        
        X_input = pd.DataFrame(input_data)
        for col in mlb.classes_:
            X_input[col] = amenities_df[col].values
            
        X_input = X_input[feature_columns]
        prediction = model.predict(X_input)
        
        return {
            "predicted_monthly_rent": round(float(prediction[0]), 0),
            "currency": "DZD"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))