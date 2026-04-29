from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import pandas as pd
import joblib
from typing import List, Literal
from fastapi.middleware.cors import CORSMiddleware

# =============================================================================
# Smart Rent DZ - API v2.0
# التحسينات:
# - دعم rent_type: monthly | daily
# - توجيه الطلب للنموذج المناسب تلقائياً
# - التحقق من صحة المدخلات عبر Pydantic validators
# - إرجاع فاصل ثقة (confidence interval) تقريبي
# =============================================================================

try:
    artifacts = joblib.load('model_artifacts.joblib')
    model_monthly           = artifacts['model_monthly']
    model_daily             = artifacts['model_daily']
    mlb_monthly             = artifacts['mlb_monthly']
    mlb_daily               = artifacts['mlb_daily']
    feature_columns_monthly = artifacts['feature_columns_monthly']
    feature_columns_daily   = artifacts['feature_columns_daily']
    print("✅ تم تحميل النماذج بنجاح.")
except Exception as e:
    exit(f"❌ خطأ في تحميل النموذج: {e}")

app = FastAPI(
    title="Smart Rent DZ - AI Pricing API",
    description="تنبؤ بأسعار كراء العقارات في الجزائر (شهري ويومي)",
    version="2.0.0"
)

# 👇 هنا مباشرة (مهم جدًا)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # لاحقًا ضع دومين الفرونت
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ApartmentFeatures(BaseModel):
    area:          float = Field(..., gt=20,  description="المساحة بالمتر المربع")
    rooms:         int   = Field(..., ge=1,   description="عدد الغرف")
    bathrooms:     int   = Field(..., ge=1,   description="عدد الحمامات")
    wilaya:        str   = Field(...,          description="الولاية: Algiers | Oran | Setif | M'Sila | Constantine")
    municipality:  str   = Field(...,          description="البلدية")
    property_type: str   = Field(...,          description="نوع السكن: Apartment | Villa | Studio")
    floor:         int   = Field(0,   ge=0,   description="الطابق (0 للأرضي)")
    is_furnished:  int   = Field(...,          description="مؤثث؟ 1=نعم، 0=لا")
    has_elevator:  int   = Field(0,            description="مصعد؟ 1=نعم، 0=لا")
    rent_type:     Literal['monthly', 'daily'] = Field(..., description="نوع الكراء: monthly أو daily")
    amenities:     List[str] = Field(default=[], description="المرافق: Wifi, Pool, Parking...")

    @validator('property_type')
    def validate_property_type(cls, v):
        allowed = ['Apartment', 'Villa', 'Studio']
        if v not in allowed:
            raise ValueError(f"property_type يجب أن يكون أحد: {allowed}")
        return v

    @validator('is_furnished', 'has_elevator')
    def validate_binary(cls, v):
        if v not in [0, 1]:
            raise ValueError("يجب أن تكون القيمة 0 أو 1")
        return v


def prepare_input(features: ApartmentFeatures, model_type: str) -> pd.DataFrame:
    """
    تحويل بيانات الطلب إلى DataFrame مطابق لهيكل التدريب.
    يُطبّق نفس هندسة الميزات المستخدمة في train.py.
    """
    mlb             = mlb_monthly    if model_type == 'monthly' else mlb_daily
    feature_columns = feature_columns_monthly if model_type == 'monthly' else feature_columns_daily

    # --- ميزات رقمية أساسية ---
    input_data = {
        'area':          [features.area],
        'rooms':         [features.rooms],
        'bathrooms':     [features.bathrooms],
        'floor':         [features.floor],
        'is_furnished':  [features.is_furnished],
        'has_elevator':  [features.has_elevator],

        # --- ميزات مهندَسة (يجب أن تتطابق مع train.py) ---
        'floor_penalty': [int(features.floor > 3 and features.has_elevator == 0)],
        'room_density':  [features.rooms / max(features.area, 1)],
        'total_capacity':[features.rooms + features.bathrooms],
        'is_villa':      [int(features.property_type == 'Villa')],
    }

    # --- تصفير كل الأعمدة الفئوية ---
    cat_cols = [c for c in feature_columns if c.startswith(('wilaya_', 'municipality_', 'property_type_'))]
    for col in cat_cols:
        input_data[col] = [0]

    # --- تفعيل القيم الحالية ---
    for key in [f"wilaya_{features.wilaya}",
                f"municipality_{features.municipality}",
                f"property_type_{features.property_type}"]:
        if key in feature_columns:
            input_data[key] = [1]

    # --- ترميز المرافق ---
    amenities_enc = mlb.transform([features.amenities])
    amenities_df  = pd.DataFrame(amenities_enc, columns=mlb.classes_)

    X = pd.DataFrame(input_data)
    for col in mlb.classes_:
        X[col] = amenities_df[col].values

    # إعادة الترتيب ليطابق ترتيب التدريب تماماً
    missing_cols = set(feature_columns) - set(X.columns)
    for col in missing_cols:
        X[col] = 0

    return X[feature_columns]


@app.get("/", summary="صفحة الترحيب")
def root():
    return {
        "message": "مرحباً بك في Smart Rent DZ API v2.0",
        "endpoints": {
            "predict": "POST /predict",
            "health":  "GET /health"
        }
    }


@app.get("/health", summary="فحص حالة النماذج")
def health():
    return {
        "status":         "healthy",
        "models_loaded":  True,
        "monthly_features": len(feature_columns_monthly),
        "daily_features":   len(feature_columns_daily),
    }


@app.post("/predict", summary="التنبؤ بسعر الكراء")
def predict_rent(features: ApartmentFeatures):
    """
    يُعيد:
    - predicted_price: السعر المتوقع
    - rent_type: نوع الكراء
    - currency: العملة (دج)
    - unit: وحدة السعر (شهر أو ليلة)
    - confidence_range: نطاق ثقة تقريبي (±10% للشهري، ±15% لليومي)
    """
    try:
        X = prepare_input(features, features.rent_type)

        if features.rent_type == 'monthly':
            model = model_monthly
            margin = 0.10  # هامش خطأ ±10%
            unit   = "per month"
        else:
            model = model_daily
            margin = 0.15  # هامش خطأ ±15% (البيانات اليومية أقل)
            unit   = "per night"

        predicted = float(model.predict(X)[0])
        predicted = round(max(0, predicted) / 100) * 100  # تقريب لأقرب 100 دج

        return {
            "predicted_price":  predicted,
            "rent_type":        features.rent_type,
            "currency":         "DZD",
            "unit":             unit,
            "confidence_range": {
                "low":  round(predicted * (1 - margin)),
                "high": round(predicted * (1 + margin)),
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في التنبؤ: {str(e)}")