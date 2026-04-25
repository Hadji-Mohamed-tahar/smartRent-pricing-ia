import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import os

def train_model():
    # 1. قراءة البيانات من ملف CSV
    file_name = 'apartments_rent_data.csv'
    
    if not os.path.exists(file_name):
        print(f"خطأ: الملف {file_name} غير موجود!")
        return

    df = pd.read_csv(file_name)

    # 2. معالجة عمود المرافق (تحويل النص إلى قائمة)
    df['amenities'] = df['amenities'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

    # 3. تحويل المرافق باستخدام MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    amenities_encoded = mlb.fit_transform(df['amenities'])
    amenities_df = pd.DataFrame(amenities_encoded, columns=mlb.classes_)

    # 4. تحويل الميزات الفئوية (Categorical) باستخدام One-Hot Encoding
    # تشمل الآن: الولاية، البلدية، ونوع السكن
    df_encoded = pd.get_dummies(df[['wilaya', 'municipality', 'property_type']], drop_first=False)

    # 5. دمج كافة الميزات
    # أضفنا هنا: الطابق (floor) والتجهيز (is_furnished) كميزات رقمية/منطقية
    X = pd.concat([
        df[['area', 'rooms', 'bathrooms', 'floor', 'is_furnished']], 
        df_encoded, 
        amenities_df
    ], axis=1)
    
    y = df['price']

    print(f"--- جاري التدريب على {len(df)} سجل مع الميزات الجديدة ---")
    
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    # 7. حفظ النموذج والأدوات المساعدة
    artifacts = {
        'model': model,
        'mlb': mlb,
        'feature_columns': X.columns.tolist()
    }

    joblib.dump(artifacts, 'model_artifacts.joblib')
    print("--- تم التحديث! النموذج الآن يدعم نوع السكن، الطابق، والتجهيز ---")

if __name__ == "__main__":
    train_model()