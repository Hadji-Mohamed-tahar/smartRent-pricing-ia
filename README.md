🏠 Smart Rent DZ - AI Pricing Engine (v2.0)هذا النظام هو المحرك الذكي لمشروع Smart Rent، يقوم بالتنبؤ بأسعار كراء العقارات في الجزائر بناءً على نماذج تعلم آلي (Machine Learning) مدربة على بيانات واقعية لسوق العقارات الجزائري لعام 2026.🚀 المميزاتنظام مزدوج: نموذجين منفصلين للكراء الشهري (Monthly) والكراء اليومي (Daily).دقة عالية: استخدام خوارزميات Gradient Boosting و Random Forest.هندسة ميزات ذكية: يأخذ في الاعتبار الطابق، وجود المصعد، التجهيز، ونوع السكن.نطاق الثقة: يوفر حد أدنى وأقصى للسعر لضمان واقعية التوقع.🛠️ التثبيت والتشغيل المحلي1. المتطلبات الأساسيةيجب أن يكون لديك Python 3.9 أو أحدث مثبتًا.2. تثبيت المكتباتقم بتشغيل الأمر التالي لتثبيت كافة التبعيات:Bashpip install -r requirements.txt
3. تدريب النموذجإذا قمت بتغيير البيانات في ملف الـ CSV أو أردت إعادة التدريب:Bashpython train.py
سيقوم هذا الأمر بإنشاء ملف model_artifacts.joblib الذي يحتوي على النماذج المدربة.4. تشغيل الـ APIلتشغيل السيرفر محلياً:Bashuvicorn main:app --reload
سيكون الرابط المحلي: http://127.0.0.1:8000🛰️ الاتصال بالـ API (API Documentation)Endpoint: POST /predictهذا هو المسار الأساسي للحصول على التوقعات.Request Body (JSON):يجب إرسال البيانات بالتنسيق التالي:الحقلالنوعالوصفareafloatالمساحة بالمتر المربع (يجب أن تكون > 20)roomsintعدد الغرفbathroomsintعدد الحماماتwilayastringالولاية (مثال: Algiers, Oran, M'Sila)municipalitystringالبلديةproperty_typestring(Apartment, Villa, Studio)rent_typestring(monthly, daily)floorintرقم الطابق (0 للأرضي)is_furnishedint1 للمؤثث، 0 للفارغhas_elevatorint1 للمصعد، 0 بدونهamenitiesarrayقائمة نصوص (مثال: ["Wifi", "Pool"])مثال للطلب (Request Example):JSON{
  "area": 120,
  "rooms": 3,
  "bathrooms": 1,
  "wilaya": "Algiers",
  "municipality": "Hydra",
  "property_type": "Apartment",
  "floor": 2,
  "is_furnished": 1,
  "has_elevator": 1,
  "rent_type": "monthly",
  "amenities": ["Wifi", "Air Conditioning", "Parking"]
}
📥 مخرجات الـ API (Response)عند نجاح الطلب، ستتلقى رداً بهذا الشكل:JSON{
    "predicted_price": 137600.0,
    "rent_type": "monthly",
    "currency": "DZD",
    "unit": "per month",
    "confidence_range": {
        "low": 123840,
        "high": 151360
    }
}
🌐 الربط مع Laravelلاستدعاء هذا النموذج من مشروع Laravel الخاص بك، استخدم الـ Http Client:PHPuse Illuminate\Support\Facades\Http;

$response = Http::post('https://your-app-name.onrender.com/predict', [
    'area' => 120,
    'rooms' => 3,
    'rent_type' => 'monthly',
    // باقي البيانات...
]);

if ($response->successful()) {
    $price = $response->json()['predicted_price'];
}