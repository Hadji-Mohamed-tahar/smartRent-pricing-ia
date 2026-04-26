import pandas as pd
import numpy as np

# =============================================================================
# Smart Rent DZ - Data Generator v2.0
# التحسينات:
# 1. دعم نوعين من الكراء: شهري ويومي مع منطق سعري مستقل لكل نوع
# 2. تصحيح أسعار الفيلات لتعكس السوق الجزائري الفعلي
# 3. إضافة عوامل الموسمية والخدمات للكراء اليومي
# =============================================================================

wilayas = {
    'Algiers':    {'mult_monthly': 2.1,  'mult_daily': 2.8,  'mun': ["Sidi M'Hamed", 'Hydra', 'Zeralda', 'Bab Ezzouar', 'El Biar']},
    'Oran':       {'mult_monthly': 1.6,  'mult_daily': 2.0,  'mun': ['Akid Lotfi', 'Bir El Djir', 'Es Senia', 'Oran City']},
    'Setif':      {'mult_monthly': 1.2,  'mult_daily': 1.4,  'mun': ['Setif City', 'El Eulma', 'Ain Arnat']},
    "M'Sila":     {'mult_monthly': 0.85, 'mult_daily': 0.9,  'mun': ["M'Sila City", 'Boussaada', 'Magra']},
    'Constantine':{'mult_monthly': 1.35, 'mult_daily': 1.6,  'mun': ['Constantine City', 'Ali Mendjeli', 'El Khroub']}
}

prop_types    = ['Apartment', 'Villa', 'Studio']
amenities_list = ['Wifi', 'Air Conditioning', 'Elevator', 'Parking', 'Heating', 'Security 24/7', 'Pool', 'Garden']

# نسبة الكراء اليومي مقابل الشهري في السوق الجزائري
DAILY_RATIO = 0.25  # 25% من الإعلانات كراء يومي

data = []
np.random.seed(42)

for _ in range(2000):
    w_name = np.random.choice(list(wilayas.keys()))
    w_info = wilayas[w_name]

    # نوع الكراء
    rent_type = np.random.choice(['monthly', 'daily'], p=[1 - DAILY_RATIO, DAILY_RATIO])

    # توزيع واقعي لأنواع السكن
    p_type = np.random.choice(prop_types, p=[0.75, 0.10, 0.15])

    # أبعاد السكن حسب النوع
    if p_type == 'Villa':
        area  = np.random.randint(180, 500)
        rooms = np.random.randint(4, 9)
        floor = 0
    elif p_type == 'Studio':
        area  = np.random.randint(28, 55)
        rooms = 1
        floor = np.random.randint(0, 5)
    else:  # Apartment
        area  = np.random.randint(60, 165)
        rooms = np.random.randint(2, 6)
        floor = np.random.randint(0, 10)

    is_furnished   = np.random.choice([0, 1], p=[0.60, 0.40])
    has_elevator   = np.random.choice([0, 1])

    # -----------------------------------------------------------------
    # منطق تسعير الكراء الشهري (يعتمد على المساحة + الموقع)
    # -----------------------------------------------------------------
    if rent_type == 'monthly':
        base = (area * 320) + (rooms * 4_500)

        # تصحيح الفيلات: السوق الجزائري لا يُضاعف سعر الفيلا كثيراً
        # بسبب ضعف الطلب على الكراء طويل الأمد لها
        if p_type == 'Villa':
            base += 35_000           # زيادة ثابتة معقولة (وليس مضاعفة)
        elif p_type == 'Studio':
            base *= 0.85             # تخفيض طفيف لأن الاستوديو أصغر

        if is_furnished:
            base *= 1.50             # الأثاث يرفع السعر 50%

        # عقوبة الطابق العالي بدون مصعد
        if not has_elevator and floor > 3:
            base *= 0.87

        final_price = base * w_info['mult_monthly']
        final_price += np.random.normal(0, 2_500)
        final_price  = round(max(15_000, final_price) / 500) * 500

    # -----------------------------------------------------------------
    # منطق تسعير الكراء اليومي (يعتمد على الخدمات + الموسمية)
    # -----------------------------------------------------------------
    else:
        # السعر اليومي يبدأ من قاعدة مختلفة (تكلفة/ليلة)
        base = (area * 12) + (rooms * 300)

        if p_type == 'Villa':
            base += 4_500            # الفيلات اليومية أكثر جاذبية سياحياً
        elif p_type == 'Studio':
            base *= 0.80

        if is_furnished:
            base *= 1.65             # الأثاث أكثر أهمية في الكراء اليومي

        # عامل الموسمية (صيف/شتاء): نحاكيه بعشوائية
        season_factor = np.random.choice([1.0, 1.25, 1.5], p=[0.5, 0.3, 0.2])
        base *= season_factor

        # الخدمات الإضافية تزيد السعر اليومي أكثر
        n_premium = np.random.randint(0, 3)  # عدد الخدمات المميزة
        base += n_premium * 500

        final_price = base * w_info['mult_daily']
        final_price += np.random.normal(0, 300)
        final_price  = round(max(1_500, final_price) / 100) * 100

    # اختيار المرافق
    n_amenities = np.random.randint(1, 6)
    current_amenities = list(np.random.choice(amenities_list, n_amenities, replace=False))
    if has_elevator and p_type == 'Apartment':
        if 'Elevator' not in current_amenities:
            current_amenities.append('Elevator')

    data.append({
        'area':          area,
        'rooms':         rooms,
        'bathrooms':     np.random.randint(1, 4),
        'wilaya':        w_name,
        'municipality':  np.random.choice(w_info['mun']),
        'property_type': p_type,
        'floor':         floor,
        'is_furnished':  is_furnished,
        'has_elevator':  has_elevator,
        'rent_type':     rent_type,
        'amenities':     ','.join(list(set(current_amenities))),
        'price':         final_price
    })

df = pd.DataFrame(data)
df.to_csv('apartments_rent_data.csv', index=False)

print("✅ تم توليد 'apartments_rent_data.csv' بنجاح!")
print(f"   إجمالي السجلات : {len(df)}")
print(f"   كراء شهري      : {(df['rent_type']=='monthly').sum()}")
print(f"   كراء يومي      : {(df['rent_type']=='daily').sum()}")
print(f"\n📊 إحصائيات الأسعار الشهرية (دج):")
print(df[df['rent_type']=='monthly']['price'].describe().apply(lambda x: f"{x:,.0f}"))
print(f"\n📊 إحصائيات الأسعار اليومية (دج/ليلة):")
print(df[df['rent_type']=='daily']['price'].describe().apply(lambda x: f"{x:,.0f}"))