import pandas as pd
import numpy as np

# إعداد المعايير الجزائرية لضمان الدقة
wilayas = {
    'Algiers': {'mult': 2.1, 'mun': ['Sidi M\'Hamed', 'Hydra', 'Zeralda', 'Bab Ezzouar', 'El Biar']},
    'Oran': {'mult': 1.6, 'mun': ['Akid Lotfi', 'Bir El Djir', 'Es Senia', 'Oran City']},
    'Setif': {'mult': 1.2, 'mun': ['Setif City', 'El Eulma', 'Ain Arnat']},
    'M\'Sila': {'mult': 0.85, 'mun': ['M\'Sila City', 'Boussaada', 'Magra']},
    'Constantine': {'mult': 1.35, 'mun': ['Constantine City', 'Ali Mendjeli', 'El Khroub']}
}

prop_types = ['Apartment', 'Villa', 'Studio']
amenities_list = ['Wifi', 'Air Conditioning', 'Elevator', 'Parking', 'Heating', 'Security 24/7']

data = []
np.random.seed(42)

for _ in range(2000):
    w_name = np.random.choice(list(wilayas.keys()))
    w_info = wilayas[w_name]
    
    # توزيع واقعي لأنواع السكن
    p_type = np.random.choice(prop_types, p=[0.75, 0.1, 0.15]) 
    
    # ضبط المساحة حسب نوع السكن
    if p_type == 'Villa':
        area = np.random.randint(150, 450)
        rooms = np.random.randint(4, 9)
        floor = 0
    elif p_type == 'Studio':
        area = np.random.randint(30, 55)
        rooms = 1
        floor = np.random.randint(0, 5)
    else: # Apartment
        area = np.random.randint(60, 160)
        rooms = np.random.randint(2, 5)
        floor = np.random.randint(0, 9)
    
    is_furnished = np.random.choice([0, 1], p=[0.65, 0.35])
    
    # حساب السعر بمنطق السوق الجزائري (دج/شهر)
    base_price = (area * 320) + (rooms * 4500)
    
    if p_type == 'Villa': base_price += 45000
    if is_furnished: base_price *= 1.55 # زيادة معتبرة للأثاث
    
    # تأثير الطابق بدون مصعد
    has_elevator = np.random.choice([0, 1])
    if not has_elevator and floor > 3:
        base_price *= 0.88 
    
    # تطبيق معامل الولاية
    final_price = base_price * w_info['mult']
    
    # لمسة عشوائية للسوق
    final_price += np.random.normal(0, 2500)
    
    # تقريب السعر لأقرب 500 دج
    final_price = round(max(15000, final_price) / 500) * 500

    # اختيار المرافق
    n_amenities = np.random.randint(1, 5)
    current_amenities = list(np.random.choice(amenities_list, n_amenities, replace=False))
    if has_elevator and p_type == 'Apartment':
        current_amenities.append('Elevator')
    
    data.append({
        'area': area,
        'rooms': rooms,
        'bathrooms': np.random.randint(1, 3),
        'wilaya': w_name,
        'municipality': np.random.choice(w_info['mun']),
        'property_type': p_type,
        'floor': floor,
        'is_furnished': is_furnished,
        'amenities': ",".join(list(set(current_amenities))),
        'price': final_price
    })

df = pd.DataFrame(data)
df.to_csv('apartments_rent_data.csv', index=False)
print("Done! 'apartments_rent_data.csv' has been generated with 2000 records.")