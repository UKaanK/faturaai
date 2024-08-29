from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

app = Flask(__name__)

def preprocess_image(img):
    try:
        img = img.resize((224, 224))  # Modelin beklediği boyut
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalizasyon
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Resim işleme hatası: {e}")
        raise

# VGG16 modelini yükleyin
try:
    vgmodel = load_model('C:/Users/Casper EXCALIBUR/Desktop/faturaai/eArsiv_VGG16_Model.keras')
    print("VGG16 model başarıyla yüklendi.")
except Exception as e:
    print(f"VGG16 model yüklenirken hata oluştu: {e}")

# Siamese modelini yükleyin
try:
    siamese_model = load_model('C:/Users/Casper EXCALIBUR/Desktop/faturaai/Saha_VGG16_Model.keras', custom_objects={'l1_distance': lambda x: K.abs(x[0] - x[1])}, safe_mode=False)
    print("Siamese model başarıyla yüklendi.")
except Exception as e:
    print(f"Siamese model yüklenirken hata oluştu: {e}")

# Üçüncü VGG16 modelini yükleyin
try:
    vgmodel3 = load_model('C:/Users/Casper EXCALIBUR/Desktop/faturaai/eFatura_VGG16_Model.keras')
    print("Üçüncü VGG16 model başarıyla yüklendi.")
except Exception as e:
    print(f"Üçüncü VGG16 model yüklenirken hata oluştu: {e}")

def get_image_embedding(img):
    try:
        img = preprocess_image(img)
        embedding = siamese_model.predict(img)
        return embedding
    except Exception as e:
        print(f"Gömme vektörü alma hatası: {e}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dogru_fatura')
def dogru_fatura():
    return render_template('dogru_fatura.html')

@app.route('/qr_hatasi')
def qr_hatasi():
    return render_template('qr_hatasi.html')

@app.route('/bos_fatura')
def bos_fatura():
    return render_template('bos_fatura.html')

@app.route('/hatali_banka_isim')
def hatali_banka_isim():
    return render_template('hatali_banka_isim.html')

@app.route('/eski_logo')
def eski_logo():
    return render_template('eski_logo.html')

@app.route('/yanlis_fatura')
def yanlis_fatura():
    return render_template('yanlis_fatura.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file1' not in request.files:
        return jsonify({'error': 'A file is required'}), 400

    file1 = request.files['file1']

    try:
        user_img1 = Image.open(file1).convert('RGB')

        # İlk dosyanın gömme vektörünü alın
        embedding1 = get_image_embedding(user_img1)
        
        # İkinci dosyanın gömme vektörünü modelin içinden alın
        reference_img_path = 'C:/Users/Casper EXCALIBUR/Desktop/sunumproje/earsiv/birleşmiş1.jpg'
        reference_img = Image.open(reference_img_path).convert('RGB')
        embedding2 = get_image_embedding(reference_img)

        # Benzerliği hesaplayın
        similarity = np.linalg.norm(embedding1 - embedding2)

        # VGG16 modelini kullanarak tahmin yapın
        user_img1 = preprocess_image(user_img1)
        prediction = vgmodel.predict(user_img1)
        
        # Üçüncü VGG16 modelini kullanarak tahmin yapın
        prediction3 = vgmodel3.predict(user_img1)
        
        # Sınıf isimleri
        class_names = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7']

        # En yüksek olasılığa sahip sınıfı bulun
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_probability = np.max(prediction, axis=1)[0]
        
        # Üçüncü modelin tahminini kullanarak en yüksek olasılığa sahip sınıfı bulun
        predicted_class3 = np.argmax(prediction3, axis=1)[0]
        predicted_probability3 = np.max(prediction3, axis=1)[0]

        similarity_score = 1 - similarity

        # Ağırlıkları belirle
        alpha = 0.4  # Siamese modelinin ağırlığı
        beta = 0.3   # VGG16 modelinin ağırlığı
        gamma = 0.3  # Üçüncü modelin ağırlığı

        # Son skoru hesaplayın
        ensemble_score = alpha * similarity_score + beta * predicted_probability + gamma * predicted_probability3

        print(ensemble_score)
        print(predicted_probability)
        result = {
            'final_prediction': class_names[predicted_class],
            'ensemble_score': float(ensemble_score),
            'predicted_class': class_names[predicted_class],
            'predicted_probability': float(predicted_probability),
            'similarity': float(similarity)
        }

        if 0.315 <= ensemble_score <= 0.341:
            return jsonify({'redirect': '/hatali_banka_isim'})

        if 0.528 <= ensemble_score <= 0.5615:
            return jsonify({'redirect': '/hatali_banka_isim'})
        
        if  0.34 <= ensemble_score <=0.3663:
            return jsonify({'redirect': '/eski_logo'})

        if 0.5035 <= ensemble_score <= 0.51991:
            return jsonify({'redirect': '/qr_hatasi'})
        
        if 0.279 <= ensemble_score < 0.2908:
            return jsonify({'redirect': '/qr_hatasi'})
        
        if 0.458 <= ensemble_score < 0.47:
            return jsonify({'redirect': '/qr_hatasi'})
        
        if 0.2908 <=ensemble_score <=0.31202 :
            return jsonify({'redirect': '/dogru_fatura'})
        
        if 0.57  <=ensemble_score <=0.63 :
            return jsonify({'redirect': '/dogru_fatura'})
        
        if 0.978 <=ensemble_score <=0.984:
            return jsonify({'redirect': '/bos_fatura'})
        
        if 0.502<=ensemble_score < 0.5035:
            return jsonify({'redirect': '/bos_fatura'})
        
        if 0.81<=ensemble_score <= 0.86:
            return jsonify({'redirect': '/yanlis_fatura'})
        return jsonify(result)
    except Exception as e:
        print(f"Predict fonksiyonunda hata oluştu: {e}")
        return jsonify({'error': str(e)}), 500

    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
