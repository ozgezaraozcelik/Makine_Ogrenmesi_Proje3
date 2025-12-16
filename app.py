from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Modeli yüklüyoruz
model = joblib.load('elmas_modeli.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Kullanıcıdan sayısal verileri al
            carat = float(request.form['carat'])
            depth = float(request.form['depth'])
            table = float(request.form['table'])
            x = float(request.form['x'])
            
            # 2. Kategorik seçimleri al
            cut = request.form['cut']
            color = request.form['color']
            clarity = request.form['clarity']

            # --- ONE-HOT ENCODING (Modelin Dili) ---
            # Cut (Good, Ideal, Premium, Very Good)
            cut_good = 1.0 if cut == 'Good' else 0.0
            cut_ideal = 1.0 if cut == 'Ideal' else 0.0
            cut_premium = 1.0 if cut == 'Premium' else 0.0
            cut_very_good = 1.0 if cut == 'Very Good' else 0.0

            # Color (E, F, G, H, I, J)
            color_e = 1.0 if color == 'E' else 0.0
            color_f = 1.0 if color == 'F' else 0.0
            color_g = 1.0 if color == 'G' else 0.0
            color_h = 1.0 if color == 'H' else 0.0
            color_i = 1.0 if color == 'I' else 0.0
            color_j = 1.0 if color == 'J' else 0.0

            # Clarity (IF, SI1, SI2, VS1, VS2, VVS1, VVS2, VVS1, VVS2)
            clarity_if = 1.0 if clarity == 'IF' else 0.0
            clarity_si1 = 1.0 if clarity == 'SI1' else 0.0
            clarity_si2 = 1.0 if clarity == 'SI2' else 0.0
            clarity_vs1 = 1.0 if clarity == 'VS1' else 0.0
            clarity_vs2 = 1.0 if clarity == 'VS2' else 0.0
            clarity_vvs1 = 1.0 if clarity == 'VVS1' else 0.0
            clarity_vvs2 = 1.0 if clarity == 'VVS2' else 0.0

            # Özellikleri birleştir
            features = [carat, depth, table, x, 
                        cut_good, cut_ideal, cut_premium, cut_very_good,
                        color_e, color_f, color_g, color_h, color_i, color_j,
                        clarity_if, clarity_si1, clarity_si2, clarity_vs1, clarity_vs2, clarity_vvs1, clarity_vvs2]
            
            final_features = np.array([features])
            prediction = model.predict(final_features)[0]

            if prediction < 0: prediction = 0
            
            return render_template('index.html', prediction_text=f'💎 Tahmini Fiyat: ${int(prediction)}')
        
        except Exception as e:
            return render_template('index.html', prediction_text=f'Hata: {e}')

if __name__ == '__main__':
    app.run(debug=True)