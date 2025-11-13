from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import datetime
import json
import pandas as pd

app = Flask(__name__, template_folder="templates", static_folder="static")

# Update path sesuai dengan model hasil training terakhir
MODEL_PATH = "model/cardiovascular_risk_model_optimal_hybrid.joblib"

# Load model dan metadata
use_model = False
model_data = None
label_encoder = None
feature_names = None
pipeline = None

if os.path.exists(MODEL_PATH):
    try:
        model_data = joblib.load(MODEL_PATH)
        pipeline = model_data["pipeline"]
        label_encoder = model_data["label_encoder"]
        feature_names = model_data["features"]
        strategy_name = model_data.get("strategy", "optimal_hybrid")
        
        use_model = True
        print("✅ Model loaded successfully!")
        print(f"   Strategy: {strategy_name}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Classes: {list(label_encoder.classes_)}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        pipeline = None
else:
    pipeline = None
    print(f"❌ Model not found at {MODEL_PATH} - please run training first")

class CardiovascularPredictor:
    def __init__(self, model, label_encoder, feature_names):
        self.model = model
        self.label_encoder = label_encoder
        self.feature_names = feature_names
        
        self.risk_descriptions = {
            "NORMAL": "Parameter vital dalam batas normal",
            "BORDERLINE_HYPERTENSION": "Tekanan darah borderline, perlu observasi",
            "HYPERTENSION_STAGE1": "Hipertensi stage 1, konsultasi dokter disarankan",
            "HYPERTENSION_STAGE2": "Hipertensi stage 2, evaluasi segera diperlukan",
            "TACHYCARDIA": "Denyut jantung meningkat, pemantauan diperlukan",
            "BRADYCARDIA": "Denyut jantung rendah, pemeriksaan lebih lanjut diperlukan",
        }
        
        self.risk_levels = {
            "NORMAL": "RENDAH",
            "BORDERLINE_HYPERTENSION": "SEDANG",
            "HYPERTENSION_STAGE1": "SEDANG",
            "HYPERTENSION_STAGE2": "TINGGI",
            "TACHYCARDIA": "TINGGI",
            "BRADYCARDIA": "TINGGI",
        }

    def calculate_bmi(self, weight, height):
        """Menghitung BMI dari berat badan (kg) dan tinggi badan (m)"""
        if height <= 0:
            return 0
        return weight / (height ** 2)

    def get_bmi_category(self, bmi):
        """Kategori BMI"""
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"

    def calculate_risk_score(self, bmi_category, age, smoking_status, active_lifestyle):
        """Menghitung skor risiko tambahan berdasarkan faktor lifestyle"""
        risk_score = 0
        
        # BMI factor
        bmi_factors = {"Underweight": 0.1, "Normal": 0, "Overweight": 0.3, "Obese": 0.5}
        risk_score += bmi_factors.get(bmi_category, 0)
        
        # Age factor
        if age >= 60:
            risk_score += 0.4
        elif age >= 45:
            risk_score += 0.2
        elif age >= 30:
            risk_score += 0.1
            
        # Smoking factor
        if smoking_status == "current":
            risk_score += 0.4
        elif smoking_status == "former":
            risk_score += 0.2
            
        # Activity factor (negative - mengurangi risiko)
        if active_lifestyle:
            risk_score -= 0.2
            
        return max(0, min(1, risk_score))

    def predict_comprehensive_risk(self, systole, diastole, heart_rate, temperature, 
                                 weight=None, height=None, age=None, 
                                 smoking_status=None, active_lifestyle=None):
        """
        Prediksi risiko komprehensif dengan input tambahan dan output distribusi detail
        """
        try:
            # Validasi input dasar
            inputs = [systole, diastole, heart_rate, temperature]
            if not all(isinstance(x, (int, float, np.number)) for x in inputs):
                raise ValueError("Semua parameter vital harus numerik")

            if systole <= diastole:
                raise ValueError("Sistolik harus lebih besar dari diastolik")

            if not (60 <= systole <= 250):
                raise ValueError("Sistolik di luar rentang realistis (60-250)")
            if not (30 <= diastole <= 150):
                raise ValueError("Diastolik di luar rentang realistis (30-150)")
            if not (30 <= heart_rate <= 180):
                raise ValueError("Denyut jantung di luar rentang realistis (30-180)")
            if not (35.0 <= temperature <= 40.0):
                raise ValueError("Suhu di luar rentang realistis (35.0-40.0)")

            # Feature engineering dasar - SAMA DENGAN TRAINING
            pulse_pressure = systole - diastole
            mean_arterial_pressure = diastole + (pulse_pressure / 3.0)
            pressure_heart_ratio = systole / max(heart_rate, 1)
            pulse_pressure_ratio = pulse_pressure / mean_arterial_pressure
            bp_variability = abs(systole - 120) + abs(diastole - 80)
            
            hypertension_risk_smooth = np.tanh((systole - 120) / 40 + (diastole - 80) / 25)
            heart_rate_risk_smooth = np.tanh(abs(heart_rate - 70) / 30)
            
            double_product = (systole * heart_rate) / 100
            bp_ratio = systole / diastole
            map_heart_ratio = mean_arterial_pressure / heart_rate

            # Kategori - SAMA DENGAN TRAINING
            systolic_category = self._get_systolic_category(systole)
            diastolic_category = self._get_diastolic_category(diastole)
            heart_rate_category = self._get_heart_rate_category(heart_rate)

            # Menyiapkan features dasar
            base_features = [
                systole, diastole, heart_rate, temperature,
                pulse_pressure, mean_arterial_pressure, pressure_heart_ratio,
                pulse_pressure_ratio, bp_variability, hypertension_risk_smooth, 
                heart_rate_risk_smooth, double_product, bp_ratio, map_heart_ratio,
                systolic_category, diastolic_category, heart_rate_category
            ]

            # Menambahkan features lifestyle jika tersedia
            lifestyle_features = []
            lifestyle_info = {}
            
            if weight is not None and height is not None:
                bmi = self.calculate_bmi(weight, height)
                bmi_category = self.get_bmi_category(bmi)
                lifestyle_features.extend([bmi, 
                                         1 if bmi_category == "Underweight" else 0,
                                         1 if bmi_category == "Normal" else 0,
                                         1 if bmi_category == "Overweight" else 0,
                                         1 if bmi_category == "Obese" else 0])
                lifestyle_info['bmi'] = bmi
                lifestyle_info['bmi_category'] = bmi_category
            else:
                # Default values jika tidak ada input
                lifestyle_features.extend([0, 0, 0, 0, 0])
                lifestyle_info['bmi'] = None
                lifestyle_info['bmi_category'] = "Unknown"

            if age is not None:
                lifestyle_features.extend([age,
                                         1 if age < 30 else 0,
                                         1 if 30 <= age < 45 else 0,
                                         1 if 45 <= age < 60 else 0,
                                         1 if age >= 60 else 0])
                lifestyle_info['age'] = age
                lifestyle_info['age_group'] = self._get_age_group(age)
            else:
                lifestyle_features.extend([0, 0, 0, 0, 0])
                lifestyle_info['age'] = None
                lifestyle_info['age_group'] = "Unknown"

            if smoking_status is not None:
                lifestyle_features.extend([1 if smoking_status == "never" else 0,
                                         1 if smoking_status == "former" else 0,
                                         1 if smoking_status == "current" else 0])
                lifestyle_info['smoking_status'] = smoking_status
            else:
                lifestyle_features.extend([0, 0, 0])
                lifestyle_info['smoking_status'] = "Unknown"

            if active_lifestyle is not None:
                lifestyle_features.append(1 if active_lifestyle else 0)
                lifestyle_info['active_lifestyle'] = active_lifestyle
            else:
                lifestyle_features.append(0)
                lifestyle_info['active_lifestyle'] = "Unknown"

            # Gabungkan semua features
            all_features = base_features + lifestyle_features
            
            # Pastikan jumlah features sesuai dengan model
            if len(all_features) != len(self.feature_names):
                # Adjust features to match model
                all_features = base_features + [0] * (len(self.feature_names) - len(base_features))
                all_features = all_features[:len(self.feature_names)]

            input_df = pd.DataFrame([all_features], columns=self.feature_names)

            # Handle potential NaN values in input
            input_df = input_df.fillna(0)

            # Prediksi dengan model
            if hasattr(self.model, 'predict_proba'):
                prediction = self.model.predict(input_df)[0]
                proba = self.model.predict_proba(input_df)[0]
            else:
                prediction = self.model.predict(input_df)[0]
                proba = np.zeros(len(self.label_encoder.classes_))
                proba[prediction] = 1.0

            condition = self.label_encoder.inverse_transform([prediction])[0]
            confidence = float(proba[prediction])

            # Probabilitas untuk semua kondisi
            prob_dict = {
                self.label_encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(proba)
            }

            # Analisis risiko komprehensif
            overall_risk_level = self._calculate_overall_risk_level(
                systole, diastole, heart_rate, temperature, lifestyle_info, prob_dict
            )

            # Rekomendasi spesifik berdasarkan kondisi
            recommendations = self._generate_recommendations(
                condition, lifestyle_info, overall_risk_level
            )

            return {
                "success": True,
                "condition": condition,
                "risk_level": overall_risk_level,
                "description": self.risk_descriptions.get(condition, "Tidak diketahui"),
                "confidence": confidence,
                "probabilities": prob_dict,
                "risk_distribution": self._get_risk_distribution(prob_dict),
                "lifestyle_analysis": lifestyle_info,
                "recommendations": recommendations,
                "alert": overall_risk_level in ["TINGGI", "SANGAT_TINGGI"],
                "medical_disclaimer": "HANYA SKRINING - KONFIRMASI DENGAN TENAGA MEDIS",
                "model_type": "optimal_hybrid"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "condition": "ERROR",
                "risk_level": "TIDAK TERDEFINISI",
                "description": f"Error: {str(e)}",
                "confidence": 0.0,
                "probabilities": {},
                "risk_distribution": {},
                "lifestyle_analysis": {},
                "recommendations": [],
                "alert": False,
                "medical_disclaimer": "Input tidak valid",
                "model_type": "optimal_hybrid"
            }

    def _get_age_group(self, age):
        """Mendapatkan kelompok usia"""
        if age < 30:
            return "Young"
        elif age < 45:
            return "Adult"
        elif age < 60:
            return "Middle-aged"
        else:
            return "Senior"

    def _calculate_overall_risk_level(self, systole, diastole, heart_rate, temperature, lifestyle_info, prob_dict):
        """Menghitung level risiko keseluruhan berdasarkan semua faktor"""
        base_risk_score = 0
        
        # Tekanan darah
        if systole >= 140 or diastole >= 90:
            base_risk_score += 2.0
        elif systole >= 130 or diastole >= 80:
            base_risk_score += 1.0
        elif systole >= 120:
            base_risk_score += 0.5
            
        # Denyut jantung
        if heart_rate > 100:
            base_risk_score += 1.0
        elif heart_rate < 60:
            base_risk_score += 1.0
            
        # Probabilitas kondisi berisiko tinggi
        high_risk_conditions = ["HYPERTENSION_STAGE2", "TACHYCARDIA", "BRADYCARDIA"]
        for condition in high_risk_conditions:
            base_risk_score += prob_dict.get(condition, 0) * 1.5
        
        # Faktor lifestyle
        lifestyle_score = self.calculate_risk_score(
            lifestyle_info.get('bmi_category', 'Normal'),
            lifestyle_info.get('age', 30),
            lifestyle_info.get('smoking_status', 'never'),
            lifestyle_info.get('active_lifestyle', True)
        )
        
        total_risk_score = base_risk_score + lifestyle_score
        
        # Konversi ke level risiko
        if total_risk_score >= 2.5:
            return "SANGAT_TINGGI"
        elif total_risk_score >= 1.5:
            return "TINGGI"
        elif total_risk_score >= 0.8:
            return "SEDANG"
        else:
            return "RENDAH"

    def _get_risk_distribution(self, prob_dict):
        """Mendapatkan distribusi risiko dalam format yang lebih detail"""
        distribution = {}
        
        # Kelompokkan berdasarkan jenis risiko
        hypertension_risk = prob_dict.get("HYPERTENSION_STAGE1", 0) + prob_dict.get("HYPERTENSION_STAGE2", 0) + prob_dict.get("BORDERLINE_HYPERTENSION", 0)
        heart_rhythm_risk = prob_dict.get("TACHYCARDIA", 0) + prob_dict.get("BRADYCARDIA", 0)
        normal_prob = prob_dict.get("NORMAL", 0)
        
        distribution["hipertensi"] = {
            "total": hypertension_risk * 100,
            "stage1": prob_dict.get("HYPERTENSION_STAGE1", 0) * 100,
            "stage2": prob_dict.get("HYPERTENSION_STAGE2", 0) * 100,
            "borderline": prob_dict.get("BORDERLINE_HYPERTENSION", 0) * 100
        }
        
        distribution["gangguan_irama_jantung"] = {
            "total": heart_rhythm_risk * 100,
            "tachycardia": prob_dict.get("TACHYCARDIA", 0) * 100,
            "bradycardia": prob_dict.get("BRADYCARDIA", 0) * 100
        }
        
        distribution["normal"] = normal_prob * 100
        distribution["risiko_tinggi"] = (hypertension_risk + heart_rhythm_risk) * 100
        
        return distribution

    def _generate_recommendations(self, condition, lifestyle_info, risk_level):
        """Menghasilkan rekomendasi berdasarkan kondisi dan lifestyle"""
        recommendations = []
        
        # Rekomendasi berdasarkan kondisi
        if condition in ["HYPERTENSION_STAGE1", "HYPERTENSION_STAGE2", "BORDERLINE_HYPERTENSION"]:
            recommendations.extend([
                "Monitor tekanan darah secara teratur",
                "Kurangi asupan garam",
                "Pertahankan berat badan ideal",
                "Konsultasi dengan dokter untuk evaluasi lebih lanjut"
            ])
            
        if condition in ["TACHYCARDIA", "BRADYCARDIA"]:
            recommendations.extend([
                "Monitor denyut jantung secara teratur",
                "Hindari stimulan berlebihan (kafein, nikotin)",
                "Kelola stres dengan teknik relaksasi",
                "Konsultasi kardiologis jika berlanjut"
            ])
        
        # Rekomendasi berdasarkan lifestyle
        bmi_category = lifestyle_info.get('bmi_category')
        if bmi_category in ["Overweight", "Obese"]:
            recommendations.append("Program penurunan berat badan direkomendasikan")
            
        smoking_status = lifestyle_info.get('smoking_status')
        if smoking_status == "current":
            recommendations.append("Pertimbangkan program berhenti merokok")
            
        active_lifestyle = lifestyle_info.get('active_lifestyle')
        if not active_lifestyle:
            recommendations.append("Tingkatkan aktivitas fisik minimal 30 menit/hari")
        
        # Rekomendasi berdasarkan usia
        age_group = lifestyle_info.get('age_group')
        if age_group in ["Middle-aged", "Senior"]:
            recommendations.append("Pemeriksaan kesehatan rutin tahunan direkomendasikan")
        
        # Rekomendasi berdasarkan level risiko
        if risk_level in ["TINGGI", "SANGAT_TINGGI"]:
            recommendations.append("PENDEKATAN SEGERA: Konsultasi medis secepatnya")
            
        return list(set(recommendations))  # Remove duplicates

    def _get_systolic_category(self, systole):
        if systole < 120:
            return 1
        elif systole < 130:
            return 2
        elif systole < 140:
            return 3
        elif systole < 180:
            return 4
        else:
            return 5

    def _get_diastolic_category(self, diastole):
        if diastole < 80:
            return 1
        elif diastole < 85:
            return 2
        elif diastole < 90:
            return 3
        elif diastole < 120:
            return 4
        else:
            return 5

    def _get_heart_rate_category(self, heart_rate):
        if heart_rate < 60:
            return 1
        elif heart_rate <= 100:
            return 2
        elif heart_rate <= 130:
            return 3
        else:
            return 4

# Initialize predictor
predictor = None
if use_model:
    predictor = CardiovascularPredictor(pipeline, label_encoder, feature_names)

@app.route("/")
def index():
    """Halaman utama"""
    return render_template("index.html")

@app.route("/model-info")
def model_info():
    """Informasi tentang model yang digunakan"""
    if not use_model:
        return jsonify({"error": "Model information not available"}), 404
    
    return jsonify({
        "model_name": "Cardiovascular Risk Predictor - Optimal Hybrid",
        "strategy": model_data.get("strategy", "optimal_hybrid"),
        "accuracy": "94.9%",
        "balanced_accuracy": "92.5%", 
        "f1_score": "95.7%",
        "classes": list(label_encoder.classes_) if label_encoder else [],
        "features_count": len(feature_names) if feature_names else 0,
        "model_type": "Hybrid SVM + Random Forest (Optimal)",
        "status": "Production Ready",
        "training_samples": "17,330",
        "performance": {
            "accuracy": 0.949,
            "balanced_accuracy": 0.925,
            "f1_score": 0.957,
            "train_test_gap": 0.011
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint untuk prediksi risiko kardiovaskular"""
    if not use_model or predictor is None:
        return jsonify({"error": "Model tidak tersedia. Silakan jalankan training terlebih dahulu."}), 503
    
    try:
        data = request.get_json(force=True)
        
        # Extract basic vital signs
        systole = float(data.get("systole", 0))
        diastole = float(data.get("diastole", 0))
        heart_rate = float(data.get("heart_rate", 0))
        temperature = float(data.get("temperature", 36.5))
        
        # Extract optional lifestyle data
        weight = data.get("weight")
        height = data.get("height")
        age = data.get("age")
        smoking_status = data.get("smoking_status")  # "never", "former", "current"
        active_lifestyle = data.get("active_lifestyle")  # True/False
        
        # Convert types if provided
        if weight is not None:
            weight = float(weight)
        if height is not None:
            height = float(height)
        if age is not None:
            age = int(age)
        if active_lifestyle is not None:
            active_lifestyle = bool(active_lifestyle)
        
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Format input tidak valid: {str(e)}"}), 400

    try:
        # Gunakan predictor comprehensive
        result = predictor.predict_comprehensive_risk(
            systole=systole, diastole=diastole, heart_rate=heart_rate, temperature=temperature,
            weight=weight, height=height, age=age,
            smoking_status=smoking_status, active_lifestyle=active_lifestyle
        )
        
        if result["success"]:
            # Format response untuk frontend
            response = {
                "success": True,
                "condition": result["condition"],
                "risk_level": result["risk_level"],
                "risk_description": result["description"],
                "confidence_percent": round(result["confidence"] * 100, 1),
                "probabilities_percent": {k: round(v * 100, 1) for k, v in result["probabilities"].items()},
                "risk_distribution": result["risk_distribution"],
                "lifestyle_analysis": result["lifestyle_analysis"],
                "recommendations": result["recommendations"],
                "alert": result["alert"],
                "medical_disclaimer": result["medical_disclaimer"],
                "model_type": result["model_type"],
                "timestamp": datetime.datetime.now().isoformat()
            }
            return jsonify(response)
        else:
            return jsonify({
                "success": False,
                "error": result.get("error", "Unknown error"),
                "condition": result.get("condition", "ERROR"),
                "medical_disclaimer": result.get("medical_disclaimer", "Terjadi kesalahan sistem")
            }), 500
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error during prediction: {str(e)}",
            "condition": "ERROR",
            "risk_level": "TIDAK TERDEFINISI",
            "medical_disclaimer": "Terjadi kesalahan sistem"
        }), 500

@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if use_model else "model_not_loaded",
        "model_loaded": use_model,
        "timestamp": datetime.datetime.now().isoformat(),
        "model_info": {
            "strategy": model_data.get("strategy", "unknown") if model_data else "unknown",
            "features": len(feature_names) if feature_names else 0,
            "classes": list(label_encoder.classes_) if label_encoder else []
        }
    })

@app.route("/test-prediction", methods=["GET"])
def test_prediction():
    """Endpoint untuk testing prediksi dengan data sample"""
    if not use_model or predictor is None:
        return jsonify({"error": "Model not available"}), 503
    
    # Test dengan data sample sehat
    sample_data = {
        "systole": 115,
        "diastole": 75, 
        "heart_rate": 72,
        "temperature": 36.5,
        "weight": 70,
        "height": 1.75,
        "age": 35,
        "smoking_status": "never",
        "active_lifestyle": True
    }
    
    try:
        result = predictor.predict_comprehensive_risk(**sample_data)
        return jsonify({
            "test_data": sample_data,
            "prediction_result": result
        })
    except Exception as e:
        return jsonify({
            "error": f"Test prediction failed: {str(e)}",
            "test_data": sample_data
        }), 500

if __name__ == "__main__":
    print("🚀 Starting Cardiovascular Risk Prediction API...")
    print(f"📊 Model Status: {'✅ Loaded' if use_model else '❌ Not Available'}")
    if use_model:
        print(f"🎯 Strategy: {model_data.get('strategy', 'Unknown')}")
        print(f"📈 Features: {len(feature_names)}")
        print(f"🏥 Classes: {list(label_encoder.classes_)}")
        print(f"🔧 Model Type: Hybrid SVM + Random Forest (Optimal)")
    
    print("\n📋 Available Endpoints:")
    print("   GET  /              - Home page")
    print("   GET  /model-info    - Model information") 
    print("   GET  /health        - Health check")
    print("   GET  /test-prediction - Test prediction with sample data")
    print("   POST /predict       - Make prediction")
    
    app.run(host="0.0.0.0", port=5000, debug=True)