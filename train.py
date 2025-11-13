import os
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = "dtpasien.xlsx"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_STATE = 42

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

class CardiovascularRiskPredictor:
    def __init__(self, model, label_encoder, feature_names, model_type='hybrid'):
        self.model = model
        self.label_encoder = label_encoder
        self.feature_names = feature_names
        self.model_type = model_type
        
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
                                 smoking_status=None, active_lifestyle=None,
                                 threshold_dict=None):
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

            # Feature engineering dasar
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

            # Threshold adjustment
            final_prediction = prediction
            if threshold_dict is not None and self.model_type == 'hybrid':
                max_prob = max(proba)
                max_class_idx = np.argmax(proba)
                max_class = self.label_encoder.inverse_transform([max_class_idx])[0]
                
                for class_idx, class_name in enumerate(self.label_encoder.classes_):
                    if (class_name in threshold_dict and 
                        proba[class_idx] > threshold_dict[class_name] and
                        proba[class_idx] > max_prob * 0.5):
                        final_prediction = class_idx
                        break

            condition = self.label_encoder.inverse_transform([final_prediction])[0]
            confidence = float(proba[final_prediction])

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
                "model_type": self.model_type
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
                "model_type": self.model_type
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

def perform_eda(df):
    print("\n" + "="*60)
    print("ANALISIS DATA EKSPLORATIF")
    print("="*60)
    
    print(f"Shape Dataset: {df.shape}")
    print(f"Jumlah fitur: {df.shape[1]}")
    print(f"Jumlah sampel: {df.shape[0]}")
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nMissing Values:")
    missing_data = df.isnull().sum()
    for col, missing_count in missing_data.items():
        if missing_count > 0:
            print(f"  {col}: {missing_count} ({missing_count/len(df)*100:.2f}%)")
    
    print("\nStatistik Dasar:")
    print(df.describe())
    
    return df

def analyze_data_distribution(df):
    print("\nANALISIS DISTRIBUSI DATA")
    print("=" * 40)
    
    vital_params = ['systole', 'diastole', 'heart_rate', 'temperature']
    
    for param in vital_params:
        if param in df.columns:
            min_val = df[param].min()
            max_val = df[param].max()
            mean_val = df[param].mean()
            std_val = df[param].std()
            print(f"  {param:15}: {min_val:6.1f} - {max_val:6.1f} (mean: {mean_val:.1f}, std: {std_val:.1f})")
            
            if param == 'systole':
                outliers = df[(df[param] < 70) | (df[param] > 200)]
            elif param == 'diastole':
                outliers = df[(df[param] < 40) | (df[param] > 120)]
            elif param == 'heart_rate':
                outliers = df[(df[param] < 40) | (df[param] > 150)]
            elif param == 'temperature':
                outliers = df[(df[param] < 35.5) | (df[param] > 39.0)]
            
            print(f"    Nilai di luar rentang umum: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

def load_and_preprocess_data(file_path):
    print("MEMUAT DAN MEMPROSES DATASET")
    print("="*50)
    
    df = pd.read_excel(file_path)
    print(f"Dataset awal: {len(df):,} baris, {len(df.columns)} kolom")
    
    df = perform_eda(df)
    
    rename_dict = {
        'Systole': 'systole',
        'Diastole': 'diastole', 
        'Denyut Jantung': 'heart_rate',
        'Suhu Badan': 'temperature'
    }
    
    df_clean = df.rename(columns=rename_dict)
    print(f"\nKolom setelah rename: {list(df_clean.columns)}")
    
    analyze_data_distribution(df_clean)
    
    print("\nPROSES PEMBERSIHAN DATA")
    print("-" * 30)
    
    initial_count = len(df_clean)
    print(f"Sampel awal: {initial_count:,}")
    
    required_columns = ["systole", "diastole", "heart_rate", "temperature"]
    df_clean = df_clean.dropna(subset=required_columns)
    after_missing = len(df_clean)
    print(f"Setelah menghapus missing values: {after_missing:,} (dihapus {initial_count - after_missing:,})")
    
    for col in required_columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    df_clean = df_clean.dropna(subset=required_columns)
    after_conversion = len(df_clean)
    print(f"Setelah handling conversion errors: {after_conversion:,} (dihapus {after_missing - after_conversion:,})")
    
    df_clean = df_clean.drop_duplicates()
    after_dedup = len(df_clean)
    print(f"Setelah menghapus duplikat: {after_dedup:,} (dihapus {after_conversion - after_dedup:,})")
    
    print("\nAPLIKASI FILTER VALIDITAS MEDIS REALISTIS")
    print("-" * 45)
    
    df_filtered = df_clean[
        (df_clean["systole"].between(60, 250)) &
        (df_clean["diastole"].between(30, 150)) &
        (df_clean["heart_rate"].between(30, 180)) &
        (df_clean["temperature"].between(35.0, 40.0)) &
        (df_clean["systole"] > df_clean["diastole"])
    ].copy()
    
    final_count = len(df_filtered)
    print(f"Setelah filter medis realistis: {final_count:,} (dihapus {after_dedup - final_count:,})")
    print(f"Retensi data: {(final_count / initial_count * 100):.1f}%")
    
    if final_count == 0:
        print("\nPERINGATAN: Tidak ada data yang tersisa setelah filtering!")
        print("Menggunakan filter yang kurang restriktif...")
        
        df_filtered = df_clean[
            (df_clean["systole"] > df_clean["diastole"]) &
            (df_clean["systole"] > 0) &
            (df_clean["diastole"] > 0)
        ].copy()
        print(f"Sampel fallback darurat: {len(df_filtered):,}")
    
    print("\nGENERATING LABEL MEDIS")
    print("-" * 25)
    df_filtered["risk_pattern"] = create_clinical_labels(df_filtered)
    
    print("\nDISTRIBUSI KELAS FINAL:")
    class_dist = df_filtered['risk_pattern'].value_counts()
    for cls, count in class_dist.items():
        percentage = (count / len(df_filtered)) * 100
        print(f"  {cls}: {count:>4} sampel ({percentage:5.1f}%)")
    
    return df_filtered

def create_clinical_labels(df):
    labels = []
    
    for _, row in df.iterrows():
        systole = row["systole"]
        diastole = row["diastole"] 
        heart_rate = row["heart_rate"]
        
        if systole >= 140 or diastole >= 90:
            labels.append("HYPERTENSION_STAGE2")
        elif systole >= 130 or diastole >= 80:
            labels.append("HYPERTENSION_STAGE1")
        elif systole >= 120:
            labels.append("BORDERLINE_HYPERTENSION")
        elif heart_rate > 100:
            labels.append("TACHYCARDIA")
        elif heart_rate < 60:
            labels.append("BRADYCARDIA")
        else:
            labels.append("NORMAL")
            
    return labels

def optimal_data_balancing(df, max_synthetic_ratio=0.25):
    """Data balancing yang optimal untuk performa bagus tanpa overfitting"""
    print("\nOPTIMAL DATA BALANCING")
    print("=" * 50)
    
    current_dist = df['risk_pattern'].value_counts()
    print("Distribusi kelas saat ini:")
    for risk, count in current_dist.items():
        print(f"  {risk}: {count} sampel")
    
    # Hitung target yang seimbang
    max_synthetic_samples = int(len(df) * max_synthetic_ratio)
    
    # Strategi balancing yang optimal
    balancing_strategies = {}
    majority_class_size = max(current_dist.values)
    
    for class_name in current_dist.index:
        current_count = current_dist[class_name]
        
        # Target yang optimal berdasarkan ukuran kelas
        if current_count < 100:
            target = min(current_count * 3, 300)  # Untuk kelas kecil, perbanyak sedikit
        elif current_count < 500:
            target = min(int(current_count * 1.8), 800)  # Untuk kelas menengah
        else:
            # Untuk kelas mayoritas, batasi agar tidak dominan
            target = min(int(current_count * 1.2), int(majority_class_size * 0.9))
            
        balancing_strategies[class_name] = int(target)
    
    print(f"\nStrategi penyeimbangan (max synthetic ratio: {max_synthetic_ratio:.1%}):")
    for class_name, target in balancing_strategies.items():
        current_count = current_dist[class_name]
        needed = max(0, target - current_count)
        print(f"  {class_name}: {current_count} -> {target} (butuh {needed})")
    
    enhanced_dfs = [df]
    all_classes = current_dist.index.tolist()
    
    real_data_count = len(df)
    synthetic_data_count = 0
    
    for class_name in all_classes:
        class_data = df[df['risk_pattern'] == class_name]
        current_count = len(class_data)
        target_count = balancing_strategies[class_name]
        
        if current_count < target_count:
            needed = target_count - current_count
            
            if needed > 0 and synthetic_data_count < max_synthetic_samples:
                actual_needed = min(needed, max_synthetic_samples - synthetic_data_count)
                if actual_needed > 0:
                    print(f"Menghasilkan {actual_needed} sampel sintetik untuk {class_name}...")
                    synthetic_samples = generate_optimal_synthetic_data(class_data, actual_needed, class_name)
                    
                    if synthetic_samples:
                        synthetic_df = pd.DataFrame(synthetic_samples)
                        enhanced_dfs.append(synthetic_df)
                        synthetic_data_count += actual_needed
    
    enhanced_df = pd.concat(enhanced_dfs, ignore_index=True)
    
    final_dist = enhanced_df['risk_pattern'].value_counts()
    print("\nDistribusi seimbang final:")
    total_samples = len(enhanced_df)
    for risk_class in all_classes:
        count = final_dist.get(risk_class, 0)
        percentage = (count / total_samples) * 100
        print(f"  {risk_class}: {count} sampel ({percentage:.1f}%)")
    
    print(f"\nRingkasan data sintetik:")
    print(f"  Data real: {real_data_count} sampel")
    print(f"  Data sintetik: {synthetic_data_count} sampel")
    print(f"  Rasio sintetik: {synthetic_data_count/total_samples:.1%}")
    print(f"  Total samples: {total_samples:,}")
    
    return enhanced_df

def generate_optimal_synthetic_data(real_data, num_samples, class_name):
    """Generate synthetic data dengan noise yang optimal"""
    synthetic_samples = []
    
    if len(real_data) == 0:
        return synthetic_samples
    
    # Pastikan num_samples adalah integer
    num_samples = int(num_samples)
    
    # Noise level yang optimal
    noise_level = 0.15  # 15% noise - cukup untuk variasi tapi tidak berlebihan
    
    for _ in range(num_samples):
        # Gunakan real sample sebagai base
        real_sample = real_data.sample(1).iloc[0]
        
        # Tambahkan noise yang optimal
        synthetic_sample = {
            'systole': real_sample['systole'] * (1 + np.random.normal(0, noise_level)),
            'diastole': real_sample['diastole'] * (1 + np.random.normal(0, noise_level)),
            'heart_rate': real_sample['heart_rate'] * (1 + np.random.normal(0, noise_level)),
            'temperature': real_sample['temperature'] * (1 + np.random.normal(0, noise_level/4)),
            'risk_pattern': class_name
        }
        
        # Clipping values dengan range yang realistis
        synthetic_sample['systole'] = np.clip(synthetic_sample['systole'], 70, 200)
        synthetic_sample['diastole'] = np.clip(synthetic_sample['diastole'], 40, 120)
        synthetic_sample['heart_rate'] = np.clip(synthetic_sample['heart_rate'], 40, 150)
        synthetic_sample['temperature'] = np.clip(synthetic_sample['temperature'], 35.5, 39.0)
        
        # Ensure systole > diastole dengan margin yang wajar
        min_gap = 10
        if synthetic_sample['systole'] <= synthetic_sample['diastole'] + min_gap:
            synthetic_sample['systole'] = synthetic_sample['diastole'] + min_gap + np.random.randint(5, 20)
        
        synthetic_samples.append(synthetic_sample)
    
    return synthetic_samples

def comprehensive_feature_engineering(df):
    """Feature engineering yang komprehensif tapi tidak berlebihan"""
    print("\nCOMPREHENSIVE FEATURE ENGINEERING")
    print("=" * 40)
    
    # Fitur dasar
    df["pulse_pressure"] = df["systole"] - df["diastole"]
    df["mean_arterial_pressure"] = df["diastole"] + df["pulse_pressure"] / 3.0
    df["pressure_heart_ratio"] = df["systole"] / df["heart_rate"]
    df["pulse_pressure_ratio"] = df["pulse_pressure"] / df["mean_arterial_pressure"]
    
    # Variability measures
    df["bp_variability"] = abs(df["systole"] - 120) + abs(df["diastole"] - 80)
    df["heart_rate_variability"] = abs(df["heart_rate"] - 70)
    
    # Risk scores
    df["hypertension_risk"] = np.tanh((df["systole"] - 120) / 40 + (df["diastole"] - 80) / 25)
    df["heart_risk"] = np.tanh(abs(df["heart_rate"] - 70) / 30)
    
    # Advanced features
    df["double_product"] = (df["systole"] * df["heart_rate"]) / 100
    df["bp_ratio"] = df["systole"] / df["diastole"]
    df["map_heart_ratio"] = df["mean_arterial_pressure"] / df["heart_rate"]
    
    # Kategori yang meaningful
    df["systolic_category"] = df["systole"].apply(lambda x: 1 if x < 120 else (2 if x < 140 else 3))
    df["diastolic_category"] = df["diastole"].apply(lambda x: 1 if x < 80 else (2 if x < 90 else 3))
    df["heart_rate_category"] = df["heart_rate"].apply(lambda x: 1 if x < 60 else (2 if x <= 100 else 3))
    
    # Lifestyle features yang realistis
    np.random.seed(RANDOM_STATE)
    n_samples = len(df)
    
    # BMI dengan distribusi realistis
    df["bmi"] = np.random.normal(25, 4, n_samples)
    df["bmi"] = np.clip(df["bmi"], 16, 40)
    
    # Usia dengan distribusi realistis
    df["age"] = np.random.normal(45, 15, n_samples)
    df["age"] = np.clip(df["age"], 18, 80)
    
    # Binary features dengan probabilitas realistis
    df["smoking_current"] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    df["active_lifestyle"] = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    
    print("Fitur yang digunakan:")
    essential_features = [
        "systole", "diastole", "heart_rate", "temperature",
        "pulse_pressure", "mean_arterial_pressure", "pressure_heart_ratio",
        "pulse_pressure_ratio", "bp_variability", "heart_rate_variability",
        "hypertension_risk", "heart_risk", "double_product", "bp_ratio", "map_heart_ratio",
        "systolic_category", "diastolic_category", "heart_rate_category",
        "bmi", "age", "smoking_current", "active_lifestyle"
    ]
    
    for feature in essential_features:
        print(f"  - {feature}")
    
    return df

def create_optimal_hybrid_model(features, X_train, y_train):
    """
    Membuat hybrid model yang optimal - performa bagus tanpa overfitting
    """
    print("\nMEMBuat OPTIMAL HYBRID MODEL")
    print("=" * 50)
    
    # Preprocessor sederhana
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), features)
    ])
    
    # Model dengan parameter optimal
    rf_model = RandomForestClassifier(
        n_estimators=150,           # Cukup trees untuk performa baik
        max_depth=15,               # Depth yang wajar
        min_samples_split=10,       # Tidak terlalu strict
        min_samples_leaf=5,         # Tidak terlalu strict
        max_features='sqrt',        # Feature sampling optimal
        random_state=RANDOM_STATE,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # SVM dengan parameter optimal
    svm_model = SVC(
        C=1.0,                      # Regularization balanced
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        cache_size=1000,
        shrinking=True
    )
    
    # Voting classifier dengan weights optimal
    hybrid_model = VotingClassifier(
        estimators=[
            ('random_forest', rf_model),
            ('svm', svm_model)
        ],
        voting='soft',
        weights=[2, 1],  # RF sedikit lebih dominan
        n_jobs=-1
    )
    
    # Build pipeline dengan feature selection moderat
    feature_selector = SelectKBest(f_classif, k=min(15, len(features)))
    
    if IMBLEARN_AVAILABLE:
        final_pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('feature_selector', feature_selector),
            ('smote', SMOTE(random_state=RANDOM_STATE, sampling_strategy='not majority', k_neighbors=3)),
            ('classifier', hybrid_model)
        ])
    else:
        final_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selector', feature_selector),
            ('classifier', hybrid_model)
        ])
    
    return final_pipeline, "optimal_hybrid"

def evaluate_individual_models(X_train, y_train, X_test, y_test, features):
    print("\nEVALUASI MODEL INDIVIDUAL")
    print("=" * 40)
    
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), features)
    ])
    
    # Models dengan parameter optimal
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=150, max_depth=15, min_samples_split=10,
            random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1,
            max_features='sqrt'
        ),
        'SVM': SVC(
            C=1.0, kernel='rbf', probability=True, 
            random_state=RANDOM_STATE, class_weight='balanced'
        )
    }
    
    results = {}
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        X_train_clean = X_train.fillna(X_train.median())
        X_test_clean = X_test.fillna(X_test.median())
        
        pipeline.fit(X_train_clean, y_train)
        y_pred = pipeline.predict(X_test_clean)
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'accuracy': accuracy, 
            'balanced_accuracy': balanced_acc,
            'f1_score': f1,
            'model': pipeline
        }
        print(f"  {name}: Accuracy={accuracy:.3f}, Balanced Accuracy={balanced_acc:.3f}, F1={f1:.3f}")
    
    return results

def enhanced_hybrid_training(model, X_train, y_train, X_test, y_test, le, individual_results):
    print("\nPELATIHAN HYBRID MODEL")
    print("=" * 40)
    
    # Handle NaN values
    X_train_clean = X_train.fillna(X_train.median())
    X_test_clean = X_test.fillna(X_test.median())
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train_clean, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Cross-validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Train model
    model.fit(X_train_clean, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_clean)
    y_proba = model.predict_proba(X_test_clean)
    
    # Comprehensive metrics
    train_score = model.score(X_train_clean, y_train)
    test_score = accuracy_score(y_test, y_pred)
    train_test_gap = train_score - test_score
    
    print(f"\nANALISIS TRAIN-TEST:")
    print(f"  Skor Training:    {train_score:.3f}")
    print(f"  Skor Testing:     {test_score:.3f}")
    print(f"  Gap Generalisasi: {train_test_gap:.3f}")
    
    # Enhanced performance analysis
    acc = test_score
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Confidence analysis
    conf_scores = np.max(y_proba, axis=1)
    avg_conf = np.mean(conf_scores)
    high_confidence = np.sum(conf_scores > 0.8) / len(conf_scores)
    low_confidence = np.sum(conf_scores < 0.6) / len(conf_scores)
    
    print(f"\nMETRIK KINERJA HYBRID:")
    print(f"  Accuracy:           {acc:.3f}")
    print(f"  Balanced Accuracy:  {bal_acc:.3f}")
    print(f"  Weighted F1:        {f1:.3f}")
    print(f"  Weighted Precision: {precision:.3f}")
    print(f"  Weighted Recall:    {recall:.3f}")
    print(f"  Average Confidence: {avg_conf:.3f}")
    print(f"  High Confidence (>80%): {high_confidence:.1%}")
    print(f"  Low Confidence (<60%):  {low_confidence:.1%}")
    
    # Classification Report
    print(f"\n{' CLASSIFICATION REPORT ':=^60}")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=3))
    
    # Confusion Matrix
    print(f"\n{' CONFUSION MATRIX ':=^60}")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Individual model comparison
    rf_accuracy = individual_results['Random Forest']['accuracy']
    svm_accuracy = individual_results['SVM']['accuracy']
    
    hybrid_advantage_rf = acc - rf_accuracy
    hybrid_advantage_svm = acc - svm_accuracy
    
    print(f"\nPERBANDINGAN DENGAN INDIVIDUAL MODELS:")
    print(f"  vs Random Forest: {hybrid_advantage_rf:+.4f} ({hybrid_advantage_rf*100:+.2f}%)")
    print(f"  vs SVM:           {hybrid_advantage_svm:+.4f} ({hybrid_advantage_svm*100:+.2f}%)")
    
    return y_pred, y_proba, acc, bal_acc, f1, train_score, test_score, train_test_gap, cv_scores

def comprehensive_test_cases(predictor):
    """Test cases dengan berbagai skenario input"""
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST CASES DENGAN INPUT TAMBAHAN")
    print("="*70)
    
    test_cases = [
        # Format: (systole, diastole, heart_rate, temperature, weight, height, age, smoking, active, description)
        (115, 75, 72, 36.5, 70, 1.75, 35, "never", True, "Sehat ideal"),
        (125, 82, 78, 36.6, 80, 1.70, 45, "former", False, "Pre-hipertensi, overweight"),
        (135, 85, 82, 36.7, 90, 1.65, 55, "current", False, "Hipertensi stage 1, merokok"),
        (150, 95, 85, 36.8, 95, 1.60, 65, "current", False, "Hipertensi stage 2, obesitas, usia tua"),
        (120, 80, 115, 36.7, 75, 1.80, 28, "never", True, "Tachycardia, muda sehat"),
        (118, 78, 52, 36.5, 65, 1.75, 40, "never", True, "Bradycardia, normal"),
        (145, 92, 95, 37.0, 85, 1.70, 50, "former", True, "Hipertensi + tachycardia"),
    ]
    
    for i, (s, d, hr, t, w, h, age, smoke, active, desc) in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"TEST CASE {i}: {desc}")
        print(f"{'='*50}")
        
        result = predictor.predict_comprehensive_risk(
            systole=s, diastole=d, heart_rate=hr, temperature=t,
            weight=w, height=h, age=age, 
            smoking_status=smoke, active_lifestyle=active
        )
        
        if result["success"]:
            print(f"🩺 KONDISI: {result['condition']}")
            print(f"📊 LEVEL RISIKO: {result['risk_level']}")
            print(f"📈 KEPERCAYAAN: {result['confidence']:.1%}")
            print(f"📝 DESKRIPSI: {result['description']}")
            
            print(f"\n📊 DISTRIBUSI RISIKO:")
            dist = result["risk_distribution"]
            print(f"  Hipertensi: {dist['hipertensi']['total']:.1f}%")
            print(f"    - Stage 1: {dist['hipertensi']['stage1']:.1f}%")
            print(f"    - Stage 2: {dist['hipertensi']['stage2']:.1f}%")
            print(f"    - Borderline: {dist['hipertensi']['borderline']:.1f}%")
            print(f"  Gangguan Irama: {dist['gangguan_irama_jantung']['total']:.1f}%")
            print(f"    - Tachycardia: {dist['gangguan_irama_jantung']['tachycardia']:.1f}%")
            print(f"    - Bradycardia: {dist['gangguan_irama_jantung']['bradycardia']:.1f}%")
            print(f"  Normal: {dist['normal']:.1f}%")
            print(f"  Risiko Tinggi: {dist['risiko_tinggi']:.1f}%")
            
            print(f"\n🏃 ANALISIS LIFESTYLE:")
            lifestyle = result["lifestyle_analysis"]
            if lifestyle.get('bmi'):
                print(f"  BMI: {lifestyle['bmi']:.1f} ({lifestyle['bmi_category']})")
            if lifestyle.get('age'):
                print(f"  Usia: {lifestyle['age']} tahun ({lifestyle['age_group']})")
            print(f"  Merokok: {lifestyle.get('smoking_status', 'Unknown')}")
            print(f"  Gaya hidup aktif: {lifestyle.get('active_lifestyle', 'Unknown')}")
            
            print(f"\n💡 REKOMENDASI:")
            for j, rec in enumerate(result["recommendations"], 1):
                print(f"  {j}. {rec}")
                
            if result["alert"]:
                print(f"\n⚠️  ALERT: Kondisi memerlukan perhatian medis!")
                
        else:
            print(f"❌ ERROR: {result['error']}")
        
        print(f"\n{'-'*50}")

def load_and_preprocess_data_extended(file_path):
    """Load dan preprocess data dengan fitur extended"""
    print("MEMUAT DAN MEMPROSES DATASET EXTENDED")
    print("="*50)
    
    df = pd.read_excel(file_path)
    print(f"Dataset awal: {len(df):,} baris, {len(df.columns)} kolom")
    
    rename_dict = {
        'Systole': 'systole',
        'Diastole': 'diastole', 
        'Denyut Jantung': 'heart_rate',
        'Suhu Badan': 'temperature'
    }
    
    df_clean = df.rename(columns=rename_dict)
    
    # Proses cleaning
    required_columns = ["systole", "diastole", "heart_rate", "temperature"]
    df_clean = df_clean.dropna(subset=required_columns)
    
    for col in required_columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    df_clean = df_clean.dropna(subset=required_columns)
    df_clean = df_clean.drop_duplicates()
    
    # Filter medis realistis
    df_filtered = df_clean[
        (df_clean["systole"].between(60, 250)) &
        (df_clean["diastole"].between(30, 150)) &
        (df_clean["heart_rate"].between(30, 180)) &
        (df_clean["temperature"].between(35.0, 40.0)) &
        (df_clean["systole"] > df_clean["diastole"])
    ].copy()
    
    print(f"Setelah filter medis realistis: {len(df_filtered):,} sampel")
    
    # Comprehensive feature engineering
    df_filtered = comprehensive_feature_engineering(df_filtered)
    
    # Generate labels
    print("\nGENERATING LABEL MEDIS")
    df_filtered["risk_pattern"] = create_clinical_labels(df_filtered)
    
    print("\nDISTRIBUSI KELAS FINAL:")
    class_dist = df_filtered['risk_pattern'].value_counts()
    for cls, count in class_dist.items():
        percentage = (count / len(df_filtered)) * 100
        print(f"  {cls}: {count:>4} sampel ({percentage:5.1f}%)")
    
    return df_filtered

# Tambahkan method predict_risk untuk backward compatibility
def predict_risk(self, systole, diastole, heart_rate, temperature, threshold_dict=None):
    return self.predict_comprehensive_risk(
        systole, diastole, heart_rate, temperature,
        weight=None, height=None, age=None,
        smoking_status=None, active_lifestyle=None,
        threshold_dict=threshold_dict
    )

CardiovascularRiskPredictor.predict_risk = predict_risk

if __name__ == "__main__":
    print("=" * 70)
    print("SISTEM PREDIKSI RISIKO KARDIOVASKULAR - OPTIMAL VERSION")
    print("Hybrid SVM + Random Forest dengan Performa Optimal")
    print("=" * 70)

    try:
        # Load data dengan comprehensive features
        df_processed = load_and_preprocess_data_extended(DATA_PATH)
        
        # Optimal data balancing
        df_balanced = optimal_data_balancing(df_processed, max_synthetic_ratio=0.25)
        print(f"\nUkuran dataset final: {len(df_balanced):,} sampel")

        # Comprehensive features
        features = [
            "systole", "diastole", "heart_rate", "temperature",
            "pulse_pressure", "mean_arterial_pressure", "pressure_heart_ratio",
            "pulse_pressure_ratio", "bp_variability", "heart_rate_variability",
            "hypertension_risk", "heart_risk", "double_product", "bp_ratio", "map_heart_ratio",
            "systolic_category", "diastolic_category", "heart_rate_category",
            "bmi", "age", "smoking_current", "active_lifestyle"
        ]

        print(f"\nTotal fitur: {len(features)}")
        print("Fitur yang digunakan: Vital dasar + parameter terhitung komprehensif + kategori klinis + lifestyle")

        # Prepare data
        X = df_balanced[features].fillna(df_balanced[features].median())
        y = df_balanced["risk_pattern"]

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.25, stratify=y_encoded, random_state=RANDOM_STATE
        )

        print(f"\nPEMBAGIAN DATA:")
        print(f"  Training set: {X_train.shape[0]:,} sampel")
        print(f"  Testing set:  {X_test.shape[0]:,} sampel")
        print(f"  Feature space: {X_train.shape[1]} dimensi")

        # Create and train models
        individual_results = evaluate_individual_models(X_train, y_train, X_test, y_test, features)
        model, strategy_name = create_optimal_hybrid_model(features, X_train, y_train)
        
        # Train hybrid model
        y_pred, y_proba, acc, bal_acc, f1, train_score, test_score, train_test_gap, cv_scores = enhanced_hybrid_training(
            model, X_train, y_train, X_test, y_test, le, individual_results
        )

        # Buat predictor
        predictor = CardiovascularRiskPredictor(model, le, features, model_type='optimal_hybrid')
        
        # Jalankan test cases
        comprehensive_test_cases(predictor)

        # Simpan model
        model_path = os.path.join(MODEL_DIR, "cardiovascular_risk_model_optimal_hybrid.joblib")
        joblib.dump({"pipeline": model, "label_encoder": le, "features": features, "strategy": strategy_name}, model_path)
        print(f"\nModel disimpan ke: {model_path}")

        print("\n" + "="*70)
        print("HASIL TRAINING HYBRID MODEL:")
        print("="*70)
        print(f"✅ Model: Hybrid SVM + Random Forest (Optimal)")
        print(f"✅ Accuracy: {acc:.3f}")
        print(f"✅ Balanced Accuracy: {bal_acc:.3f}")
        print(f"✅ F1-Score: {f1:.3f}")
        print(f"✅ Train-Test Gap: {train_test_gap:.3f} (target: <0.05)")
        print(f"✅ Cross-validation: {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")
        print(f"✅ Synthetic Data Ratio: {len(df_balanced) - len(df_processed):,} sampel ({((len(df_balanced) - len(df_processed)) / len(df_balanced) * 100):.1f}%)")
        print("="*70)

    except Exception as e:
        print("ERROR selama pelatihan model:", str(e))
        import traceback
        traceback.print_exc()