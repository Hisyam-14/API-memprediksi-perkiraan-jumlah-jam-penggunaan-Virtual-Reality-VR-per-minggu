from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Inisialisasi FastAPI
app = FastAPI(title="Education Inequality Prediction API")

# Load model dan scaler
scaler = joblib.load("scaler_feature.pkl")  # Scaler untuk normalisasi
rf_model = joblib.load("rf_vr_education_model.pkl")  # Random Forest Regressor model

# Definisi struktur input
class InputData(BaseModel):
    Age: int
    Gender: str
    Grade_Level: str
    Field_of_Study: str
    Usage_of_VR_in_Education: str
    Hours_of_VR_Usage_Per_Week: int
    Engagement_Level: int
    Improvement_in_Learning_Outcomes: str
    Subject: str
    Instructor_VR_Proficiency: str
    Perceived_Effectiveness_of_VR: int
    Access_to_VR_Equipment: str
    Impact_on_Creativity: int
    Stress_Level_with_VR_Usage: str
    Collaboration_with_Peers_via_VR: str
    Feedback_from_Educators_on_VR: str
    Interest_in_Continuing_VR_Based_Learning: str
    Region: str
    School_Support_for_VR_in_Curriculum: str

# Fungsi Preprocessing
def preprocess_input(data: InputData):
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    # Mapping kategori
    usage_mapping = {'Low': 0, 'Moderate': 1, 'Average': 2, 'High': 3, 'Very High': 4}
    subject_mapping = {'Physics': 0, 'Biology': 1, 'Chemistry': 2, 'Mathematics': 3,
                       'History': 4, 'Geography': 5, 'Literature': 6, 'Computer Science': 7}
    region_mapping = {'South Asia': 0, 'East Asia': 1, 'Europe': 2}
    gender_mapping = {'Male': 0, 'Female': 1}
    feedback_mapping = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
    grade_mapping = {'Undergraduate': 0, 'Graduate': 1, 'High School': 2, 'Middle School': 3} # Contoh mapping
    improvement_mapping = {'Yes': 1, 'No': 0, 'Neutral': 2} # Contoh mapping
    proficiency_mapping = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2} # Contoh mapping
    access_mapping = {'Yes': 1, 'No': 0} # Contoh mapping
    stress_mapping = {'Low': 0, 'Moderate': 1, 'High': 2} # Contoh mapping
    collaboration_mapping = {'Yes': 1, 'No': 0} # Contoh mapping
    interest_mapping = {'Yes': 1, 'No': 0, 'Maybe': 2} # Contoh mapping
    support_mapping = {'Yes': 1, 'No': 0, 'Partial': 2} # Contoh mapping

    df['Usage_of_VR_in_Education'] = df['Usage_of_VR_in_Education'].map(usage_mapping)
    df['Subject'] = df['Subject'].map(subject_mapping)
    df['Region'] = df['Region'].map(region_mapping)
    df['Gender'] = df['Gender'].map(gender_mapping)
    df['Feedback_from_Educators_on_VR'] = df['Feedback_from_Educators_on_VR'].map(feedback_mapping)
    df['Grade_Level'] = df['Grade_Level'].map(grade_mapping) # Terapkan mapping untuk Grade_Level
    df['Improvement_in_Learning_Outcomes'] = df['Improvement_in_Learning_Outcomes'].map(improvement_mapping) # Terapkan mapping
    df['Instructor_VR_Proficiency'] = df['Instructor_VR_Proficiency'].map(proficiency_mapping) # Terapkan mapping
    df['Access_to_VR_Equipment'] = df['Access_to_VR_Equipment'].map(access_mapping) # Terapkan mapping
    df['Stress_Level_with_VR_Usage'] = df['Stress_Level_with_VR_Usage'].map(stress_mapping) # Terapkan mapping
    df['Collaboration_with_Peers_via_VR'] = df['Collaboration_with_Peers_via_VR'].map(collaboration_mapping) # Terapkan mapping
    df['Interest_in_Continuing_VR_Based_Learning'] = df['Interest_in_Continuing_VR_Based_Learning'].map(interest_mapping) # Terapkan mapping
    df['School_Support_for_VR_in_Curriculum'] = df['School_Support_for_VR_in_Curriculum'].map(support_mapping) # Terapkan mapping

    # One-hot encoding untuk kolom Field_of_Study
    field_of_study_dummies = pd.get_dummies(df['Field_of_Study'], prefix='Field_of_Study')
    df = pd.concat([df, field_of_study_dummies], axis=1)
    df = df.drop('Field_of_Study', axis=1)

    # Drop kolom yang seharusnya tidak ikut
    if 'Hours_of_VR_Usage_Per_Week' in df.columns:
        df = df.drop('Hours_of_VR_Usage_Per_Week', axis=1)

    # Scaling kolom numerik
    scaler = StandardScaler()
    scaler.fit(df[['Age']])

    # Pastikan semua feature sama urutannya dengan model
    expected_features = rf_model.feature_names_in_
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0  # tambahkan kolom yang hilang dengan nilai 0

    df = df[expected_features]  # urutkan kolom sesuai yang diharapkan model

    return df

# Endpoint root
@app.get("/")
def read_root():
    return {"message": "VR Education Prediction API is running"}

# Endpoint prediksi
@app.post("/predict")
def predict_output(data: InputData):
    df_processed = preprocess_input(data)
    prediction = rf_model.predict(df_processed)[0]
    return {"predicted_hours_of_vr_usage_per_week": int(prediction)}