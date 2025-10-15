import joblib
import pandas as pd
import os

MODEL_PATH_DEFAULT = 'models/sleep_model_v1.joblib'

def load_model(path=MODEL_PATH_DEFAULT):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Train first.")
    obj = joblib.load(path)
    pipeline = obj['pipeline']
    label_encoder = obj['label_encoder']
    return pipeline, label_encoder

def prepare_input_row(input_dict):
    """
    Accepts a dict of raw inputs and returns a one-row DataFrame matching training features.
    Expected keys (strings):
      sleep_duration (float hours),
      bedtime (HH:MM),
      wakeup_time (HH:MM),
      caffeine_intake (None/Low/Moderate/High),
      exercise_duration (int minutes),
      screen_time_before_bed (int),
      stress_level (0-10),
      mood (Happy/Neutral/Sad/Anxious),
      sleep_interruptions (Yes/No)
    """
    def time_to_minutes(ts):
        try:
            hh, mm = str(ts).split(':')
            return int(hh) * 60 + int(mm)
        except Exception:
            return None

    row = {}
    row['sleep_duration'] = float(input_dict.get('sleep_duration', 0.0))
    row['exercise_duration'] = int(input_dict.get('exercise_duration', 0))
    row['screen_time_before_bed'] = int(input_dict.get('screen_time_before_bed', 0))
    row['stress_level'] = int(input_dict.get('stress_level', 0))
    row['bedtime_minutes'] = time_to_minutes(input_dict.get('bedtime', '23:00'))
    row['wakeup_minutes'] = time_to_minutes(input_dict.get('wakeup_time', '07:00'))
    row['caffeine_intake'] = input_dict.get('caffeine_intake', 'Low')
    row['mood'] = input_dict.get('mood', 'Neutral')
    row['sleep_interruptions'] = input_dict.get('sleep_interruptions', 'No')
    return pd.DataFrame([row])

def generate_tips(row):
    tips = []
    if row['sleep_duration'].iloc[0] < 7:
        tips.append("Try increasing sleep to at least 7 hours.")
    if row['screen_time_before_bed'].iloc[0] > 60:
        tips.append("Reduce screen time 30–60 minutes before bed.")
    if row['exercise_duration'].iloc[0] < 30:
        tips.append("Aim for ≥30 minutes of exercise during the day.")
    if row['caffeine_intake'].iloc[0] in ['Moderate','High']:
        tips.append("Avoid caffeine after mid-afternoon.")
    if row['stress_level'].iloc[0] > 5:
        tips.append("Try relaxation (breathing, meditation) before bed to reduce stress.")
    if row['sleep_interruptions'].iloc[0] == 'Yes':
        tips.append("If you wake frequently, keep a sleep log and consult a physician if persistent.")
    if not tips:
        tips.append("Good! Maintain consistent sleep routines and habits.")
    return tips

def predict(input_dict, model_path=MODEL_PATH_DEFAULT):
    pipeline, le = load_model(model_path)
    df = prepare_input_row(input_dict)
    pred_enc = pipeline.predict(df)
    pred_label = le.inverse_transform(pred_enc)[0]
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(df)[0].max()
    else:
        proba = None
    tips = generate_tips(df)
    return {'label': pred_label, 'confidence': float(proba) if proba is not None else None, 'tips': tips}
