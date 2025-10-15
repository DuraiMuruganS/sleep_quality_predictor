import os
import subprocess
import sys
import time
from src.predict import predict

MODEL_PATH = 'models/sleep_model_v1.joblib'
DATA_PATH = 'data/sleep_data.csv'

def test_training_and_prediction():
    # ensure data exists
    if not os.path.exists(DATA_PATH):
        subprocess.check_call([sys.executable, 'src/generate_data.py', '--out', DATA_PATH, '--n', '500', '--force'])
    # train if model missing
    if not os.path.exists(MODEL_PATH):
        subprocess.check_call([sys.executable, 'src/train_model.py', '--data', DATA_PATH, '--out', MODEL_PATH])
    sample = {
        'bedtime': '23:30',
        'wakeup_time': '07:00',
        'sleep_duration': 7.5,
        'caffeine_intake': 'Low',
        'exercise_duration': 40,
        'screen_time_before_bed': 30,
        'stress_level': 3,
        'mood': 'Happy',
        'sleep_interruptions': 'No'
    }
    res = predict(sample, model_path=MODEL_PATH)
    assert res['label'] in ['Good','Average','Poor']
