import pandas as pd
from datetime import datetime

def time_str_to_minutes(ts):
    """'HH:MM' -> minutes since midnight (int)"""
    try:
        hh, mm = ts.split(':')
        return int(hh) * 60 + int(mm)
    except Exception:
        return None

def prepare_features(df):
    """
    Ensure expected columns exist and clean lightly.
    Returns (X, y) where y may be None if target not present.
    """
    df = df.copy()
    # compute minutes fields if missing
    if 'bedtime_minutes' not in df.columns and 'bedtime' in df.columns:
        df['bedtime_minutes'] = df['bedtime'].apply(lambda x: time_str_to_minutes(x))
    if 'wakeup_minutes' not in df.columns and 'wakeup_time' in df.columns:
        df['wakeup_minutes'] = df['wakeup_time'].apply(lambda x: time_str_to_minutes(x))

    # expected feature set
    feature_cols = [
        'sleep_duration',
        'exercise_duration',
        'screen_time_before_bed',
        'stress_level',
        'bedtime_minutes',
        'wakeup_minutes',
        'caffeine_intake',
        'mood',
        'sleep_interruptions'
    ]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    X = df[feature_cols].copy()
    y = df['sleep_quality'].copy() if 'sleep_quality' in df.columns else None
    return X, y
