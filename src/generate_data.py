#!/usr/bin/env python3
"""
Generate synthetic sleep dataset.
Columns:
- bedtime (HH:MM)
- wakeup_time (HH:MM)
- bedtime_minutes, wake_minutes (numeric)
- sleep_duration (hours, float)
- caffeine_intake (None/Low/Moderate/High)
- exercise_duration (minutes)
- screen_time_before_bed (minutes)
- stress_level (0-10)
- mood (Happy/Neutral/Sad/Anxious)
- sleep_interruptions (Yes/No)
- sleep_quality (Good/Average/Poor)
"""
import argparse
import csv
import os
import random
from math import floor
import numpy as np
import pandas as pd

RNG = np.random.RandomState(42)

def _min_to_hhmm(m):
    m = int(round(m)) % (24*60)
    hh = m // 60
    mm = m % 60
    return f"{hh:02d}:{mm:02d}"

def generate_row():
    # bedtime: between 22:00 and 02:00 next day (expressed as absolute minutes > 1440 possible)
    bed_abs = RNG.randint(22*60, 26*60)   # 22:00 .. 02:00 (as absolute)
    wake_abs = RNG.randint(5*60, 10*60)   # 05:00 .. 10:00
    # compute duration in minutes
    duration_min = (wake_abs + 24*60 - bed_abs) if bed_abs > 24*60 else (wake_abs + 24*60 - bed_abs)
    # add small noise
    duration_min = max(3*60, int(duration_min + RNG.normal(0, 20)))
    sleep_duration_h = round(duration_min / 60.0 + RNG.normal(0, 0.2), 2)

    bedtime = _min_to_hhmm(bed_abs)
    wakeup_time = _min_to_hhmm(wake_abs)

    caffeine = RNG.choice(['None', 'Low', 'Moderate', 'High'], p=[0.25,0.45,0.2,0.1])
    exercise = int(abs(RNG.normal(30, 25)))  # minutes
    screen_time = int(abs(RNG.normal(60, 50)))
    stress = int(np.clip(round(RNG.normal(4, 2)), 0, 10))
    mood = RNG.choice(['Happy','Neutral','Sad','Anxious'], p=[0.35,0.35,0.2,0.1])
    interruptions = RNG.choice(['No','Yes'], p=[0.8,0.2])

    # scoring to assign label (handcrafted but reasonable)
    score = 0
    if sleep_duration_h >= 7:
        score += 2
    elif sleep_duration_h >= 6:
        score += 1

    if exercise >= 30:
        score += 1
    if screen_time <= 60:
        score += 1
    if stress <= 3:
        score += 1
    if caffeine in ['None','Low']:
        score += 1
    if interruptions == 'No':
        score += 1

    # label thresholds
    if score >= 5:
        label = 'Good'
    elif score >= 3:
        label = 'Average'
    else:
        label = 'Poor'

    return {
        'bedtime': bedtime,
        'wakeup_time': wakeup_time,
        'bedtime_minutes': int((int(bedtime.split(':')[0]) * 60 + int(bedtime.split(':')[1]))),
        'wakeup_minutes': int((int(wakeup_time.split(':')[0]) * 60 + int(wakeup_time.split(':')[1]))),
        'sleep_duration': float(round(sleep_duration_h, 2)),
        'caffeine_intake': caffeine,
        'exercise_duration': exercise,
        'screen_time_before_bed': screen_time,
        'stress_level': stress,
        'mood': mood,
        'sleep_interruptions': interruptions,
        'sleep_quality': label
    }

def generate_csv(out_path='data/sleep_data.csv', n=2000, force=False):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    if os.path.exists(out_path) and not force:
        print(f"{out_path} already exists. Use --force to overwrite.")
        return
    rows = [generate_row() for _ in range(int(n))]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Generated {n} rows -> {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/sleep_data.csv')
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    generate_csv(out_path=args.out, n=args.n, force=args.force)
