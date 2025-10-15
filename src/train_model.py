#!/usr/bin/env python3
import argparse
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

from src.utils import prepare_features

def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', cat_pipeline, categorical_features)
    ])
    return preprocessor

def train(data_path, out_path, random_state=42, test_size=0.2):
    df = pd.read_csv(data_path)
    X, y = prepare_features(df)
    # numeric / categorical split
    numeric_features = ['sleep_duration','exercise_duration','screen_time_before_bed','stress_level','bedtime_minutes','wakeup_minutes']
    categorical_features = ['caffeine_intake','mood','sleep_interruptions']

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # label encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc)

    candidates = {
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=random_state),
        'DecisionTree': DecisionTreeClassifier(random_state=random_state),
        'SVC': SVC(probability=True, random_state=random_state)
    }

    best = {'name': None, 'model': None, 'f1': -1}
    results = {}
    for name, clf in candidates.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('clf', clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        f1 = f1_score(y_test, preds, average='macro')
        acc = accuracy_score(y_test, preds)
        results[name] = {'f1_macro': f1, 'accuracy': acc}
        print(f"[{name}] f1_macro={f1:.4f} acc={acc:.4f}")
        if f1 > best['f1']:
            best = {'name': name, 'model': pipe, 'f1': f1}

    print("Best model:", best['name'], "f1:", best['f1'])
    # Evaluate best model final
    y_pred = best['model'].predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # save both model pipeline and label encoder
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    joblib.dump({'pipeline': best['model'], 'label_encoder': le}, out_path)
    print(f"Saved best model to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/sleep_data.csv')
    parser.add_argument('--out', default='models/sleep_model_v1.joblib')
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print("Data not found. Generating synthetic data...")
        # generate data using generate_data script
        import subprocess, sys
        subprocess.check_call([sys.executable, 'src/generate_data.py', '--out', args.data, '--n', '2000', '--force'])
    train(args.data, args.out)
