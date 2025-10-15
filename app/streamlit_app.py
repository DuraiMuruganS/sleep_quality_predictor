import streamlit as st
from src.predict import predict

st.set_page_config(page_title="Sleep Quality Predictor", layout="centered")

st.title("Sleep Quality Predictor")
st.write("Enter your daily details and get a sleep quality prediction + tips.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        bedtime = st.time_input("Bedtime", value=__import__('datetime').time(23, 0))
        wakeup_time = st.time_input("Wake-up time", value=__import__('datetime').time(7, 0))
        sleep_duration = st.number_input("Sleep duration (hours)", min_value=0.0, max_value=24.0, value=7.0, step=0.25)
        caffeine = st.selectbox("Caffeine intake", ['None','Low','Moderate','High'])
    with col2:
        exercise = st.number_input("Exercise duration (minutes)", min_value=0, max_value=300, value=30)
        screen_time = st.number_input("Screen time before bed (minutes)", min_value=0, max_value=600, value=45)
        stress_level = st.slider("Stress level (0–10)", 0, 10, 4)
        mood = st.selectbox("Mood before sleep", ['Happy','Neutral','Sad','Anxious'])
        interruptions = st.radio("Sleep interruptions during night?", ['No','Yes'])

    submitted = st.form_submit_button("Predict Sleep Quality")

if submitted:
    input_dict = {
        'bedtime': bedtime.strftime("%H:%M"),
        'wakeup_time': wakeup_time.strftime("%H:%M"),
        'sleep_duration': float(sleep_duration),
        'caffeine_intake': caffeine,
        'exercise_duration': int(exercise),
        'screen_time_before_bed': int(screen_time),
        'stress_level': int(stress_level),
        'mood': mood,
        'sleep_interruptions': interruptions
    }
    try:
        result = predict(input_dict)
        st.subheader("Prediction")
        st.markdown(f"**Sleep quality:** `{result['label']}`")
        if result['confidence'] is not None:
            st.markdown(f"**Confidence:** {result['confidence']*100:.1f}%")
        st.subheader("Suggestions to improve sleep")
        for t in result['tips']:
            st.write("- " + t)
    except Exception as e:
        st.error(f"Prediction error: {e}")
