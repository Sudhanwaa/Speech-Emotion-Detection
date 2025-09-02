import streamlit as st
from st_audiorec import st_audiorec
import numpy as np
import librosa
import tensorflow as tf
import io
import soundfile as sf
from tensorflow.keras.models import load_model
from utils import Utilities
import warnings
import soundfile
import tempfile
warnings.filterwarnings("ignore")

utils_obj = Utilities()
sampling_rate = 22050

st.set_page_config(page_title="Mood Detection", page_icon=":microphone:")

st.title("🎵 Audio Mood Detection 🎵")
st.markdown("Record your audio below and let the app detect your mood!")

# Load models
models = {
    "angry": load_model(r"D:\Projects\MoodMate\paper_code\models\angry_model.h5"),
    "happy": load_model(r"D:\Projects\MoodMate\paper_code\models\happy_model.h5"),
    "sad": load_model(r"D:\Projects\MoodMate\paper_code\models\sad_model.h5"),
    "fear": load_model(r"D:\Projects\MoodMate\paper_code\models\fear_model.h5"),
    "neutral": load_model(r"D:\Projects\MoodMate\paper_code\models\neutral_model.h5"),
    "disgust": load_model(r"D:\Projects\MoodMate\paper_code\models\disgust_model.h5")
}

# Step 1: Record Audio
st.header("Step 1: Record Your Voice")
wav_audio_data = st_audiorec()
print("After recording")

if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')
    audio_data, sr = sf.read(io.BytesIO(wav_audio_data))
    print("After audio_data")
    
    # Store in session state
    st.session_state.audio_data = audio_data
    print("After session state")


# Step 2: Predict
if "audio_data" in st.session_state and st.button("🔍 Predict Mood"):
    print("IN if condition")
    try:
        
        # audio_data = st.session_state.audio_data
        # if audio_data.ndim == 2:  # stereo case
        #     audio_data = np.mean(audio_data, axis=1)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            sf.write(tmpfile.name, st.session_state.audio_data, 16000)  # 16kHz assumed, adjust if needed
            tmp_wav_path = tmpfile.name

        features = utils_obj.feature_extraction_and_concatenation(tmp_wav_path)
        print("After  Features")

        scaled_features = utils_obj.scaling_values(features).reshape(1, 5, 17)
        print("After scaled Features")
        predictions = {emotion: model.predict(scaled_features)[0][0] for emotion, model in models.items()}
        print("After predictions")
        predicted_mood = max(predictions, key=predictions.get)
        print("After final prediction")

        st.markdown(f"### 🎭 Mood Detected: **{predicted_mood}**")
        st.markdown(f'Moods Stats {predictions}')
        print("After Markdown")
        st.balloons()

    except Exception as e:
        print(e)
        st.error(f"An error occurred: {e}")
