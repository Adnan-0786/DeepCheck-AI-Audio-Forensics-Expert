import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import tempfile
from pathlib import Path

# Import your custom feature extractor
from feature_extractor import process_audio_file

# --- CUSTOM LAYER DEFINITION ---
@tf.keras.utils.register_keras_serializable()
class MaxFeatureMap1D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaxFeatureMap1D, self).__init__(**kwargs)
    def call(self, inputs):
        x0, x1 = tf.split(inputs, num_or_size_splits=2, axis=-1)
        return tf.maximum(x0, x1)

# --- GLOBAL SETTINGS ---
SEQ_LEN = 300
FEATURE_DIM = 710  # Fused LFCC + BFCC + MGD dimension

# --- UI CONFIGURATION ---
st.set_page_config(page_title="DeepCheck AI", page_icon="🎙️", layout="centered")

st.title("🎙️ DeepCheck AI")
st.subheader("Synthetic Audio & Deepfake Detection Terminal")
st.markdown("---")

# --- SIDEBAR: MODEL SELECTION ---
st.sidebar.header("⚙️ Engine Configuration")
model_choice = st.sidebar.radio(
    "Select AI Engine:",
    ("Model 2.0 (English / Global Base)", "Model 2.1h (Hindi Telecom)")
)

# Map UI choices to your actual saved model files
if "2.0" in model_choice:
    model_path = "MOdel2.0.keras"
else:
    model_path = "MOdel2.1h.keras"

# Cache the model loading
@st.cache_resource
def load_deepcheck_model(path):
    if os.path.exists(path):
        return tf.keras.models.load_model(path, custom_objects={'MaxFeatureMap1D': MaxFeatureMap1D})
    return None

model = load_deepcheck_model(model_path)

if model is None:
    st.sidebar.error(f"❌ Model file not found: {model_path}")
else:
    st.sidebar.success("✅ Neural Engine Online")

# --- MAIN UI: FILE UPLOAD OR RECORDING ---
st.write("### 📂 Target Audio Selection")
st.write("Upload a suspected audio file or record a live sample to run a forensic phase-anomaly scan.")

# Create tabs for Upload vs Record
tab1, tab2 = st.tabs(["📂 Upload Audio File", "🎤 Record Live Audio"])

audio_source = None

with tab1:
    uploaded_file = st.file_uploader(
        "Supported formats: WAV, FLAC, MP3, M4A, AAC, OGG", 
        type=['wav', 'flac', 'mp3', 'm4a', 'aac', 'ogg']
    )
    if uploaded_file is not None:
        audio_source = uploaded_file

with tab2:
    st.write("Click the microphone icon to record your voice.")
    recorded_audio = st.audio_input("Record a voice sample")
    if recorded_audio is not None:
        audio_source = recorded_audio

# Audio Player (Displays if either upload or record is used)
if audio_source is not None:
    st.write("**Audio Preview:**")
    st.audio(audio_source)

st.markdown("---")

# --- INFERENCE ENGINE ---
if audio_source is not None and model is not None:
    
    # Big, prominent scan button
    if st.button("🔍 Run Forensic Scan", type="primary", use_container_width=True):
        
        with st.spinner("Extracting features and running full-file forensic scan..."):
            
            # Determine suffix (.wav for recorded, or the actual file extension for uploads)
            suffix = Path(audio_source.name).suffix if hasattr(audio_source, 'name') else ".wav"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(audio_source.getvalue())
                tmp_path = tmp_file.name
                
            try:
                # 1. Feature Extraction
                features = process_audio_file(tmp_path)
                t_frames = features.shape[0]
                
                # 2. Chop the audio into 3-second (300 frame) chunks
                chunks = []
                for i in range(0, t_frames, SEQ_LEN):
                    chunk = features[i : i + SEQ_LEN, :]
                    if chunk.shape[0] == SEQ_LEN:
                        chunks.append(chunk)
                
                # Handle edge-case: If file is shorter than 3 seconds
                if len(chunks) == 0:
                    padded_chunk = np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)
                    padded_chunk[:t_frames, :] = features
                    chunks.append(padded_chunk)
                
                # 3. Format as Batch Tensor
                X_input = np.array(chunks, dtype=np.float32)
                
                # 4. AI Prediction (Scans all chunks simultaneously)
                chunk_predictions = model.predict(X_input, verbose=0).flatten()
                
                # 5. Clean up temp file
                os.remove(tmp_path)
                
                # 6. Aggregation Mathematics (Majority Rules / Mean)
                mean_prob = float(np.mean(chunk_predictions)) 
                
                # --- DISPLAY RESULTS ---
                st.write("### 📊 Forensic Report")
                
                # Calculate Verdict based on the Average probability across the file
                is_fake = mean_prob > 0.5
                confidence = (mean_prob if is_fake else (1 - mean_prob)) * 100
                
                # Dynamic UI based on result
                if is_fake:
                    st.error(f"**🚨 VERDICT: SYNTHETIC AUDIO DETECTED**")
                    st.metric(label="Overall File Assessment", value="FAKE (Deepfake)", delta="Phase Anomalies Dominant", delta_color="inverse")
                    st.warning("⚠️ **Note:** If this file contains music/instruments, the perfect harmonics of the instruments may trigger false positives.")
                else:
                    st.success(f"**✅ VERDICT: AUTHENTIC HUMAN VOICE**")
                    st.metric(label="Overall File Assessment", value="REAL (Human)", delta="Vocal Tract Verified", delta_color="normal")
                
                # Confidence Progress Bar
                st.write(f"**Overall Neural Confidence: {confidence:.2f}%**")
                st.progress(int(confidence))
                
                st.markdown("---")
                
                # --- TEMPORAL ANOMALY GRAPH ---
                st.write("### 📈 Temporal Anomaly Graph")
                st.write("This graph maps the deepfake probability across the audio timeline.")
                
                # Create a properly structured Pandas dataframe for Streamlit
                chart_data = pd.DataFrame({
                    "AI Fake Probability": chunk_predictions,
                    "Danger Threshold (0.5)": [0.5] * len(chunk_predictions)
                })

                # Use area chart for a more dramatic, professional look
                st.area_chart(chart_data, color=["#FF4B4B", "#FFFFFF"])
                        
            except Exception as e:
                st.error(f"Error processing audio file. Details: {e}")