import io
import os
import csv
import random
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

APP_TITLE = "Smart Human Emotion Detection"
DB_PATH = os.environ.get("EMOTION_DB_PATH", "emotion_results.db")
VOICE_MODEL_NAME = "superb/wav2vec2-base-superb-er"

EMOTIONS = {
    "happy": ("😊", "#79D70F", "Keep smiling, and enjoy the little things."),
    "sad": ("😔", "#55B9FF", "It is okay to have difficult moments. Be kind to yourself."),
    "angry": ("😠", "#FF665A", "Pause for a moment and give yourself time."),
    "fear": ("😨", "#FF9A3D", "Focus on one manageable step at a time."),
    "surprise": ("😲", "#FFD34D", "Unexpected moments can be a chance to learn."),
    "disgust": ("🤢", "#55DDAA", "Take a moment to reset."),
    "neutral": ("😐", "#D8DDE8", "Stay focused and move at your own pace."),
}

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")

# High-contrast styling for desktop and mobile. Sidebar and all labels use dark backgrounds.
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp {
  background: linear-gradient(135deg, #07152f 0%, #101b4b 100%);
  color: #f7f9ff;
}
[data-testid="stSidebar"] {
  background: #07152f !important;
  border-right: 1px solid #253866;
}
[data-testid="stSidebar"] * {
  color: #f7f9ff !important;
  opacity: 1 !important;
}
h1, h2, h3, h4 { color: #59e3ff !important; }
p, label, li, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {
  color: #f7f9ff;
}
[data-testid="stRadio"] label, [data-testid="stCheckbox"] label {
  color: #ffffff !important;
  opacity: 1 !important;
  font-weight: 600 !important;
}
[data-testid="stTabs"] button {
  color: #ffffff !important;
  font-weight: 700 !important;
}
[data-testid="stFileUploader"] section {
  background: #ffffff !important;
  border: 2px dashed #12bde0 !important;
  border-radius: 12px !important;
}
[data-testid="stFileUploader"] section * {
  color: #152443 !important;
  opacity: 1 !important;
}
[data-testid="stCameraInput"] button { font-weight: 700 !important; }
button[kind="primary"] { font-weight: 700 !important; }
div[data-testid="stMetric"] {
  background: #111f4b;
  border: 1px solid #263c70;
  padding: 14px;
  border-radius: 12px;
}
div[data-testid="stAlert"] { border-radius: 12px; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
.small-note { color: #cbd7f4; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


def connect_db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    with connect_db() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS emotion_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                detected_at TEXT NOT NULL
            )
        """)


def save_result(emotion, confidence, source):
    init_db()
    with connect_db() as con:
        con.execute(
            "INSERT INTO emotion_results(emotion, confidence, source, detected_at) VALUES (?, ?, ?, ?)",
            (str(emotion).lower(), float(confidence), source,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )


def get_history():
    init_db()
    with connect_db() as con:
        return con.execute(
            "SELECT id, emotion, confidence, source, detected_at FROM emotion_results ORDER BY id DESC"
        ).fetchall()


@st.cache_resource(show_spinner="Loading face emotion model. First use may take a while...")
def face_model():
    from fer import FER
    return FER(mtcnn=False)


@st.cache_resource(show_spinner="Loading voice emotion model. First use may download model files...")
def voice_model():
    from transformers import pipeline
    return pipeline("audio-classification", model=VOICE_MODEL_NAME)


def show_result(emotion, confidence, source):
    emotion = str(emotion).lower()
    emoji, color, message = EMOTIONS.get(
        emotion, ("🙂", "#59e3ff", "Prediction completed.")
    )
    st.markdown(
        f"""
        <div style="background:#101f49;border:2px solid {color};border-radius:18px;
                    padding:22px;text-align:center;margin:12px 0">
          <div style="font-size:52px">{emoji}</div>
          <h2 style="color:{color}!important;margin:5px">{emotion.upper()}</h2>
          <h3>Confidence estimate: {float(confidence):.1f}%</h3>
          <p>{message}</p>
          <p style="font-size:12px;color:#cbd7f4">
            AI predictions are estimates, not a diagnosis or proof of someone's true feelings.
          </p>
        </div>
        """, unsafe_allow_html=True
    )


def detect_face(image_bytes, source, original_name="photo"):
    try:
        import cv2
        original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Keep a copy of the original and pass the correct BGR image to FER.
        rgb = np.array(original)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        results = face_model().detect_emotions(bgr)

        st.image(original, caption=f"Original {source.lower()} photo", use_container_width=True)
        st.download_button(
            "Download original photo",
            data=image_bytes,
            file_name=f"original_{original_name.rsplit('.', 1)[0]}.jpg",
            mime="image/jpeg",
            key=f"download_original_{source}_{len(image_bytes)}"
        )

        if not results:
            st.warning("Face detect aagala. Face clear-ah, front-facing-ah, nalla light-la irukkura photo try pannunga.")
            st.info("Face romba small-ah irundha, mask/hand cover pannirundha, or image blur-ah irundha detect aagama irukkalaam.")
            return

        largest = max(results, key=lambda r: max(0, r["box"][2]) * max(0, r["box"][3]))
        scores = largest.get("emotions", {})
        if not scores:
            st.warning("Face region kandupidichachu, aana emotion score varala.")
            return

        emotion = max(scores, key=scores.get)
        confidence = float(scores[emotion]) * 100.0
        x, y, w, h = [int(v) for v in largest["box"]]
        marked = original.copy()
        draw = ImageDraw.Draw(marked)
        x = max(0, x); y = max(0, y)
        draw.rectangle((x, y, x + max(1, w), y + max(1, h)), outline=(0, 230, 140), width=5)
        draw.text((x, max(0, y - 24)), f"{emotion.upper()} {confidence:.1f}%", fill=(0, 90, 255))

        st.image(marked, caption="Face-detected photo", use_container_width=True)
        output = io.BytesIO()
        marked.save(output, format="JPEG", quality=92)
        st.download_button(
            "Download face-detected photo",
            data=output.getvalue(),
            file_name=f"face_detected_{original_name.rsplit('.', 1)[0]}.jpg",
            mime="image/jpeg",
            key=f"download_detected_{source}_{len(image_bytes)}"
        )
        save_result(emotion, confidence, source)
        show_result(emotion, confidence, source)

    except Exception as e:
        st.error(f"Face analysis failed: {type(e).__name__}: {e}")
        st.warning("If this is a package error, open Manage app → Logs and check the first import/install error. The first model load can take several minutes.")


def detect_voice(audio_bytes, source, filename="recording.wav"):
    try:
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) == 0:
            st.warning("Audio file empty-ah irukku. Vera recording try pannunga.")
            return
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        preds = voice_model()(
            {"array": np.asarray(audio, dtype=np.float32), "sampling_rate": int(sr)},
            top_k=8
        )
        if not preds:
            st.warning("Voice prediction varala. Clear-aana recording try pannunga.")
            return
        mapping = {
            "ang": "angry", "anger": "angry", "angry": "angry",
            "hap": "happy", "happiness": "happy", "happy": "happy",
            "sadness": "sad", "sad": "sad", "neu": "neutral", "calm": "neutral",
            "neutral": "neutral", "fea": "fear", "fear": "fear",
            "dis": "disgust", "disgust": "disgust", "sur": "surprise",
            "surprised": "surprise", "surprise": "surprise"
        }
        grouped = {}
        for item in preds:
            raw = str(item.get("label", "neutral")).strip().lower()
            label = mapping.get(raw, raw)
            grouped[label] = grouped.get(label, 0.0) + float(item.get("score", 0))
        emotion = max(grouped, key=grouped.get)
        confidence = min(100.0, grouped[emotion] * 100)
        st.audio(audio_bytes)
        save_result(emotion, confidence, source)
        show_result(emotion, confidence, source)
    except Exception as e:
        st.error(f"Audio analysis failed: {type(e).__name__}: {e}")
        st.info("Use a clear WAV/FLAC/OGG/MP3 recording. First voice-model load needs time and internet access.")


init_db()

with st.sidebar:
    st.title("🧠 Emotion Space")
    st.caption("Smart Human Emotion Detection")
    page = st.radio(
        "MENU — choose a page",
        [
            "Home", "Webcam / Photo", "Voice Emotion", "Face + Voice Compare",
            "Emotion Companion", "Emotion Challenge", "Photo / Audio Guide",
            "Dashboard & Reports"
        ],
        key="main_navigation"
    )
    st.markdown("---")
    st.caption("AI predictions are estimates, not a diagnosis.")

if page == "Home":
    st.title("🧠 SMART HUMAN EMOTION DETECTION")
    st.subheader("AI-powered face and voice emotion analysis")
    st.write("Phone-friendly website: capture a face photo, upload an image, record/upload audio, compare results, and view saved history.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 📷 Face analysis")
        st.write("Take a camera snapshot or upload a JPG/PNG photo.")
    with c2:
        st.markdown("### 🎙️ Voice analysis")
        st.write("Record speech in the browser or upload an audio file.")
    with c3:
        st.markdown("### 📊 Reports")
        st.write("View emotion history and download a CSV report.")
    st.info("Choose a page from the dark-blue menu on the left. On mobile, tap the » menu icon to open it.")
    st.caption("For better face detection, use a front-facing, well-lit photo with the whole face visible.")

elif page == "Webcam / Photo":
    st.title("📷 Webcam / Photo Emotion Scan")
    tab1, tab2 = st.tabs(["📸 Take webcam photo", "🖼️ Upload photo"])
    with tab1:
        st.info("Phone-la camera permission Allow pannunga. Idhu snapshot capture pannum; continuous live video illa.")
        cam = st.camera_input("Open camera and capture a clear face photo")
        if cam:
            st.image(cam, caption="Captured webcam photo", use_container_width=True)
            if st.button("Analyse webcam photo", key="cam_analyse", type="primary", use_container_width=True):
                detect_face(cam.getvalue(), "WEBCAM", "webcam_photo.jpg")
    with tab2:
        st.info("Gallery-la irundhu clear JPG, PNG, WEBP or BMP face photo select pannunga.")
        photo = st.file_uploader("Choose a face photo", type=["jpg", "jpeg", "png", "bmp", "webp"], key="face_photo_upload")
        if photo:
            try:
                preview = Image.open(photo).convert("RGB")
                st.image(preview, caption=f"Selected photo: {photo.name}", use_container_width=True)
                if st.button("Analyse uploaded photo", key="photo_analyse", type="primary", use_container_width=True):
                    detect_face(photo.getvalue(), "PHOTO_UPLOAD", photo.name)
            except Exception as e:
                st.error(f"Could not open this image file: {e}")

elif page == "Voice Emotion":
    st.title("🎙️ Voice Emotion Detection")
    st.write("Speak naturally in a short recording. Noise and language can affect predictions.")
    recorded = st.audio_input("Record your voice")
    if recorded and st.button("Analyse recorded voice", type="primary", use_container_width=True):
        detect_voice(recorded.getvalue(), "VOICE_RECORD", "recording.wav")
    st.divider()
    st.subheader("Upload audio file")
    audio = st.file_uploader("Choose audio", type=["wav", "flac", "ogg", "mp3"], key="audio_upload")
    if audio:
        st.audio(audio)
        if st.button("Analyse uploaded audio", type="primary", use_container_width=True):
            detect_voice(audio.getvalue(), "VOICE_UPLOAD", audio.name)

elif page == "Face + Voice Compare":
    st.title("🔎 Face + Voice Comparison")
    history = get_history()
    face_row = next((r for r in history if r[3] in ("WEBCAM", "PHOTO_UPLOAD")), None)
    voice_row = next((r for r in history if r[3] in ("VOICE_RECORD", "VOICE_UPLOAD")), None)
    a, b = st.columns(2)
    with a:
        st.subheader("Latest face/photo result")
        if face_row:
            show_result(face_row[1], face_row[2], face_row[3])
        else:
            st.info("Run a face/photo scan first.")
    with b:
        st.subheader("Latest voice result")
        if voice_row:
            show_result(voice_row[1], voice_row[2], voice_row[3])
        else:
            st.info("Run a voice scan first.")
    if face_row and voice_row:
        st.write("Predictions match." if face_row[1] == voice_row[1] else "Predictions differ; the models use different signals.")
    st.caption("Comparison cannot confirm someone's true feelings.")

elif page == "Emotion Companion":
    st.title("🤖 AI Emotion Companion")
    history = get_history()
    if history:
        latest = history[0]
        emotion = latest[1].lower()
        emoji, color, message = EMOTIONS.get(emotion, ("🙂", "#59e3ff", "Take a moment for yourself."))
        st.markdown(f"## {emoji} {emotion.upper()}")
        st.success(message)
        st.caption(f"Based on the latest saved estimate: {latest[4]}")
    else:
        st.info("Run a face or voice scan first to get a supportive message.")
    st.write("This provides general supportive text only, not a mental-health diagnosis or professional advice.")

elif page == "Emotion Challenge":
    st.title("🎮 Emotion Challenge")
    questions = [
        ("A relaxed face has no clear positive or negative cues. Best cautious label?", "neutral"),
        ("A person recoils from a strong unpleasant smell. Best match?", "disgust"),
        ("A sudden unexpected event causes widened eyes. Best immediate guess?", "surprise"),
        ("A person looks uneasy before an unfamiliar presentation. Possible emotion?", "fear"),
        ("Someone loses a valued item and looks downcast. Best match?", "sad"),
        ("Two emotion models disagree. What is the safest choice?", "neutral"),
        ("A person smiles while their voice sounds worried. What can you conclude?", "neutral"),
        ("A person becomes tense after repeated interruptions. Possible emotion?", "angry"),
        ("A person receives pleasant news and smiles broadly. Best match?", "happy"),
        ("Can one facial cue prove what someone truly feels?", "neutral"),
        ("A person reacts strongly to something unpleasant. Best match?", "disgust"),
        ("The evidence is unclear. What should you avoid?", "neutral"),
    ]
    if "quiz_qs" not in st.session_state:
        st.session_state.quiz_qs = random.sample(questions, 8)
        st.session_state.quiz_index = 0
        st.session_state.quiz_score = 0
        st.session_state.quiz_done = False
    if st.session_state.quiz_done:
        st.metric("Final score", f"{st.session_state.quiz_score} / {len(st.session_state.quiz_qs)}")
        if st.button("Play again", type="primary"):
            for key in ("quiz_qs", "quiz_index", "quiz_score", "quiz_done"):
                st.session_state.pop(key, None)
            st.rerun()
    else:
        idx = st.session_state.quiz_index
        question, correct = st.session_state.quiz_qs[idx]
        st.progress(idx / len(st.session_state.quiz_qs))
        st.write(f"Question {idx + 1} / {len(st.session_state.quiz_qs)}")
        st.subheader(question)
        answer = st.radio("Choose an answer", ["happy", "sad", "angry", "fear", "surprise", "neutral", "disgust"], key=f"answer_{idx}")
        if st.button("Submit answer", type="primary"):
            if answer == correct:
                st.session_state.quiz_score += 1
                st.success("Correct!")
            else:
                st.info(f"Best cautious answer: {correct}")
            st.session_state.quiz_index += 1
            if st.session_state.quiz_index >= len(st.session_state.quiz_qs):
                st.session_state.quiz_done = True
            st.rerun()

elif page == "Photo / Audio Guide":
    st.title("📘 Photo / Audio Guide")
    st.subheader("For face photos")
    st.markdown("- Face should face the camera.\n- Keep eyes, nose and mouth visible.\n- Use bright, even lighting.\n- Avoid blurry, very dark, or distant photos.\n- JPG, JPEG, PNG, BMP and WEBP are supported.")
    st.subheader("For voice recordings")
    st.markdown("- Record a short, clear voice clip.\n- Reduce background noise.\n- WAV, FLAC, OGG and MP3 are supported.\n- Voice emotion models can be inaccurate, especially for languages or speech styles not represented in training data.")

elif page == "Dashboard & Reports":
    st.title("📊 Dashboard & Reports")
    history = get_history()
    if not history:
        st.info("No saved results yet. Run a face or voice analysis first.")
    else:
        df = pd.DataFrame(history, columns=["ID", "Emotion", "Confidence (%)", "Source", "Date & Time"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Total scans", len(df))
        c2.metric("Most frequent emotion", str(df["Emotion"].mode().iloc[0]).title())
        c3.metric("Average confidence", f"{df['Confidence (%)'].mean():.1f}%")
        st.subheader("Emotion counts")
        st.bar_chart(df["Emotion"].value_counts())
        st.subheader("Scan history")
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv_data = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download history as CSV", csv_data, "emotion_history.csv", "text/csv", use_container_width=True)
        if st.button("Clear all history", type="secondary"):
            with connect_db() as con:
                con.execute("DELETE FROM emotion_results")
            st.success("History cleared.")
            st.rerun()
