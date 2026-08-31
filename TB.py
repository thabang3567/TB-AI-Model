import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import librosa
import matplotlib.pyplot as plt
import noisereduce as nr
import cv2
import tempfile
import os
from datetime import datetime
from fpdf import FPDF

# ---------------------------------------------------------
# 1. Page Configuration & Session State
# ---------------------------------------------------------
st.set_page_config(page_title="AI Respiratory Screening", page_icon="🫁", layout="wide")

if 'upload_result' not in st.session_state: st.session_state['upload_result'] = None
if 'audio_result' not in st.session_state: st.session_state['audio_result'] = None
if 'last_uploaded_file' not in st.session_state: st.session_state['last_uploaded_file'] = None
if 'last_audio_input' not in st.session_state: st.session_state['last_audio_input'] = None

# ---------------------------------------------------------
# 2. Comprehensive Patient Survey (Sidebar)
# ---------------------------------------------------------
st.sidebar.header("📋 Patient Clinical Survey")
st.sidebar.write("Combine AI acoustics with clinical metadata for accurate triage.")

# Set defaults to neutral/unknown so they do not trigger logic unless changed
age = st.sidebar.number_input("Patient Age", min_value=1, max_value=120, value=30)
smoking = st.sidebar.selectbox("Smoking History",
                               ["Unknown/Prefer not to say", "Never Smoked", "Past Smoker", "Current Smoker"], index=0)
tb_contact = st.sidebar.radio("Recent contact with a known TB patient?", ["Unknown", "No", "Yes"], index=0)
covid_exp = st.sidebar.radio("Recent COVID-19 exposure?", ["Unknown", "No", "Yes"], index=0)
cough_duration = st.sidebar.selectbox("How long have you been coughing?",
                                      ["No Cough / Unknown", "Less than 1 week", "1-2 weeks", "More than 2 weeks"],
                                      index=0)

st.sidebar.markdown("---")
st.sidebar.write("**Current Symptoms (Leave unchecked if none):**")

symp_blood = st.sidebar.checkbox("Coughing up blood (Hemoptysis)")
symp_sweats = st.sidebar.checkbox("Severe Night Sweats")
symp_weight = st.sidebar.checkbox("Unexplained Weight Loss")
symp_fever = st.sidebar.checkbox("Fever")
symp_breath = st.sidebar.checkbox("Shortness of breath")
symp_chest = st.sidebar.checkbox("Chest pain")
symp_fatigue = st.sidebar.checkbox("Severe fatigue")

patient_data = {
    "Age": age, "Smoking": smoking, "TB_Contact": tb_contact, "COVID_Exp": covid_exp,
    "Duration": cough_duration,
    "Symptoms": {
        "Blood": symp_blood, "Sweats": symp_sweats, "WeightLoss": symp_weight,
        "Fever": symp_fever, "Breath": symp_breath, "Chest": symp_chest, "Fatigue": symp_fatigue
    }
}

# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.title("🫁 3-Class Respiratory Screening System")
st.write("A multimodal clinical triage tool focusing on **Healthy, TB, and COVID-19**.")
st.info(
    "💡 **Clinical Concept:** This system fuses Deep Learning Audio extraction with WHO-standard symptom surveys to recommend the safest clinical next step.")


# ---------------------------------------------------------
# 3. Load Model
# ---------------------------------------------------------
@st.cache_resource
def load_trained_model():
    try:
        return tf.keras.models.load_model("tb_cough_model.keras")
    except Exception as e:
        st.error(
            f"Model not found or error loading model: {str(e)}\nPlease ensure 'tb_cough_model.keras' is in the directory.")
        return None


model = load_trained_model()


# ---------------------------------------------------------
# 4. Audio Preprocessing & Quality Meter
# ---------------------------------------------------------
def audio_bytes_to_spectrogram_image(audio_bytes):
    audio_buffer = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_buffer, sr=22050)

    if len(y) == 0 or np.max(np.abs(y)) < 0.05: return None, "No cough sound detected.", 0

    y_trimmed, index = librosa.effects.trim(y, top_db=20)
    noise_part = np.concatenate([y[:index[0]], y[index[1]:]])

    if len(noise_part) > (sr * 0.1):
        p_noise = np.mean(noise_part ** 2)
    else:
        p_noise = np.mean(np.sort(np.abs(y))[:max(1, len(y) // 10)] ** 2)

    p_signal, p_noise = np.mean(y_trimmed ** 2), max(p_noise, 1e-10)
    snr_db = 10 * np.log10(p_signal / p_noise)
    quality_score = min(max(int(((snr_db - 5) / 25) * 100), 0), 100)

    if (len(y_trimmed) / sr) < 0.15: return None, "Sound was too short to be a valid cough.", quality_score

    y_denoised = nr.reduce_noise(y=y_trimmed, sr=sr)
    y_fixed = librosa.util.fix_length(y_denoised, size=3 * sr)
    mel_spec_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y_fixed, sr=sr, n_mels=128), ref=np.max)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis("off")
    ax.imshow(mel_spec_db, aspect="auto", origin="lower", cmap="viridis")
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf), None, quality_score


# ---------------------------------------------------------
# 5. Grad-CAM Explainability (Fixed Class Targeting)
# ---------------------------------------------------------
def generate_gradcam_overlay(img_array, model, raw_image):
    try:
        conv_layer_idx = next((idx for idx, layer in reversed(list(enumerate(model.layers))) if
                               isinstance(layer, tf.keras.layers.Conv2D)), None)
        if conv_layer_idx is None: return raw_image.resize((150, 150))

        img_tensor = tf.cast(img_array, tf.float32)
        with tf.GradientTape() as tape:
            x = img_tensor
            for layer in model.layers[:conv_layer_idx + 1]:
                x = layer(x)
            conv_outputs = x
            tape.watch(conv_outputs)

            y = conv_outputs
            for layer in model.layers[conv_layer_idx + 1:]:
                y = layer(y)

            predicted_class = tf.argmax(y[0])
            class_channel = y[:, predicted_class]

        grads = tape.gradient(class_channel, conv_outputs)
        if grads is None: return raw_image.resize((150, 150))

        heatmap = tf.squeeze(conv_outputs[0] @ tf.reduce_mean(grads, axis=(0, 1, 2))[..., tf.newaxis])
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap.numpy(), (150, 150))), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(cv2.cvtColor(np.array(raw_image.resize((150, 150))), cv2.COLOR_RGB2BGR), 0.6, heatmap,
                                  0.4, 0)
        return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")
        return raw_image.resize((150, 150))


# ---------------------------------------------------------
# 6. Dynamic 3-Class Multimodal Risk Engine
# ---------------------------------------------------------
def calculate_comprehensive_risk(image_pil, p_data):
    """Fuses 3-Class AI audio data with explicitly provided Clinical Metadata."""
    img_array = np.expand_dims(np.array(image_pil.convert("RGB").resize((150, 150))), axis=0)
    raw_preds = model.predict(img_array)[0]

    # Map the 3 classes based on alphabetical folders: 0_Healthy, 1_Tuberculosis, 2_COVID-19
    if len(raw_preds) == 3:
        base_healthy = float(raw_preds[0])
        base_tb = float(raw_preds[1])
        base_covid = float(raw_preds[2])
    else:
        # Fallback
        base_tb = float(raw_preds[0])
        base_healthy = 1.0 - base_tb
        base_covid = 0.05

    heatmap_img = generate_gradcam_overlay(img_array, model, image_pil)
    s = p_data["Symptoms"]

    # ---------------------------------------------------------
    # NEW: FALSE-POSITIVE GUARDRAIL (MICROPHONE BIAS CORRECTION)
    # ---------------------------------------------------------
    # Check if the user is completely asymptomatic
    has_no_symptoms = not any(s.values())
    has_no_exposure = (p_data["COVID_Exp"] != "Yes") and (p_data["TB_Contact"] != "Yes")

    if has_no_symptoms and has_no_exposure:
        # Aggressively boost the Healthy baseline and slash disease confidence
        # to prevent out-of-distribution microphone noise from causing a false alarm.
        base_healthy += 0.80
        base_covid *= 0.2
        base_tb *= 0.2

        # ---------------------------------------------------------
    # Existing Clinical Rule Overrides (Escalations)
    # ---------------------------------------------------------
    if p_data["Duration"] == "More than 2 weeks": base_tb += 0.20
    if s["Blood"] or s["Sweats"] or s["WeightLoss"]: base_tb += 0.25
    if p_data["TB_Contact"] == "Yes": base_tb += 0.20

    if p_data["COVID_Exp"] == "Yes": base_covid += 0.40
    if s["Fever"] and s["Fatigue"] and p_data["Duration"] == "Less than 1 week": base_covid += 0.35

    # 4. Normalize Probabilities to equal 100%
    total_score = base_tb + base_covid + base_healthy
    risks = {
        "Healthy": base_healthy / total_score,
        "Tuberculosis": base_tb / total_score,
        "COVID-19": base_covid / total_score
    }

    disease_risks = {k: v for k, v in risks.items() if k != "Healthy"}
    primary_concern = max(disease_risks, key=disease_risks.get)

    # Stricter threshold: Disease must overcome a 40% confidence barrier to trigger an alarm
    if risks[primary_concern] < 0.40:
        primary_concern = "Healthy"

    return risks, primary_concern, heatmap_img


# ---------------------------------------------------------
# 7. PDF Report Generator
# ---------------------------------------------------------
def create_medical_pdf(risks, primary_concern, p_data, heatmap, spec_img, quality):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)

    pdf.cell(200, 10, txt="3-Class Respiratory Triage Report", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Audio SNR Quality: {quality}%", ln=True,
             align='C')
    pdf.line(10, 30, 200, 30)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="1. Patient Clinical Profile:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 6, txt=f"Age: {p_data['Age']} | Smoker: {p_data['Smoking']} | Cough: {p_data['Duration']}", ln=True)

    active_symp = [k for k, v in p_data["Symptoms"].items() if v]
    symp_text = ", ".join(active_symp) if active_symp else "None reported"
    pdf.cell(200, 6, txt=f"Symptoms: {symp_text}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="2. Multimodal Risk Assessment:", ln=True)
    pdf.set_font("Arial", size=11)
    for disease in ["Healthy", "Tuberculosis", "COVID-19"]:
        pdf.cell(200, 6, txt=f"- {disease}: {risks[disease] * 100:.1f}%", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="3. Final Triage Recommendation:", ln=True)
    pdf.set_font("Arial", size=11)
    if primary_concern != "Healthy":
        pdf.set_text_color(220, 53, 69)
        pdf.cell(200, 6, txt=f"ELEVATED RISK ({primary_concern}). Further clinical diagnostic testing is required.",
                 ln=True)
    else:
        pdf.set_text_color(40, 167, 69)
        pdf.cell(200, 6, txt="LOW RISK. Acoustic patterns and symptoms indicate standard healthy baseline.", ln=True)
    pdf.set_text_color(0, 0, 0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as spec_file, \
            tempfile.NamedTemporaryFile(delete=False, suffix=".png") as heat_file:
        spec_img.save(spec_file.name)
        heatmap.save(heat_file.name)
        pdf.image(spec_file.name, 20, pdf.get_y() + 10, 70)
        pdf.image(heat_file.name, 110, pdf.get_y() + 10, 70)
    os.unlink(spec_file.name)
    os.unlink(heat_file.name)

    return pdf.output(dest='S').encode('latin-1')


# ---------------------------------------------------------
# 8. Unified UI Result Renderer
# ---------------------------------------------------------
def display_full_results(risks, primary_concern, heatmap, spec_img, quality, p_data, key_suffix):
    st.markdown("---")
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📊 3-Class Triage Probabilities")

        for disease in ["Healthy", "Tuberculosis", "COVID-19"]:
            pct = int(risks[disease] * 100)
            st.markdown(f"**{disease}**: {pct}%")
            st.progress(risks[disease])

    with col2:
        st.subheader("🏥 Triage Recommendation")

        if primary_concern != "Healthy":
            st.error(
                f"**Elevated Risk for {primary_concern}**\n\nThe combined acoustic and symptom pattern indicates an elevated risk. **Standard diagnostic testing (X-Ray / Molecular testing) is highly recommended.**")
        else:
            st.success(
                "**Low Risk Pattern**\n\nAcoustics and clinical markers appear normal. No immediate clinical action required.")

        if quality > 0:
            st.markdown(f"**Audio Recording Quality:** {quality}%")
        st.image(heatmap, caption="AI Focus Heatmap", width=200)

        pdf_bytes = create_medical_pdf(risks, primary_concern, p_data, heatmap, spec_img, quality)
        st.download_button(
            label="📄 Download 3-Class Medical Report",
            data=pdf_bytes,
            file_name=f"3Class_Screening_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            type="primary",
            key=f"dl_pdf_{key_suffix}"
        )


# ---------------------------------------------------------
# 9. Tabbed Interface
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["🎤 Record Cough Audio", "📁 Upload Cough File"])

with tab1:
    st.write("Ensure you are in a quiet room. Press record and cough 2–3 times clearly.")
    audio_input = st.audio_input("Record Cough")

    if audio_input != st.session_state['last_audio_input']:
        st.session_state['last_audio_input'] = audio_input
        st.session_state['audio_result'] = None

    if audio_input is not None:
        if st.button("Run Risk Assessment", type="primary", key="btn_audio"):
            if model is not None:
                with st.spinner("Processing audio and calculating multimodal risk..."):
                    spec_img, error_msg, quality = audio_bytes_to_spectrogram_image(audio_input.getvalue())

                    if error_msg:
                        st.error(f"⚠️ {error_msg} (Quality: {quality}%)")
                    else:
                        risks, concern, heatmap = calculate_comprehensive_risk(spec_img, patient_data)
                        st.session_state['audio_result'] = {"risks": risks, "concern": concern, "heatmap": heatmap,
                                                            "spec": spec_img, "quality": quality}

        if st.session_state['audio_result']:
            res = st.session_state['audio_result']
            display_full_results(res['risks'], res['concern'], res['heatmap'], res['spec'], res['quality'],
                                 patient_data, key_suffix="audio")

with tab2:
    st.write("Upload a pre-recorded cough spectrogram image.")
    uploaded_file = st.file_uploader("Choose a Spectrogram (.png, .jpg)...", type=["png", "jpg"])

    if uploaded_file != st.session_state['last_uploaded_file']:
        st.session_state['last_uploaded_file'] = uploaded_file
        st.session_state['upload_result'] = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Spectrogram", width=250)

        if st.button("Run Risk Assessment", type="primary", key="btn_upload"):
            if model is not None:
                with st.spinner("Analyzing image and calculating multimodal risk..."):
                    risks, concern, heatmap = calculate_comprehensive_risk(image, patient_data)
                    st.session_state['upload_result'] = {"risks": risks, "concern": concern, "heatmap": heatmap,
                                                         "spec": image, "quality": 0}

        if st.session_state['upload_result']:
            res = st.session_state['upload_result']
            display_full_results(res['risks'], res['concern'], res['heatmap'], res['spec'], res['quality'],
                                 patient_data, key_suffix="upload")