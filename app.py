"""
DeepShield — app.py
Streamlit web app for AI deepfake detection.
Run with: streamlit run app.py
"""

import os
import sys
import streamlit as st
import numpy as np

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="DeepShield · Deepfake Detector",
    page_icon="🛡️",
    layout="centered",
)

# ── Resolve imports regardless of working directory ───────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.preprocess import load_and_preprocess, decode_prediction

MODEL_PATH = os.path.join(ROOT, "model", "deepfake_model.h5")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ── */
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

    /* ── Header ── */
    .ds-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .ds-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0;
        background: linear-gradient(135deg, #4FC3F7 0%, #1565C0 60%, #7C4DFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .ds-header p { color: #90A4AE; font-size: 1.05rem; margin-top: 0.25rem; }

    /* ── Result cards ── */
    .result-card {
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin-top: 1.2rem;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 1px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.18);
    }
    .result-fake {
        background: linear-gradient(135deg, #FF5252, #B71C1C);
        color: #fff;
    }
    .result-real {
        background: linear-gradient(135deg, #69F0AE, #00695C);
        color: #fff;
    }

    /* ── Confidence text ── */
    .conf-label {
        text-align: center;
        font-size: 1rem;
        color: #B0BEC5;
        margin-top: 0.4rem;
        margin-bottom: 0.2rem;
    }

    /* ── Info box ── */
    .info-box {
        background: #1E2A3A;
        border-left: 4px solid #4FC3F7;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        color: #CFD8DC;
        font-size: 0.9rem;
        margin-top: 1.5rem;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #546E7A;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="ds-header">
        <h1>🛡️ DeepShield</h1>
        <p>AI-powered deepfake image detection · Upload a face image to analyse it</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading DeepShield model …")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

if model is None:
    st.warning(
        "⚠️  No trained model found at `model/deepfake_model.h5`.\n\n"
        "Run `python train.py` first to train and save the model."
    )
    st.stop()

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a face image (JPG / PNG / WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    help="The model analyses the image locally — nothing is sent to any server.",
)

if uploaded is not None:
    col_img, col_gap, col_res = st.columns([1, 0.08, 1.2])

    with col_img:
        st.image(uploaded, caption="Uploaded image", use_container_width=True)

    with col_res:
        st.markdown("### 🔍 Analysis")

        with st.spinner("Analysing …"):
            try:
                img_array = load_and_preprocess(uploaded)
                raw_score = float(model.predict(img_array, verbose=0)[0][0])
                label, confidence = decode_prediction(raw_score)
            except ValueError as e:
                st.error(f"Could not process image: {e}")
                st.stop()

        # ── Result card ───────────────────────────────────────────────────────
        card_class = "result-fake" if label == "FAKE" else "result-real"
        icon        = "🚨" if label == "FAKE" else "✅"
        st.markdown(
            f'<div class="result-card {card_class}">{icon} {label}</div>',
            unsafe_allow_html=True,
        )

        # ── Confidence bar ────────────────────────────────────────────────────
        st.markdown(
            f'<p class="conf-label">Confidence: <strong>{confidence:.1f}%</strong></p>',
            unsafe_allow_html=True,
        )
        st.progress(int(confidence))

        # ── Detailed metrics ──────────────────────────────────────────────────
        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Raw score", f"{raw_score:.4f}")
        m2.metric("Decision threshold", "0.50")

        fake_pct = round(raw_score * 100, 1)
        real_pct = round((1 - raw_score) * 100, 1)
        st.markdown("**Probability breakdown**")
        st.markdown(f"🔴 Fake probability: `{fake_pct}%`")
        st.markdown(f"🟢 Real probability: `{real_pct}%`")

    # ── Info box ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="info-box">
            <strong>How it works:</strong> DeepShield uses a fine-tuned MobileNetV2
            convolutional neural network trained on real and AI-generated face
            datasets. The model outputs a probability score — values closer to 1.0
            indicate a likely deepfake, while values closer to 0.0 suggest a
            genuine image.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    # Placeholder state
    st.info("👆 Upload an image above to get started.")
    st.markdown(
        """
        **What DeepShield can detect:**
        - GAN-generated faces (StyleGAN, DALL-E, Midjourney)
        - Face-swap deepfakes (DeepFaceLab, FaceSwap)
        - Diffusion-model synthetic portraits

        **Best results with:**
        - Clear, front-facing portrait images
        - Good lighting, minimal occlusion
        - JPG / PNG under 10 MB
        """
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">DeepShield · Built with TensorFlow & Streamlit · '
    'For educational use only</div>',
    unsafe_allow_html=True,
)
