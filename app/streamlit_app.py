# app/streamlit_app.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from src.predict import preprocess_image


st.set_page_config(
    page_title="Digit Recognizer",
    layout="wide",
    page_icon="✏️"
)


# SESSION FOR RESET

if "reset" not in st.session_state:
    st.session_state.reset = 0

# CSS 

st.markdown("""
<style>

body { background-color: #0e0e0e; }

/* Remove big empty boxes */
div[data-testid="stVerticalBlock"] > div:first-child {
    display: none !important;
}

/* Center page */
.block-container {
    padding-top: 15px !important;
    max-width: 1100px !important;
}

/* Card UI */
.card {
    background: #111;
    padding: 20px;
    border-radius: 12px;
    color: white;
    border: 1px solid #333;
}

/* Canvas Toolbar — bottom center */
.stCanvasToolbar {
    position: absolute !important;
    bottom: -40px !important;
    left: 50% !important;
    transform: translateX(-50%);
}

/* Make toolbar icons white */
.stCanvasToolbar button svg {
    fill: white !important;
    stroke: white !important;
}

/* Hide save/reset buttons */
.stCanvasToolbar button:nth-child(3),
.stCanvasToolbar button:nth-child(4) {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)


# TITLE
st.markdown("<h1 style='text-align:center; color:white;'> Handwritten Digit Recognizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#bbbbbb;'>Draw a digit and let the AI predict it.</p>", unsafe_allow_html=True)

# MODEL LOADING

@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    mp = os.path.join(base, "..", "artifacts", "mnist_cnn.h5")
    return tf.keras.models.load_model(os.path.abspath(mp))

model = load_model()

# MAIN LAYOUT

left, right = st.columns([1, 1])


# LEFT — DRAWING CANVAS

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    rowA, rowB = st.columns([4, 1])

    with rowA:
        st.subheader(" Draw your digit below:")

    with rowB:
        if st.button("🧹 Clear Canvas"):
            st.session_state.reset += 1

    canvas = st_canvas(
        fill_color="#000000",
        stroke_width=14,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key=f"canvas-{st.session_state.reset}"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# RIGHT — PREDICTION

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(" Prediction")

    if st.button("Predict"):
        if canvas.image_data is None:
            st.warning("Please draw something first.")
        else:
            img = Image.fromarray(canvas.image_data.astype("uint8"), "RGBA").convert("RGB")
            gray = ImageOps.grayscale(img)

            x = preprocess_image(gray)
            probs = model.predict(x, verbose=0)[0]
            pred = int(np.argmax(probs))

            st.success(f"### Predicted Digit: {pred}")

            st.write("#### Confidence:")
            for i, p in enumerate(probs):
                st.write(f"{i} → {p:.4f}")

            st.image(gray.resize((150, 150)), caption="Your Drawing")

    st.markdown("</div>", unsafe_allow_html=True)
