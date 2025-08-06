import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")
st.title("🔍 Detección de Rollos Apilados (versión mejorada)")

uploaded_file = st.file_uploader("📸 Sube la foto de los rollos", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    output = img_np.copy()

    st.sidebar.header("🎛️ Ajustes de detección")

    # Sliders para ajuste fino
    param2 = st.sidebar.slider("param2 (sensibilidad)", 10, 100, 30)
    minDist = st.sidebar.slider("Distancia mínima entre círculos", 10, 100, 40)
    minRadius = st.sidebar.slider("Radio mínimo", 5, 100, 30)
    maxRadius = st.sidebar.slider("Radio máximo", 5, 150, 70)

    # Conversión a gris y mejora de bordes
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 3
    )

    # Detección de círculos
    circles = cv2.HoughCircles(
        thresh, cv2.HOUGH_GRADIENT, dp=1,
        minDist=minDist,
        param1=100, param2=param2,
        minRadius=minRadius, maxRadius=maxRadius
    )

    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i, circle in enumerate(circles[0, :]):
            x, y, r = circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.putText(output, str(i+1), (x - 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            count += 1

    st.image(output, caption=f"🟢 Rollos detectados: {count}", use_container_width=True)
    st.success(f"Total detectado: {count} rollos")
