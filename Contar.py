import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Contador de Rollos", layout="centered")
st.title("📦 Contador de Rollos desde Foto")

uploaded_file = st.file_uploader("Sube una foto desde tu celular", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_np = np.array(image.convert("RGB"))

    # Preprocesamiento
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)

    # Detección de círculos (bordes de rollos)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=30, minRadius=20, maxRadius=100)

    # Mostrar resultado
    result = img_np.copy()
    count = 0
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        count = len(circles)
        for (x, y, r) in circles:
            cv2.circle(result, (x, y), r, (0, 255, 0), 3)

    st.image(result, caption=f"Total de rollos detectados: {count}", channels="RGB", use_column_width=True)
