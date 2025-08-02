import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Contador de rollos apilados - vista frontal")

uploaded_file = st.file_uploader("Sube una foto clara de los rollos apilados", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    output = image_np.copy()

    # Escala de grises
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Suavizado y detección de bordes
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Cerrar huecos entre contornos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 20))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # Detectar contornos
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rollos_detectados = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = h / float(w)

        # Filtro: suficientemente grande y alto
        if area > 1000 and aspect_ratio > 1.2:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rollos_detectados += 1

    st.image(output, caption="Resultado con detección", channels="RGB")

    if rollos_detectados > 0:
        st.success(f"Rollos detectados: {rollos_detectados}")
    else:
        st.warning("No se detectaron rollos. Prueba con otra foto o mejor ángulo.")
