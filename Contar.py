import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Contador de rollos - Vista frontal 📷")

uploaded_file = st.file_uploader("Sube una foto frontal del recipiente con los rollos", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    output = image_np.copy()

    # Convertir a escala de grises y detectar bordes
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Encontrar contornos
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar por tamaño y forma
    rollos_detectados = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = h / float(w)
        area = cv2.contourArea(c)

        # Reglas para considerar que es un rollo
        if area > 500 and aspect_ratio > 1.2:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rollos_detectados += 1

    # Mostrar resultados
    if rollos_detectados > 0:
        st.success(f"Rollos detectados: {rollos_detectados}")
    else:
        st.warning("No se detectaron rollos. Prueba con otra imagen o mejor iluminación.")

    st.image(output, caption="Imagen procesada", channels="RGB")
