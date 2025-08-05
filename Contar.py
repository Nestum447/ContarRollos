import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Contador de Rollos", layout="wide")
st.title("🌀 Contador de Rollos (Vista Frontal con Sliders)")
st.write("Ajusta los parámetros para mejorar la detección de rollos en tiempo real.")

# 📷 Subida de imagen
archivo = st.file_uploader("Sube una imagen frontal de los rollos", type=["jpg", "jpeg", "png"])

# 🎚️ Sliders para parámetros HoughCircles
st.sidebar.title("Ajustes de detección")
dp = st.sidebar.slider("Resolución inversa (dp)", 1.0, 2.0, 1.2, 0.1)
min_dist = st.sidebar.slider("Distancia mínima entre círculos", 10, 200, 50, 5)
param1 = st.sidebar.slider("Param1 (Canny high threshold)", 10, 200, 100, 5)
param2 = st.sidebar.slider("Param2 (acumulador - sensibilidad)", 5, 100, 30, 1)
min_radius = st.sidebar.slider("Radio mínimo", 5, 100, 20, 1)
max_radius = st.sidebar.slider("Radio máximo", 10, 200, 80, 1)

# Procesamiento si se sube una imagen
if archivo:
    imagen_pil = Image.open(archivo).convert("RGB")
    img = np.array(imagen_pil)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Preprocesamiento
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # HoughCircles con parámetros interactivos
    circulos = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    total = 0
    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        for c in circulos[0, :]:
            cv2.circle(img_bgr, (c[0], c[1]), c[2], (0, 255, 0), 2)
            cv2.circle(img_bgr, (c[0], c[1]), 2, (0, 0, 255), 3)
        total = len(circulos[0])

    # Mostrar resultados
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption=f"🎯 Rollos detectados: {total}", use_container_width=True)
