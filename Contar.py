import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(layout="centered")
st.title("🔵 Contador de Rollos (Detección Frontal)")
st.write("Este sistema detecta rollos vistos desde el frente usando detección de círculos.")

# Subir imagen
archivo = st.file_uploader("Sube una imagen de rollos", type=["jpg", "png", "jpeg"])
if archivo:
    # Leer imagen y convertir a formato OpenCV
    imagen_pil = Image.open(archivo).convert("RGB")
    imagen_np = np.array(imagen_pil)
    img = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2BGR)

    # Mostrar imagen original
    st.image(imagen_pil, caption="Imagen Original", use_container_width=True)

    # Preprocesamiento
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # Detectar círculos
    circulos = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=100,
        param2=20,
        minRadius=50,
        maxRadius=80
    )

    # Dibujar y contar
    total = 0
    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        for c in circulos[0, :]:
            cv2.circle(img, (c[0], c[1]), c[2], (0, 255, 0), 2)
            cv2.circle(img, (c[0], c[1]), 2, (0, 0, 255), 3)
        total = len(circulos[0])

    # Mostrar resultados
    st.success(f"🧮 Rollos detectados: {total}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Detección de Círculos", use_container_width=True)
