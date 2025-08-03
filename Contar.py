import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Contador de Rollos con Visión Artificial")
st.write("Carga una imagen de rollos para contarlos automáticamente usando OpenCV + Watershed.")

# Subir imagen
archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
if archivo:
    # Leer imagen con PIL y convertir a OpenCV
    imagen_pil = Image.open(archivo).convert("RGB")
    img = np.array(imagen_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Mostrar imagen original
    st.image(imagen_pil, caption="Imagen original", use_container_width=True)

    # Preprocesamiento
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morfología
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marcadores
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [0, 0, 255]  # Marcar bordes

    # Contar objetos
    etiquetas = np.unique(markers)
    num_rollos = len(etiquetas) - 2  # quitar fondo y borde

    st.success(f"🧮 Rollos detectados: {num_rollos}")

    # Convertir a RGB para mostrar en Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Imagen procesada", use_container_width=True)
