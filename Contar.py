import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Contador de Rollos", layout="wide")

st.title("📷 Contador de Rollos con OpenCV + Streamlit")

# Imagen de entrada
uploaded_file = st.file_uploader("Sube una imagen frontal de los rollos", type=["jpg", "jpeg", "png"])

# SIDEBAR – Parámetros ajustables
st.sidebar.header("🔧 Parámetros de Detección")

dp = st.sidebar.slider("Resolución acumulador (dp)", 1.0, 2.0, 1.2, 0.1)
minDist = st.sidebar.slider("Distancia mínima entre centros", 10, 200, 50)
param1 = st.sidebar.slider("Param1 (Canny alto)", 50, 200, 100)
param2 = st.sidebar.slider("Param2 (Umbral de detección)", 10, 100, 40)
minRadius = st.sidebar.slider("Radio mínimo", 0, 100, 20)
maxRadius = st.sidebar.slider("Radio máximo", 0, 200, 80)

# Procesar imagen
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    output_img = img_np.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(output_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(output_img, (i[0], i[1]), 2, (0, 0, 255), 3)

        st.success(f"🎯 Rollos detectados: {len(circles[0])}")
    else:
        st.warning("No se detectaron círculos con los parámetros actuales.")

    st.image(output_img, caption="Resultado", use_column_width=True)

    # Tabla explicativa de parámetros
    st.markdown("### 📘 Significado de los parámetros")
    st.markdown("""
| Parámetro  | Descripción | Recomendación |
|------------|-------------|---------------|
| `dp` | Resolución del acumulador. | 1.0 a 1.5 normalmente. |
| `minDist` | Distancia mínima entre centros. | Mayor si hay rollos cerca. |
| `param1` | Umbral alto de Canny (bordes). | Usualmente 100. |
| `param2` | Sensibilidad del círculo. | Más bajo = más detección. |
| `minRadius` | Radio mínimo del círculo. | Evita detectar basura. |
| `maxRadius` | Radio máximo del círculo. | Filtra objetos muy grandes. |
""")

