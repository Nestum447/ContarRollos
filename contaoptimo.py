import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")
st.title("🧻 Detección Inteligente de Rollos Apilados")

uploaded_file = st.file_uploader("📷 Sube una imagen de los rollos", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    output = img_np.copy()

    # Parámetros ajustables
    st.sidebar.header("🔧 Parámetros de Detección")
    min_area = st.sidebar.slider("Área mínima", 100, 3000, 800, step=100)
    min_circularity = st.sidebar.slider("Circularidad mínima", 0.0, 1.0, 0.75, step=0.01)
    max_circularity = st.sidebar.slider("Circularidad máxima", 0.0, 2.0, 1.3, step=0.01)
    min_radius = st.sidebar.slider("Radio mínimo", 10, 100, 30)
    max_radius = st.sidebar.slider("Radio máximo", 10, 150, 70)

    # Procesamiento de imagen
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rollos_detectados = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if not (min_circularity <= circularity <= max_circularity):
            continue

        (x, y), r = cv2.minEnclosingCircle(cnt)
        if min_radius <= r <= max_radius:
            cv2.circle(output, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.putText(output, str(len(rollos_detectados)+1), (int(x)-10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            rollos_detectados.append((int(x), int(y), int(r)))

    # Mostrar resultados
    st.image(output, caption=f"🟢 Rollos detectados: {len(rollos_detectados)}", use_container_width=True)
    st.success(f"✅ Total de rollos detectados: {len(rollos_detectados)}")
