import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(layout="wide")
st.title("🧠 Detección Inteligente de Rollos sin Superposición")

uploaded_file = st.file_uploader("📸 Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    output = img_np.copy()

    st.sidebar.header("🎛️ Ajustes de Detección")

    param2 = st.sidebar.slider("param2 (sensibilidad)", 10, 100, 30)
    minDist = st.sidebar.slider("Distancia mínima entre centros", 10, 100, 40)
    minRadius = st.sidebar.slider("Radio mínimo", 5, 100, 30)
    maxRadius = st.sidebar.slider("Radio máximo", 5, 150, 70)
    superposition_factor = st.sidebar.slider("Factor de solapamiento permitido", 0.1, 1.5, 0.8)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=minDist,
        param1=100,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    filtered = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        for c in circles:
            x1, y1, r1 = c
            overlap = False
            for f in filtered:
                x2, y2, r2 = f
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < superposition_factor * (r1 + r2):
                    overlap = True
                    break
            if not overlap:
                filtered.append((x1, y1, r1))
                cv2.circle(output, (x1, y1), r1, (0, 255, 0), 2)
                cv2.putText(output, str(len(filtered)), (x1 - 10, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    st.image(output, caption=f"🟢 Rollos detectados sin superposición: {len(filtered)}", use_container_width=True)
    st.success(f"Total detectado: {len(filtered)} rollos")
