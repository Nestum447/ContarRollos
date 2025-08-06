import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")
st.title("🔍 Detección de Rollos Sin Superposición")

uploaded_file = st.file_uploader("📸 Sube la imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    output = img_np.copy()

    st.sidebar.header("🎛️ Ajustes de detección")

    param2 = st.sidebar.slider("param2 (sensibilidad)", 10, 100, 30)
    minDist = st.sidebar.slider("Distancia mínima entre centros", 10, 100, 40)
    minRadius = st.sidebar.slider("Radio mínimo", 5, 100, 30)
    maxRadius = st.sidebar.slider("Radio máximo", 5, 150, 70)
    overlap_threshold = st.sidebar.slider("Distancia mínima entre círculos aceptados", 10, 100, 40)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 3
    )

    circles = cv2.HoughCircles(
        thresh, cv2.HOUGH_GRADIENT, dp=1,
        minDist=minDist,
        param1=100, param2=param2,
        minRadius=minRadius, maxRadius=maxRadius
    )

    filtered = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Ordenar por radio descendente
        sorted_circles = sorted(circles[0, :], key=lambda x: -x[2])
        
        for new_circle in sorted_circles:
            x1, y1, r1 = new_circle
            too_close = False
            for accepted in filtered:
                x2, y2, r2 = accepted
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < overlap_threshold:
                    too_close = True
                    break
            if not too_close:
                filtered.append(new_circle)
                cv2.circle(output, (x1, y1), r1, (0, 255, 0), 2)
                cv2.putText(output, str(len(filtered)), (x1 - 10, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    st.image(output, caption=f"🟢 Rollos detectados sin superposición: {len(filtered)}", use_container_width=True)
    st.success(f"Total detectado: {len(filtered)} rollos")
