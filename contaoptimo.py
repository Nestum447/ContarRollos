# app_rollos_auto_sliders.py
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from itertools import product

st.title("🎯 Detección de Rollos con Ajuste Automático + Sliders")
st.write("La app detecta automáticamente los parámetros óptimos y te permite afinarlos manualmente.")

# Subir imagen
archivo = st.file_uploader("📷 Sube una imagen con rollos apilados", type=["jpg", "jpeg", "png"])

if archivo is not None:
    # Leer imagen
    imagen_pil = Image.open(archivo).convert("RGB")
    imagen_np = np.array(imagen_pil)
    imagen_cv = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2BGR)

    st.image(imagen_pil, caption="📸 Imagen original", use_column_width=True)

    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2GRAY)
    gris = cv2.medianBlur(gris, 5)

    # Rango de prueba para autoajuste
    dp_vals = [1.2]
    minDist_vals = [60, 70, 80]
    param2_vals = [20, 25, 30, 35]
    minRadius_vals = [30, 35]
    maxRadius_vals = [60, 70]

    mejor_circulos = None
    mejor_params = None
    max_cantidad = 0

    with st.spinner("🔍 Buscando parámetros óptimos..."):
        for dp, minDist, param2, minR, maxR in product(dp_vals, minDist_vals, param2_vals, minRadius_vals, maxRadius_vals):
            circulos = cv2.HoughCircles(
                gris,
                cv2.HOUGH_GRADIENT,
                dp=dp,
                minDist=minDist,
                param1=50,
                param2=param2,
                minRadius=minR,
                maxRadius=maxR
            )

            if circulos is not None:
                cantidad = len(circulos[0])
                if cantidad > max_cantidad:
                    max_cantidad = cantidad
                    mejor_circulos = circulos
                    mejor_params = (dp, minDist, 50, param2, minR, maxR)

    if mejor_circulos is not None:
        dp0, minDist0, param1_0, param2_0, minR0, maxR0 = mejor_params

        st.success(f"✅ Parámetros óptimos detectados. Ahora puedes ajustarlos con los sliders:")
        st.code(f"""
dp = {dp0}
minDist = {minDist0}
param1 = {param1_0}
param2 = {param2_0}
minRadius = {minR0}
maxRadius = {maxR0}
        """, language="python")

        # Sliders con valores óptimos prellenados
        dp = st.slider("dp (Resolución acumulador)", 1.0, 2.0, float(dp0), 0.1)
        minDist = st.slider("minDist (Distancia mínima entre círculos)", 10, 150, int(minDist0), 5)
        param1 = st.slider("param1 (Canny edge)", 1, 150, int(param1_0), 1)
        param2 = st.slider("param2 (Umbral Hough)", 1, 100, int(param2_0), 1)
        minRadius = st.slider("minRadius (Radio mínimo)", 1, 100, int(minR0), 1)
        maxRadius = st.slider("maxRadius (Radio máximo)", 1, 150, int(maxR0), 1)

        # Recalcular con sliders
        circulos = cv2.HoughCircles(
            gris,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius
        )

        salida = imagen_cv.copy()
        cantidad = 0

        if circulos is not None:
            circulos = np.uint16(np.around(circulos))
            for c in circulos[0, :]:
                cv2.circle(salida, (c[0], c[1]), c[2], (0, 255, 0), 2)
                cv2.circle(salida, (c[0], c[1]), 2, (0, 0, 255), 3)
            cantidad = len(circulos[0])

        salida_rgb = cv2.cvtColor(salida, cv2.COLOR_BGR2RGB)
        st.subheader(f"🔵 Rollos detectados: {cantidad}")
        st.image(salida_rgb, caption="Imagen con círculos detectados", use_container_width=True)

    else:
        st.error("❌ No se detectaron rollos con los parámetros probados.")
