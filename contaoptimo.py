# app_rollos_filtrado.py
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from itertools import product

st.title("📦 Detección de Rollos con Filtro Anti-Falsos")
st.write("Detecta automáticamente rollos en una imagen y elimina detecciones falsas con un filtro inteligente.")

# --- Función de filtrado de círculos falsos ---
def filtrar_circulos(circulos, radio_min=35, radio_max=65, distancia_minima=60):
    resultado = []
    for nuevo in circulos:
        x_n, y_n, r_n = nuevo
        if not (radio_min <= r_n <= radio_max):
            continue
        demasiado_cerca = False
        for x_e, y_e, _ in resultado:
            distancia = np.sqrt((x_n - x_e)**2 + (y_n - y_e)**2)
            if distancia < distancia_minima:
                demasiado_cerca = True
                break
        if not demasiado_cerca:
            resultado.append((x_n, y_n, r_n))
    return resultado

# --- Subida de imagen ---
archivo = st.file_uploader("📷 Sube una imagen con rollos apilados", type=["jpg", "jpeg", "png"])

if archivo is not None:
    imagen_pil = Image.open(archivo).convert("RGB")
    imagen_np = np.array(imagen_pil)
    imagen_cv = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2BGR)

    st.image(imagen_pil, caption="📸 Imagen original", use_column_width=True)

    gris = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2GRAY)
    gris = cv2.medianBlur(gris, 5)

    # --- Búsqueda automática de mejores parámetros ---
    dp_vals = [1.2]
    minDist_vals = [80, 90, 100]
    param2_vals = [30, 35, 40, 45, 50]
    minRadius_vals = [35, 40]
    maxRadius_vals = [60, 70]

    mejor_circulos = None
    mejor_params = None
    max_cantidad = 0

    with st.spinner("🔎 Ajustando parámetros automáticamente..."):
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
                circulos = np.uint16(np.around(circulos[0]))
                filtrados = filtrar_circulos(circulos, radio_min=minR, radio_max=maxR, distancia_minima=minDist)
                if len(filtrados) > max_cantidad:
                    max_cantidad = len(filtrados)
                    mejor_circulos = filtrados
                    mejor_params = (dp, minDist, 50, param2, minR, maxR)

    # --- Si encuentra buenos parámetros ---
    if mejor_circulos is not None:
        dp0, minDist0, param1_0, param2_0, minR0, maxR0 = mejor_params

        st.success(f"✅ Parámetros óptimos encontrados. Puedes afinarlos si lo deseas:")
        st.code(f"""
dp = {dp0}
minDist = {minDist0}
param1 = {param1_0}
param2 = {param2_0}
minRadius = {minR0}
maxRadius = {maxR0}
        """, language="python")

        # Sliders
        dp = st.slider("dp (Resolución acumulador)", 1.0, 2.0, float(dp0), 0.1)
        minDist = st.slider("minDist (Separación entre círculos)", 20, 150, int(minDist0), 5)
        param1 = st.slider("param1 (Canny edge)", 1, 150, int(param1_0), 1)
        param2 = st.slider("param2 (Umbral Hough)", 1, 100, int(param2_0), 1)
        minRadius = st.slider("minRadius (Radio mínimo)", 1, 100, int(minR0), 1)
        maxRadius = st.slider("maxRadius (Radio máximo)", 1, 150, int(maxR0), 1)

        # Detección final con sliders
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
            circulos = np.uint16(np.around(circulos[0]))
            circulos_filtrados = filtrar_circulos(circulos, radio_min=minRadius, radio_max=maxRadius, distancia_minima=minDist)
            for x, y, r in circulos_filtrados:
                cv2.circle(salida, (x, y), r, (0, 255, 0), 2)
                cv2.circle(salida, (x, y), 2, (0, 0, 255), 3)
            cantidad = len(circulos_filtrados)

        salida_rgb = cv2.cvtColor(salida, cv2.COLOR_BGR2RGB)
        st.subheader(f"🧮 Rollos detectados: {cantidad}")
        st.image(salida_rgb, caption="📍 Imagen procesada con filtro", use_container_width=True)

    else:
        st.error("❌ No se detectaron rollos confiables.")
