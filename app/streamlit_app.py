import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.predict import predict_image
from PIL import Image
import time

st.set_page_config(page_title="Detector de Enfermedades", layout="centered")

st.title("🌱 Detector Inteligente de Enfermedades en Hojas")
st.markdown("Analiza hojas de cultivo usando inteligencia artificial")
st.write("Sube una imagen de la hoja para analizarla")

uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Imagen subida", width=350)

    if st.button("🔍 Analizar hoja"):
        image_path = "temp.png"
        image.save(image_path)

        with st.spinner("Analizando imagen..."):
            time.sleep(2)
            resultado, confianza = predict_image(image_path)

        nombres = {
            "early_blight": "Tizón temprano",
            "late_blight": "Tizón tardío",
            "healthy": "Hoja sana",
            "septoria": "Septoriosis",
            "bacterial_spot": "Mancha bacteriana",
            "mosaic_virus": "Virus mosaico"
        }

        st.success(f"Diagnóstico: {nombres[resultado]}")
        st.metric("Confianza del diagnóstico", f"{confianza:.2f}%")
        st.info("Este resultado es orientativo. Consulte a un especialista agrícola.")