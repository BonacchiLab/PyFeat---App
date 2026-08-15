import streamlit as st
import pandas as pd

st.set_page_config(page_title="FaceReader OpenSource", page_icon="🧠", layout="wide")

st.title("Análise de Expressões Faciais 🧠")
st.markdown("Plataforma iterativa para extração de Action Units e Emoções (formato Long/Tidy).")

with st.sidebar:
    st.header("⚙️ Controlo de Extração")
    sampling_rate = st.selectbox(
        "Taxa de Amostragem",
        options=["A cada frame", "1 FPS (1 frame/seg)", "0.5 FPS (1 frame/2 segs)"],
        index=1
    )
    st.markdown("---")
    st.info("⏱️ ETA: O motor de IA (Py-FEAT) será ligado na Fase 2.")

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("📂 1. Upload de Vídeos")
    uploaded_files = st.file_uploader(
        "Arraste os vídeos do participante", 
        type=['mp4', 'avi', 'mov'], 
        accept_multiple_files=True
    )

with col2:
    st.subheader("📋 2. Metadados do Desenho Experimental")
    subject_id = st.text_input("ID do Sujeito (ex: P001)", placeholder="Insira o ID...")
    
    st.markdown("#### Marcação de Estímulos")
    t_start = st.number_input("Início do Estímulo (segundos)", min_value=0, value=30, step=1)
    t_end = st.number_input("Fim do Estímulo (segundos)", min_value=0, value=90, step=1)

st.markdown("---")
st.subheader("🚀 3. Processamento")
if st.button("Iniciar Extração", type="primary"):
    if not uploaded_files:
        st.error("Erro: Faça upload de pelo menos um vídeo.")
    elif not subject_id:
        st.warning("Aviso: Defina o ID do Sujeito para o formato Long.")
    else:
        st.success("Interface validada! Prontos para acoplar o Py-FEAT na próxima fase.")