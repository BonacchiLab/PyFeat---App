import streamlit as st
import pandas as pd
import tempfile
import os
import cv2
from feat import Detector

#Design basico da pagina 
st.set_page_config(page_title="User-Friendly Py-FEAT Interface", layout="wide")

#Core coisas e cenas base e tal
@st.cache_resource(show_spinner=False)
def load_detector():
    return Detector()

@st.cache_data(show_spinner=False)
def get_video_metadata(file_name, file_bytes):
    """Lê os metadados do vídeo usando OpenCV sem bloquear a app."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    
    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        
    return fps if fps > 0 else 30.0, total_frames


st.title("Análise de Expressões Faciais - Extração de dados através do Py-FEAT")
st.markdown("Plataforma iterativa para extração de Action Units e Emoções de forma simples e eficiente")

#Upload de Dados
st.subheader("Upload de Dados e Identificação dos sujeitos")
uploaded_files = st.file_uploader("Carregue os vídeos da sessão experimental", type=['mp4', 'avi', 'mov'], 
    accept_multiple_files=True)

video_subjects = {}
total_frames_to_process = 0
videos_info = {}

if uploaded_files:
    st.markdown("**Mapeamento de Sujeitos por Vídeo:**")
    for file in uploaded_files:
        #Ler a cache / calculamos os metadados
        fps, frames = get_video_metadata(file.name, file.read())
        videos_info[file.name] = {"fps": fps, "frames": frames}
        
        col_file, col_id = st.columns([1, 2])
        with col_file:
            st.write(f"Ficheiro: {file.name} ({frames} frames a {fps:.2f} fps)")
        with col_id:
            video_subjects[file.name] = st.text_input(f"ID do Sujeito", key=f"id_{file.name}",placeholder=f"Ex: Sujeito_01",label_visibility="collapsed")

#Barra lateral para parâmetros tecnicos e tempo de espera
with st.sidebar:
    st.header("Controlo de Extração")
    
    sampling_rate = st.selectbox(
        "Como deseja separar os frames para a análise?",
        options=["A cada frame (Máxima resolução)", "1 FPS (1 frame por segundo)", 
            "0.5 FPS (1 frame a cada 2 segs)"],index=1)
    
    st.markdown("---")
    st.subheader("Previsão de Tempo de Processamento")
    
    #Tempo de espera
    if uploaded_files:
        frames_calculados = 0
        for file_name, info in videos_info.items():
            vid_fps = info["fps"]
            vid_frames = info["frames"]
            
            # Ajuste hipermega profissional do skip_frames baseado nos frames do vídeo
            if "A cada frame" in sampling_rate:
                skip = 1
            elif "1 FPS" in sampling_rate:
                skip = max(1, int(round(vid_fps)))
            else:
                skip = max(1, int(round(vid_fps * 2)))
                
            videos_info[file_name]["skip_frames"] = skip
            frames_calculados += (vid_frames // skip)
            
        # Assunção que tem de ser corrigida (passei de 0.08s por frame para 0.8s por frame)
        tempo_estimado_segundos = frames_calculados * 0.8
        minutos = int(tempo_estimado_segundos // 60)
        segundos = int(tempo_estimado_segundos % 60)
        
        st.metric(label="Frames a Processar", value=f"{frames_calculados}")
        st.metric(label="Tempo Estimado Total", value=f"{minutos}m {segundos}s")
    else:
        st.info("Faça upload de vídeos para calcular o tempo estimado.")

st.markdown("---")

# Metadados 
st.subheader("Metadados do Desenho Experimental")
st.markdown("Defina a quantidade de estímulos e os respetivos intervalos temporais (segundos) relativos ao vídeo. Exemplo: Estímulo 1 começa no segundo 4 e acaba no segundo 9.")

num_estimulos = st.number_input("Número de Estímulos na Sessão", min_value=1, value=1, step=1)
estimulos_timestamps = []

for i in range(int(num_estimulos)):
    st.markdown(f"**Estímulo {i + 1}**")
    col_start, col_end = st.columns(2)
    with col_start:
        t_start = st.number_input(f"Início (segundos)", min_value=0, value=30 * (i + 1), step=1, key=f"start_{i}")
    with col_end:
        t_end = st.number_input(f"Fim (segundos)", min_value=0, value=90 * (i + 1), step=1, key=f"end_{i}")
    
    estimulos_timestamps.append({
        "estimulo_id": f"Estimulo_{i + 1}", 
        "inicio": t_start, 
        "fim": t_end
    })

st.markdown("---")

#Py-FEAT
st.subheader("Processamento")
if st.button("Iniciar Extração Py-FEAT", type="primary"):
    
    missing_ids = [vid for vid, subj_id in video_subjects.items() if not subj_id.strip()]
    
    if not uploaded_files:
        st.error("Erro: Carregue pelo menos um vídeo para continuar.")
    elif missing_ids:
        st.warning("Aviso: Defina o ID do Sujeito para todos os vídeos carregados antes de processar.")
    else:
        st.success("Configuração validada. A iniciar...")
        
        with st.spinner("A carregar..."):
            detector = load_detector()
        
        # barra de loading
        progress_bar = st.progress(0, text="A preparar o processamento...")
        total_files = len(uploaded_files)
        
        for idx, file in enumerate(uploaded_files):
            subject_id = video_subjects[file.name]
            skip_frames = videos_info[file.name]["skip_frames"]
            
            # Atualizar barra 
            progress_text = f"A processar vídeo {idx + 1} de {total_files}: {file.name} (Sujeito: {subject_id})"
            progress_bar.progress(idx / total_files, text=progress_text)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                file.seek(0)
                temp_video.write(file.read())
                temp_video_path = temp_video.name
            
            try:
                with st.spinner(f"A extrair Action Units (A saltar {skip_frames} frames de cada vez)..."):
                    # detecao de video py-feat
                    video_prediction = detector.detect_video(temp_video_path, skip_frames=skip_frames)
                    
                    # GARANTIR O DATAFRAME EM PANDAS :c 
                    df_raw = video_prediction.to_pandas() if hasattr(video_prediction, 'to_pandas') else video_prediction
                    
                    st.success(f"Extração concluída para {file.name}!")
                    st.write(f"Amostra dos Dados Brutos ({df_raw.shape[0]} frames capturados):")
                    st.dataframe(df_raw.head())
                    
            finally:
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
        
        # FIMMMMM da barra 
        progress_bar.progress(1.0, text="Processamento de todos os vídeos concluído!")
        st.info("Na próxima fase (Fase 3), vamos estruturar estes dados com os timestamps dos estímulos!")