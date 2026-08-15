import streamlit as st
import pandas as pd

# 1. Configuração Global da Página
st.set_page_config(
    page_title="FaceReader OpenSource", 
    layout="wide"
)

st.title("Análise de Expressões Faciais")
st.markdown("Plataforma iterativa para extração de Action Units e Emoções (formato Long/Tidy).")

# 2. Barra Lateral (Sidebar) para Parâmetros Técnicos
with st.sidebar:
    st.header("Controlo de Extração")
    
    sampling_rate = st.selectbox(
        "Taxa de Amostragem",
        options=[
            "A cada frame (Máxima resolução)", 
            "1 FPS (1 frame por segundo)", 
            "0.5 FPS (1 frame a cada 2 segs)"
        ],
        index=1,
        help="Taxas menores aceleram o processamento, ideal para vídeos longos."
    )
    
    st.markdown("---")
    st.info("O tempo estimado de processamento (ETA) será calculado após o início da extração.")

# 3. Layout Principal: Upload e Mapeamento
st.subheader("1. Upload de Dados e Identificação")
uploaded_files = st.file_uploader(
    "Carregue os vídeos da sessão experimental", 
    type=['mp4', 'avi', 'mov'], 
    accept_multiple_files=True
)

# Dicionário para guardar o mapeamento de Vídeo -> ID do Sujeito
video_subjects = {}

if uploaded_files:
    st.markdown("**Mapeamento de Sujeitos por Vídeo:**")
    for file in uploaded_files:
        col_file, col_id = st.columns([1, 2])
        with col_file:
            st.write(f"Ficheiro: {file.name}")
        with col_id:
            # A key dinâmica evita conflitos no Streamlit quando geramos múltiplos inputs
            video_subjects[file.name] = st.text_input(
                f"ID do Sujeito", 
                key=f"id_{file.name}",
                placeholder=f"Ex: Sujeito_01 (para {file.name})",
                label_visibility="collapsed"
            )

st.markdown("---")

# 4. Metadados do Desenho Experimental (Múltiplos Estímulos)
st.subheader("2. Metadados do Desenho Experimental")
st.markdown("Defina a quantidade de estímulos e os respetivos intervalos temporais. Estas marcações gerarão a variável 'Fase_do_Estimulo' no dataset final.")

num_estimulos = st.number_input("Número de Estímulos na Sessão", min_value=1, value=1, step=1)

# Lista para armazenar os tempos de cada estímulo
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

# 5. Zona de Execução
st.subheader("3. Processamento")
if st.button("Iniciar Extração Py-FEAT", type="primary"):
    
    # Validação de IDs em falta
    missing_ids = [vid for vid, subj_id in video_subjects.items() if not subj_id.strip()]
    
    if not uploaded_files:
        st.error("Erro: Carregue pelo menos um vídeo para continuar.")
    elif missing_ids:
        st.warning("Aviso: Defina o ID do Sujeito para todos os vídeos carregados antes de processar.")
    else:
        st.success("Configuração validada com sucesso.")
        
        # Bloco temporário para confirmar a correta captura de dados na Fase 1
        st.write("Resumo dos Metadados a enviar para o motor de IA:")
        st.json({
            "Mapeamento": video_subjects,
            "Estimulos": estimulos_timestamps
        })