import math
from io import BytesIO
import streamlit as st
import pandas as pd
import tempfile
import os
import cv2
import torch
from feat import Detector

#Design basico da pagina 
st.set_page_config(page_title="User-Friendly Py-FEAT Interface", layout="wide")
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --ink: #edf3f1;
            --muted: #c2cfcc;
            --line: #394951;
            --surface: #202c33;
            --canvas: #111a1f;
            --accent: #f09a52;
            --accent-dark: #d87537;
            --success: #2f9e67;
            --success-dark: #247a50;
            --deep: #0b1216;
        }

        .stApp {
            background: var(--canvas);
            color: var(--ink);
            font-family: 'DM Sans', sans-serif;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: var(--deep);
            border-right: 1px solid #26363d;
        }

        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stHeader"] {
            color: #f0f6f4;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #18242a;
            border-color: #3c5059;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] input,
        [data-testid="stSidebar"] [data-baseweb="select"] span {
            color: #f5fafb;
        }

        [data-baseweb="popover"] [role="option"],
        [data-baseweb="popover"] [role="listbox"] {
            color: var(--ink);
            background: var(--surface);
        }

        [data-baseweb="popover"] [role="option"]:hover,
        [data-baseweb="popover"] [aria-selected="true"] {
            color: var(--ink);
            background: #34454c;
        }

        .block-container {
            max-width: 1400px;
            padding: 3rem 4rem 5rem;
        }

        h1, h2, h3 {
            color: var(--ink);
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: 0;
        }

        h1 {
            font-size: clamp(2rem, 3vw, 3.2rem);
            line-height: 1.08;
            margin-bottom: 2rem;
            max-width: 900px;
        }

        h2, h3 {
            margin-top: 2rem;
            padding-bottom: 0.65rem;
            border-bottom: 1px solid var(--line);
        }

        [data-testid="stFileUploader"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.35rem;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: #18242a;
            border: 1px dashed #536b73;
        }

        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.8rem 1rem;
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
        }

        [data-testid="stMetricValue"] {
            color: var(--accent-dark);
            font-family: 'Space Grotesk', sans-serif;
        }

        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div {
            border-color: var(--line);
            border-radius: 6px;
            background: var(--surface);
        }

        [data-baseweb="input"] input,
        [data-baseweb="select"] input,
        [data-baseweb="select"] span,
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input {
            color: var(--ink);
        }

        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] section,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stCaptionContainer"] {
            color: var(--muted);
        }

        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] small {
            color: #d6e1de;
        }

        [data-testid="stCheckbox"] label {
            color: var(--ink);
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            color: #f0f6f4;
        }

        .intro-hero {
            background: #1b3036;
            border-radius: 12px;
            padding: 3.5rem 3.5rem 4rem;
            margin: 1rem 0 2rem;
            color: #f1f6f4;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.28);
        }

        .intro-hero h1 {
            color: #f1f6f4;
            max-width: 760px;
            margin: 0 0 1rem;
        }

        .intro-hero p {
            color: #d0dfda;
            font-size: 1.08rem;
            max-width: 680px;
            line-height: 1.65;
            margin: 0;
        }

        .intro-note {
            border-left: 4px solid var(--accent);
            background: var(--surface);
            padding: 1rem 1.25rem;
            color: var(--muted);
            border-radius: 0 8px 8px 0;
        }

        .intro-hero {
            max-width: 980px;
            margin: 0 auto 1.25rem;
            padding: 2.25rem 2.4rem 1.5rem;
            border-radius: 14px;
            background: rgba(27, 48, 54, 0.9);
            border: 1px solid rgba(71, 104, 112, 0.6);
            box-shadow: 0 18px 30px rgba(0, 0, 0, 0.18);
        }

        .intro-hero h1 {
            margin-bottom: 0.65rem;
        }

        .intro-hero p {
            margin: 0;
            max-width: 760px;
            color: #d7e4df;
            font-size: 1.18rem;
            line-height: 1.6;
        }

        .intro-sections {
            max-width: 980px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1rem;
        }

        .info-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 1.25rem 1.3rem 1.1rem;
            box-shadow: 0 8px 18px rgba(0, 0, 0, 0.12);
        }

        .info-card h3 {
            margin-top: 0;
            margin-bottom: 0.8rem;
            padding-bottom: 0.45rem;
            border-bottom: 1px solid rgba(57, 73, 81, 0.9);
            color: var(--ink);
            font-size: 1.15rem;
        }

        .info-card p,
        .info-card li {
            color: var(--muted);
            line-height: 1.65;
            margin: 0.25rem 0;
        }

        .info-card ul {
            margin: 0.5rem 0 0 1.15rem;
            padding: 0;
        }

        [data-testid="stButton"] button,
        [data-testid="stDownloadButton"] button {
            border-radius: 10px;
            font-weight: 700;
            min-height: 3.2rem;
            padding: 0.8rem 1.6rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
        }

        [data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(180deg, #39b36f 0%, var(--success) 100%);
            border-color: #2d9b60;
            color: #ffffff;
            font-size: 1.12rem;
            letter-spacing: 0.01em;
            box-shadow: 0 10px 22px rgba(47, 158, 103, 0.35);
        }

        [data-testid="stButton"] button[kind="primary"]:hover {
            background: linear-gradient(180deg, #34a867 0%, var(--success-dark) 100%);
            border-color: var(--success-dark);
            color: #ffffff;
            transform: translateY(-1px);
            box-shadow: 0 14px 26px rgba(36, 122, 80, 0.45);
            filter: brightness(1.02);
        }

        [data-testid="stButton"] button[kind="primary"]:focus-visible {
            outline: 3px solid rgba(255, 255, 255, 0.35);
            outline-offset: 2px;
        }

        [data-testid="stDownloadButton"] button:hover {
            background: var(--accent-dark);
            border-color: var(--accent-dark);
            color: white;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }

        hr {
            border-color: var(--line);
            margin: 2.5rem 0;
        }

        @media (max-width: 800px) {
            .block-container {
                padding: 2rem 1rem 3rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

TRANSLATIONS = {
    "pt": {
        "language": "Idioma",
        "navigation": "Navegação",
        "workspace": "Área de trabalho",
        "introduction": "Introdução",
        "home": "Voltar à página de introdução",
        "portuguese": "Português",
        "english": "English",
        "title": "Análise de Expressões Faciais - Extração de dados através do Py-FEAT",
        "intro_page_title": "Facial Expression Analysis",
        "intro_page_body": "Interface para a extração de \"workable data\" através de videos.",
        "intro_placeholder": "O conteúdo introdutório desta página será definido na próxima etapa.",
        "go_workspace": "Ferramenta de Análise",
        "upload": "Upload de Dados e Identificação dos sujeitos",
        "upload_help": "Carregue os vídeos da sessão experimental",
        "subject_mapping": "**Mapeamento de Sujeitos por Vídeo:**",
        "file_info": "Ficheiro: {name} ({frames} frames a {fps:.2f} fps)",
        "subject_id": "ID do Sujeito",
        "subject_placeholder": "Ex: Sujeito_01",
        "controls": "Controlo de Extração",
        "sampling_label": "Como deseja separar os frames para a análise?",
        "all_frames": "A cada frame (Máxima resolução)",
        "fast_fps": "1 FPS (rápido; frame representativo)",
        "exact_fps": "1 FPS (média exata de todos os frames)",
        "sampling_help": "O modo rápido analisa 1 frame por segundo. A média exata analisa todos os frames e é significativamente mais lenta.",
        "time_prediction": "Previsão de Tempo de Processamento",
        "frames_analyzed": "Frames a Analisar",
        "output_rows": "Linhas no Output",
        "estimated_time": "Tempo estimado de análise: {minutes}m {seconds}s",
        "upload_for_estimate": "Faça upload de vídeos para calcular o tempo estimado.",
        "metadata": "Metadados do Desenho Experimental",
        "metadata_help": "Defina a quantidade de estímulos e os respetivos intervalos temporais (segundos) relativos ao vídeo. Exemplo: Estímulo 1 começa no segundo 4 e acaba no segundo 9.",
        "metadata_toggle": "Adicionar marcadores de estímulo ao dataset",
        "metadata_explanation": "Quando um intervalo é definido, cada frame/linha dentro dessa janela temporal recebe o nome do estímulo numa nova coluna do dataset.",
        "stimulus_count": "Número de Estímulos na Sessão",
        "stimulus": "**Estímulo {number}**",
        "stimulus_name": "Nome do estímulo",
        "stimulus_name_placeholder": "Ex: Baseline",
        "start": "Início (segundos)",
        "end": "Fim (segundos)",
        "processing": "Processamento",
        "start_extraction": "Extrair dados",
        "missing_video": "Erro: Carregue pelo menos um vídeo para continuar.",
        "missing_ids": "Aviso: Defina o ID do Sujeito para todos os vídeos carregados antes de processar.",
        "validated": "Configuração validada. A iniciar...",
        "loading": "A carregar...",
        "preparing": "A preparar o processamento...",
        "processing_video": "A processar vídeo {current} de {total}: {name} (Sujeito: {subject})",
        "extracting": "A extrair Action Units (a processar todos os frames e depois agregando conforme a seleção)...",
        "completed_file": "Extração concluída para {name}!",
        "preview": "Pré-visualização dos dados finais ({rows} linhas) — {mode}:",
        "download": "Baixar Parquet",
        "finished": "Na próxima fase (Fase 3), vamos estruturar estes dados com os timestamps dos estímulos!",
    },
    "en": {
        "language": "Language",
        "navigation": "Navigation",
        "workspace": "Workspace",
        "introduction": "Introduction",
        "home": "Return to the introduction page",
        "portuguese": "Português",
        "english": "English",
        "title": "Facial Expression Analysis - Data Extraction with Py-FEAT",
        "intro_page_title": "Facial Expression Analysis",
        "intro_page_body": "Interface for extracting \"workable data\" through videos.",
        "intro_placeholder": "The introduction content for this page will be defined in the next stage.",
        "go_workspace": "Analysis Tool",
        "upload": "Data Upload and Subject Identification",
        "upload_help": "Upload the videos from the experimental session",
        "subject_mapping": "**Subject Mapping by Video:**",
        "file_info": "File: {name} ({frames} frames at {fps:.2f} FPS)",
        "subject_id": "Subject ID",
        "subject_placeholder": "Example: Subject_01",
        "controls": "Extraction Controls",
        "sampling_label": "How should frames be selected for analysis?",
        "all_frames": "Every frame (Maximum resolution)",
        "fast_fps": "1 FPS (fast; representative frame)",
        "exact_fps": "1 FPS (exact mean of all frames)",
        "sampling_help": "Fast mode analyzes 1 frame per second. Exact mean analyzes every frame and is significantly slower.",
        "time_prediction": "Processing Time Estimate",
        "frames_analyzed": "Frames to Analyze",
        "output_rows": "Output Rows",
        "estimated_time": "Estimated analysis time: {minutes}m {seconds}s",
        "upload_for_estimate": "Upload videos to calculate the estimated time.",
        "metadata": "Experimental Design Metadata",
        "metadata_help": "Define the number of stimuli and their time intervals (seconds) relative to the video. Example: Stimulus 1 starts at second 4 and ends at second 9.",
        "metadata_toggle": "Add stimulus markers to the dataset",
        "metadata_explanation": "When a time interval is defined, every frame/row inside that window receives the stimulus name in a new dataset column.",
        "stimulus_count": "Number of Stimuli in the Session",
        "stimulus": "**Stimulus {number}**",
        "stimulus_name": "Stimulus name",
        "stimulus_name_placeholder": "Example: Baseline",
        "start": "Start (seconds)",
        "end": "End (seconds)",
        "processing": "Processing",
        "start_extraction": "Extract data",
        "missing_video": "Error: Upload at least one video to continue.",
        "missing_ids": "Warning: Define the Subject ID for every uploaded video before processing.",
        "validated": "Configuration validated. Starting...",
        "loading": "Loading...",
        "preparing": "Preparing processing...",
        "processing_video": "Processing video {current} of {total}: {name} (Subject: {subject})",
        "extracting": "Extracting Action Units (processing all frames and aggregating according to the selection)...",
        "completed_file": "Extraction completed for {name}!",
        "preview": "Preview of final data ({rows} rows) — {mode}:",
        "download": "Download Parquet",
        "finished": "Next phase (Phase 3): structure these data with the stimulus timestamps!",
    },
}

page = st.session_state.get("app_page", "introduction")

if page == "introduction":
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"],
            [data-testid="stSidebarCollapsedControl"] {
                display: none;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

if page == "workspace":
    with st.sidebar:
        language_name = st.selectbox(
            "Idioma / Language",
            ["Português", "English"],
            index=0,
            key="interface_language",
        )
else:
    language_name = st.selectbox(
        "Idioma / Language",
        ["Português", "English"],
        index=0,
        key="interface_language",
    )

language = "en" if language_name == "English" else "pt"


def t(key, **values):
    return TRANSLATIONS[language][key].format(**values)


if page == "introduction":
    st.markdown(
        f"""
        <section class="intro-hero">
            <h1>{t("intro_page_title")}</h1>
            <p>{t("intro_page_body")}</p>
        </section>

        <div class="intro-sections">
            <div class="info-card">
                <h3>Como funciona esta interface?</h3>
                <p>Esta interface permite carregar vídeos, identificar os sujeitos, definir como os frames devem ser amostrados e extrair indicadores do Py-FEAT, como Action Units, emoções e métricas faciais. Todo o processo é guiado em etapas para reduzir erros e facilitar a análise.</p>
            </div>

            <div class="info-card">
                <h3>O que necessita para funcionar?</h3>
                <ul>
                    <li>Vídeos em formatos compatíveis, como MP4.</li>
                    <li>Identificação dos sujeitos associada a cada ficheiro.</li>
                    <li>Definição do tipo de amostragem e do desenho experimental.</li>
                    <li>Uma execução do detector Py-FEAT disponível no ambiente da aplicação.</li>
                </ul>
            </div>

            <div class="info-card">
                <h3>Que benefícios dá cada uma das maneiras de extração dos dados?</h3>
                <ul>
                    <li><strong>1 FPS média:</strong> reduz a quantidade de dados e cria uma estimativa estável por segundo.</li>
                    <li><strong>1 FPS frame representativo:</strong> escolhe um frame característico da segunda para manter a velocidade e simplificar a interpretação.</li>
                    <li><strong>Máxima resolução:</strong> preserva todos os frames e maximiza a granularidade analítica, ideal para estudos mais detalhados.</li>
                </ul>
            </div>

            <div class="info-card">
                <h3>O que significam cada uma das colunas do dataset do output?</h3>
                <p>As colunas incluem identificadores do sujeito, tempo ou frame, métricas de Action Units, valores emocionais e indicadores faciais extraídos. Em conjunto, permitem transformar cada vídeo num conjunto de observações prontos para análise estatística e comparação entre condições experimentais.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button(t("go_workspace"), type="primary", use_container_width=True):
        st.session_state["app_page"] = "workspace"
        st.rerun()
    st.stop()


sampling_labels = {
    t("all_frames"): "all_frames",
    t("fast_fps"): "fast_fps",
    t("exact_fps"): "exact_fps",
}

#Core coisas e cenas base e tal
@st.cache_resource(show_spinner=False)
def load_detector():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Detector(device=device)

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


def format_seconds_to_clock(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def aggregate_by_second(df, fps):
    """Aggregate a per-frame dataframe into one row per second, using the mean of all frames in each second."""
    if df is None or df.empty or "frame" not in df.columns:
        return df

    df = df.copy()
    df["_frame_index"] = pd.to_numeric(df["frame"], errors="coerce")
    df["_second_bin"] = (df["_frame_index"] / float(fps)).astype(float).floordiv(1).astype(int)

    numeric_cols = [
        col for col in df.columns
        if col not in {"input", "frame", "approx_time", "_frame_index", "_second_bin"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not numeric_cols:
        return df

    grouped = df.groupby("_second_bin", as_index=False)[numeric_cols].mean()
    grouped["second"] = grouped["_second_bin"].astype(int)
    grouped["time_start_sec"] = grouped["second"].astype(float)
    grouped["time_end_sec"] = grouped["second"].astype(float) + 1.0
    grouped["approx_time"] = grouped["time_start_sec"].apply(format_seconds_to_clock)
    grouped["frame"] = grouped["time_start_sec"] * float(fps)

    if "input" in df.columns:
        grouped["input"] = df.groupby("_second_bin")["input"].first().reset_index(drop=True).to_numpy()

    grouped = grouped.sort_values("second").reset_index(drop=True)
    return grouped


def prepare_frame_dataframe(df_raw, fps, sampling_mode):
    """Return a dataframe that is either raw per-frame or aggregated per second."""
    if sampling_mode == "all_frames":
        df_out = df_raw.copy()
        df_out["time_sec"] = pd.to_numeric(df_out["frame"], errors="coerce") / float(fps)
        df_out["approx_time"] = df_out["time_sec"].apply(format_seconds_to_clock)
        return df_out

    return aggregate_by_second(df_raw, fps)


def apply_stimulus_metadata(df, stimuli):
    """Label output rows whose time falls inside a configured stimulus window."""
    if df is None or df.empty:
        return df

    df = df.copy()
    if "time_sec" in df.columns:
        row_times = pd.to_numeric(df["time_sec"], errors="coerce")
    elif "time_start_sec" in df.columns:
        row_times = pd.to_numeric(df["time_start_sec"], errors="coerce")
    elif "frame" in df.columns:
        row_times = pd.to_numeric(df["frame"], errors="coerce")
    else:
        row_times = pd.Series(float("nan"), index=df.index)

    labels = pd.Series("", index=df.index, dtype="object")
    for stimulus in stimuli:
        start = float(stimulus["inicio"])
        end = float(stimulus["fim"])
        if end < start:
            start, end = end, start
        matches = row_times.ge(start) & row_times.lt(end)
        labels.loc[matches] = stimulus["nome"]

    df["stimulus"] = labels
    return df


def dataframe_to_parquet_bytes(df):
    buffer = BytesIO()
    df.to_parquet(buffer, engine="pyarrow", compression="snappy", index=False)
    return buffer.getvalue()


st.title(t("title"))

#Upload de Dados
st.subheader(t("upload"))
uploaded_files = st.file_uploader(t("upload_help"), type=['mp4', 'avi', 'mov'],
    accept_multiple_files=True)

video_subjects = {}
total_frames_to_process = 0
videos_info = {}

if uploaded_files:
    st.markdown(t("subject_mapping"))
    for file in uploaded_files:
        #Ler a cache / calculamos os metadados
        fps, frames = get_video_metadata(file.name, file.read())
        videos_info[file.name] = {"fps": fps, "frames": frames}
        
        col_file, col_id = st.columns([1, 2])
        with col_file:
            st.write(t("file_info", name=file.name, frames=frames, fps=fps))
        with col_id:
            video_subjects[file.name] = st.text_input(t("subject_id"), key=f"id_{file.name}", placeholder=t("subject_placeholder"), label_visibility="collapsed")

#Barra lateral para parâmetros tecnicos e tempo de espera
with st.sidebar:
    if st.button("⌂", help=t("home"), key="home_button", use_container_width=True):
        st.session_state["app_page"] = "introduction"
        st.rerun()

    st.header(t("controls"))
    
    sampling_rate = st.selectbox(
        t("sampling_label"),
        options=list(sampling_labels),
        index=1,
    )
    sampling_mode = sampling_labels[sampling_rate]
    st.caption(t("sampling_help"))
    
    st.markdown("---")
    st.subheader(t("time_prediction"))
    
    #Tempo de espera
    if uploaded_files:
        frames_to_analyze = 0
        output_rows = 0
        for file_name, info in videos_info.items():
            vid_fps = info["fps"]
            vid_frames = info["frames"]

            if sampling_mode in {"all_frames", "exact_fps"}:
                skip = 1
                effective_rows = vid_frames
            else:
                skip = max(1, int(round(vid_fps)))
                effective_rows = max(1, int(math.ceil(vid_frames / max(1.0, vid_fps))))

            videos_info[file_name]["skip_frames"] = skip
            videos_info[file_name]["sampling_mode"] = sampling_mode
            frames_to_analyze += vid_frames // skip
            output_rows += effective_rows

        tempo_estimado_segundos = frames_to_analyze * 0.8
        minutos = int(tempo_estimado_segundos // 60)
        segundos = int(tempo_estimado_segundos % 60)

        st.metric(label=t("frames_analyzed"), value=f"{frames_to_analyze}")
        st.metric(label=t("output_rows"), value=f"{output_rows}")
        st.caption(t("estimated_time", minutes=minutos, seconds=segundos))
    else:
        st.info(t("upload_for_estimate"))

st.markdown("---")

# Metadados 
estimulos_timestamps = []
st.subheader(t("metadata"))
add_metadata = st.checkbox(t("metadata_toggle"), value=False)
st.caption(t("metadata_explanation"))

if add_metadata:
    st.markdown(t("metadata_help"))
    num_estimulos = st.number_input(t("stimulus_count"), min_value=1, value=1, step=1)

    for i in range(int(num_estimulos)):
        st.markdown(t("stimulus", number=i + 1))
        stimulus_name = st.text_input(
            t("stimulus_name"),
            value=f"Stimulus_{i + 1}",
            placeholder=t("stimulus_name_placeholder"),
            key=f"stimulus_name_{i}",
        ).strip()
        col_start, col_end = st.columns(2)
        with col_start:
            t_start = st.number_input(t("start"), min_value=0.0, value=float(30 * (i + 1)), step=1.0, key=f"start_{i}")
        with col_end:
            t_end = st.number_input(t("end"), min_value=0.0, value=float(90 * (i + 1)), step=1.0, key=f"end_{i}")

        estimulos_timestamps.append({
            "estimulo_id": f"Stimulus_{i + 1}",
            "nome": stimulus_name or f"Stimulus_{i + 1}",
            "inicio": t_start,
            "fim": t_end,
        })

st.markdown("---")

#Py-FEAT
st.subheader(t("processing"))
if st.button(t("start_extraction"), type="primary"):
    
    missing_ids = [vid for vid, subj_id in video_subjects.items() if not subj_id.strip()]
    
    if not uploaded_files:
        st.error(t("missing_video"))
    elif missing_ids:
        st.warning(t("missing_ids"))
    else:
        st.success(t("validated"))
        
        with st.spinner(t("loading")):
            detector = load_detector()
        
        # barra de loading
        progress_bar = st.progress(0, text=t("preparing"))
        total_files = len(uploaded_files)
        
        for idx, file in enumerate(uploaded_files):
            subject_id = video_subjects[file.name]
            skip_frames = videos_info[file.name]["skip_frames"]
            sampling_mode = videos_info[file.name].get("sampling_mode", sampling_mode)

            progress_text = t("processing_video", current=idx + 1, total=total_files, name=file.name, subject=subject_id)
            progress_bar.progress(idx / total_files, text=progress_text)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                file.seek(0)
                temp_video.write(file.read())
                temp_video_path = temp_video.name

            try:
                with st.spinner(t("extracting")):
                    with torch.no_grad():
                        video_prediction = detector.detect_video(
                            temp_video_path,
                            skip_frames=skip_frames,
                            batch_size=4,
                        )
                    df_raw = video_prediction.to_pandas() if hasattr(video_prediction, 'to_pandas') else video_prediction

                    output_df = prepare_frame_dataframe(df_raw, videos_info[file.name]["fps"], sampling_mode)
                    if add_metadata:
                        output_df = apply_stimulus_metadata(output_df, estimulos_timestamps)
                    output_df["subject_id"] = subject_id
                    output_df["input_file"] = file.name

                    st.success(t("completed_file", name=file.name))
                    st.write(t("preview", rows=output_df.shape[0], mode=sampling_rate))
                    st.dataframe(output_df.head(50), use_container_width=True)

                    parquet_bytes = dataframe_to_parquet_bytes(output_df)
                    st.download_button(
                        label=t("download"),
                        data=parquet_bytes,
                        file_name=f"{subject_id}_{os.path.splitext(file.name)[0]}_pyfeat.parquet",
                        mime="application/vnd.apache.parquet",
                    )

            finally:
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
        
        # FIMMMMM da barra 
        progress_bar.progress(1.0, text=t("finished"))
        st.info(t("finished"))