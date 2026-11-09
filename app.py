import base64
import math
import re
import threading
import time
from functools import partial
from io import BytesIO
import streamlit as st
import pandas as pd
import tempfile
import os
import cv2
import torch
from feat import Detector
import feat.detector as feat_detector
import plotly.express as px
import plotly.graph_objects as go

DETECT_VIDEO_PROGRESS_LOCK = threading.Lock()
BASELINE_SECONDS_PER_FRAME = 4.0

st.set_page_config(page_title="FEDET powered by py-feat", layout="wide")
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

        .st-key-global-navigation {
            position: sticky;
            top: 0;
            z-index: 100;
            background: var(--canvas);
            border-bottom: 1px solid var(--line);
            padding: 0.2rem 0;
            margin-bottom: 0.9rem;
        }

        .global-nav-brand {
            color: var(--ink);
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            padding-left: 0.35rem;
        }

        .st-key-global-navigation [data-testid="stButton"] button {
            min-height: 2.5rem;
            border-radius: 5px;
            white-space: nowrap;
        }

        .st-key-global-navigation [data-testid="stButton"] button p {
            white-space: nowrap;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        a[aria-label="Link to heading"] {
            display: none !important;
        }

        [data-testid="stSidebar"] {
            background: var(--deep);
            border-right: 1px solid #26363d;
        }

        [data-testid="stSidebarCollapsedControl"] button,
        button[data-testid="stExpandSidebarButton"] {
            width: auto !important;
            min-width: max-content;
            padding: 0.45rem 0.7rem !important;
            border: 1px solid var(--line) !important;
            border-radius: 6px !important;
            background: var(--surface) !important;
            gap: 0.4rem;
        }

        [data-testid="stSidebarCollapsedControl"] button::after,
        button[data-testid="stExpandSidebarButton"]::after {
            content: "Extraction Controls";
            color: var(--ink);
            font-family: 'DM Sans', sans-serif;
            font-size: 0.82rem;
            font-weight: 600;
            white-space: nowrap;
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
            margin: 0;
        }

        .intro-hero .hero-subtitle {
            color: #d0dfda;
            font-size: 1.18rem;
            max-width: 680px;
            line-height: 1.65;
            margin: 0.45rem auto 0;
        }

        .intro-hero .hero-powered-by {
            color: #a9bbb6;
            font-size: 0.9rem;
            font-weight: 500;
            letter-spacing: 0;
            margin: 0.45rem auto 0;
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
            background: #1b3036;
            border: 1px solid #476870;
            border-top: 3px solid var(--accent);
            box-shadow: 0 18px 30px rgba(0, 0, 0, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.04);
            text-align: center;
        }

        .intro-hero h1 {
            margin-bottom: 0.65rem;
            margin-left: auto;
            margin-right: auto;
            font-size: 4.5rem;
            line-height: 1;
        }

        .intro-sections {
            max-width: 980px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1rem;
        }

        .st-key-intro-guide-layout {
            max-width: 980px;
            margin: 0 auto;
        }

        div:has(> .st-key-intro-guide-sticky) {
            position: sticky;
            top: 1.5rem;
            z-index: 10;
        }

        .guide-nav {
            background: #18262c;
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 1rem;
        }

        .st-key-intro-nav-workspace-button [data-testid="stButton"] button[kind="primary"] {
            background: #155b37;
            border-color: #155b37;
            color: #ffffff;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.28);
        }

        .st-key-intro-nav-workspace-button [data-testid="stButton"] button[kind="primary"]:hover {
            background: #104a2d;
            border-color: #104a2d;
            color: #ffffff;
        }

        .st-key-intro-nav-workspace-button [data-testid="stButton"] {
            width: 100%;
        }

        .guide-nav h2 {
            margin: 0 0 0.75rem;
            padding: 0;
            border: 0;
            font-size: 0.95rem;
            color: var(--ink);
        }

        .guide-nav a {
            display: block;
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.35;
            padding: 0.42rem 0.5rem;
            border-radius: 5px;
            text-decoration: none;
        }

        .guide-nav a:hover {
            color: var(--ink);
            background: #263940;
        }

        .guide-nav-warning {
            color: #ef5350;
            font-size: 1.15em;
            font-weight: 700;
            margin-right: 0.3rem;
            vertical-align: -0.04em;
        }

        .guide-content {
            min-width: 0;
        }

        .guide-section {
            scroll-margin-top: 1.25rem;
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 1.35rem 1.5rem;
            margin-bottom: 1rem;
        }

        .guide-section h2 {
            margin: 0 0 0.85rem;
            padding: 0 0 0.65rem;
            border-bottom: 1px solid var(--line);
            color: var(--ink);
            font-size: 1.35rem;
        }

        .guide-section h3 {
            margin: 1.15rem 0 0.35rem;
            padding: 0;
            border: 0;
            color: var(--ink);
            font-size: 1rem;
        }

        .guide-section p,
        .guide-section li {
            color: var(--muted);
            line-height: 1.65;
        }

        .guide-section p {
            margin: 0.45rem 0;
        }

        .guide-section ul,
        .guide-section ol {
            margin: 0.55rem 0 0.25rem 1.2rem;
            padding: 0;
        }

        .guide-section strong {
            color: var(--ink);
        }

        .guide-table-wrap {
            overflow-x: auto;
            margin-top: 0.75rem;
        }

        .guide-table {
            width: 100%;
            border-collapse: collapse;
            color: var(--muted);
            font-size: 0.86rem;
        }

        .guide-table th,
        .guide-table td {
            border: 1px solid var(--line);
            padding: 0.65rem;
            text-align: left;
            vertical-align: top;
        }

        .guide-table th {
            background: #18262c;
            color: var(--ink);
        }

        .guide-note {
            background: #18262c;
            border-left: 3px solid var(--accent);
            border-radius: 0 6px 6px 0;
            padding: 0.75rem 0.9rem;
            margin-top: 0.9rem;
        }

        .guide-note p {
            margin: 0;
        }

        .image-placeholder {
            border: 1px dashed #536b73;
            border-radius: 7px;
            color: #9eb0ac;
            padding: 0.8rem 1rem;
            margin-top: 0.85rem;
            font-size: 0.9rem;
        }

        .guide-links a {
            color: #d6a06f;
        }

        @media (max-width: 800px) {
            div:has(> .st-key-intro-guide-sticky) {
                position: static;
            }

            div:has(> .st-key-intro-nav-workspace-button) {
                position: fixed;
                left: 1rem;
                right: auto;
                bottom: max(1rem, env(safe-area-inset-bottom));
                top: auto;
                width: calc(100vw - 2rem);
                z-index: 1000;
            }
        }

        @media (min-width: 801px) and (max-height: 700px) {
            .guide-nav {
                padding: 0.65rem;
            }

            .guide-nav h2 {
                margin-bottom: 0.35rem;
            }

            .guide-nav a {
                font-size: 0.78rem;
                line-height: 1.15;
                padding: 0.18rem 0.35rem;
            }
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
                padding: 2rem 1rem 6rem;
            }

            .intro-hero {
                padding: 2rem 1.25rem 1.5rem;
            }

            .intro-hero h1 {
                font-size: 3rem;
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
        "title": "FEDET - Facial Expression Data Extraction Tool powered by py-feat",
        "intro_page_title": "FEDET",
        "intro_page_body": "Facial Expression Data Extraction Tool",
        "intro_page_powered_by": "powered by py-feat",
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
        "fast_fps": "1 FPS (frame representativo)",
        "custom_sampling": "Custom sampling",
        "custom_sampling_fps": "Target sampling frequency (frames per second)",
        "custom_sampling_help": "A frequência selecionada é aproximada e depende da taxa de frames do vídeo original.",
        "sampling_help": "O modo rápido seleciona aproximadamente um frame representativo por segundo.",
        "time_prediction": "Previsão de Tempo de Processamento",
        "frames_analyzed": "Frames a Analisar",
        "output_rows": "Linhas no Output",
        "estimated_time": "Tempo estimado",
        "estimated_time_disclaimer": "O tempo de loading é apenas uma estimativa e ficará mais preciso após a primeira utilização.",
        "upload_for_estimate": "Faça upload de vídeos para calcular o tempo estimado.",
        "metadata": "Metadados do Desenho Experimental",
        "metadata_help": "Defina a quantidade de estímulos e os respetivos intervalos temporais (segundos) relativos ao vídeo. Exemplo: Estímulo 1 começa no segundo 4 e acaba no segundo 9.",
        "metadata_toggle": "Adicionar marcadores de estímulo ao dataset",
        "metadata_explanation": "O dataset terá uma coluna por estímulo; em intervalos sobrepostos, as colunas dos dois estímulos ficam preenchidas.",
        "stimulus_count": "Número de Estímulos na Sessão",
        "stimulus": "**Estímulo {number}**",
        "stimulus_selector": "Estímulo",
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
        "processing_video_progress": "Vídeo {current}/{total}: {name} — {percent:.0f}% concluído; restante estimado: {minutes}m {seconds}s",
        "extracting": "A extrair Action Units dos frames selecionados...",
        "completed_file": "Extração concluída para {name}!",
        "preview": "Pré-visualização dos dados finais ({rows} linhas) — {mode}:",
        "download": "Baixar Parquet",
        "visualizations": "Visualizações interativas",
        "chart_help_button": "Sobre este gráfico",
        "chart_face_score_help": "Cada ponto é uma deteção num frame. O eixo horizontal mostra o tempo ou frame; o vertical mostra o FaceScore (0–1). A linha vermelha é o limite escolhido para revisão. Um score baixo pede inspeção da imagem, mas não diz se uma AU ou emoção está correta.",
        "chart_signals_help": "O eixo horizontal é o tempo ou frame e o vertical é o valor produzido pelo modelo. Escolha AUs, emoções ou pose nos controlos. As faixas sombreadas marcam estímulos. Compare tendências, não escalas diferentes; estes valores são estimativas do modelo.",
        "chart_heatmap_help": "Cada linha representa uma AU e cada coluna um frame amostrado. A cor mostra o valor estimado pelo modelo; células sem cor são dados em falta, não zeros. As faixas sombreadas assinalam estímulos para ajudar a localizar mudanças no tempo.",
        "chart_before_during_help": "Cada par de pontos é uma gravação: a média nos 5 segundos antes do início e a média durante o estímulo. A distância vertical mostra a diferença observada. Cada gravação conta uma vez; a comparação é descritiva e não prova causalidade.",
        "chart_event_aligned_help": "O eixo horizontal mostra segundos desde o início do estímulo: 0 é o início e a faixa sombreada é a sua duração. O eixo vertical mostra o sinal escolhido. Cada linha é uma gravação; lacunas indicam dados em falta. Compare formas e diferenças entre participantes.",
        "chart_similarity_help": "Cada célula compara duas gravações pela correlação de Pearson (r) da forma temporal, dividida em 20 fases relativas ao estímulo. r perto de 1 indica trajetórias semelhantes; perto de -1, trajetórias inversas. Só contam fases presentes em ambas; não mede emoção nem causalidade.",
        "visualizations_after_processing": "Carregue um vídeo e selecione **Extrair dados** para gerar o output e as respetivas visualizações interativas.",
        "quality_tab": "Qualidade dos dados",
        "signals_tab": "Sinais ao longo do tempo",
        "heatmap_tab": "Heatmap de AUs",
        "output_rows_metric": "Linhas de output",
        "frames_with_detections": "Frames com deteção",
        "median_face_score": "FaceScore mediano",
        "flagged_face_scores": "FaceScores abaixo do limite",
        "confidence_threshold": "Sinalizar FaceScore abaixo de",
        "quality_threshold_note": "Sinaliza para revisão; não remove dados automaticamente.",
        "no_face_score": "Este output não contém FaceScore para avaliar a confiança da deteção.",
        "signal_category": "Grupo de métricas",
        "au_group": "Action Units",
        "emotion_group": "Emoções",
        "pose_group": "Pose da cabeça",
        "select_signals": "Sinais a apresentar",
        "stimulus_comparison": "Comparação antes/durante o estímulo",
        "stimulus_metric": "Métrica para comparar",
        "no_stimulus_comparison": "Adicione intervalos de estímulo e processe pelo menos um vídeo para comparar períodos.",
        "advanced_stimulus_analysis": "Análise avançada entre gravações",
        "event_aligned_tab": "Trajetórias alinhadas ao estímulo",
        "similarity_tab": "Similaridade das gravações",
        "no_event_aligned_data": "Não há dados suficientes em torno deste estímulo para traçar as gravações.",
        "similarity_min_recordings": "São necessárias pelo menos duas gravações com dados suficientes neste intervalo.",
        "finished": "Processamento concluído.",
    },
    "en": {
        "language": "Language",
        "navigation": "Navigation",
        "workspace": "Workspace",
        "introduction": "Introduction",
        "home": "Return to the introduction page",
        "portuguese": "Português",
        "english": "English",
        "title": "FEDET - Facial Expression Data Extraction Tool powered by py-feat",
        "intro_page_title": "FEDET",
        "intro_page_body": "Facial Expression Data Extraction Tool",
        "intro_page_powered_by": "powered by py-feat",
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
        "fast_fps": "1 FPS (representative frame)",
        "custom_sampling": "Custom sampling",
        "custom_sampling_fps": "Target sampling frequency (frames per second)",
        "custom_sampling_help": "The selected frequency is approximate and depends on the source video's frame rate.",
        "sampling_help": "Fast mode analyzes 1 frame per second using a representative frame.",
        "time_prediction": "Processing Time Estimate",
        "frames_analyzed": "Frames to Analyze",
        "output_rows": "Output Rows",
        "estimated_time": "Estimated time",
        "estimated_time_disclaimer": "Loading time is only an estimate and will become more accurate after the first use.",
        "upload_for_estimate": "Upload videos to calculate the estimated time.",
        "metadata": "Experimental Design Metadata",
        "metadata_help": "Define the number of stimuli and their time intervals (seconds) relative to the video. Example: Stimulus 1 starts at second 4 and ends at second 9.",
        "metadata_toggle": "Add stimulus markers to the dataset",
        "metadata_explanation": "The dataset will have one column per stimulus; overlapping intervals will be marked in both columns.",
        "stimulus_count": "Number of Stimuli in the Session",
        "stimulus": "**Stimulus {number}**",
        "stimulus_selector": "Stimulus",
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
        "processing_video_progress": "Video {current}/{total}: {name} — {percent:.0f}% complete; estimated remaining: {minutes}m {seconds}s",
        "extracting": "Extracting Action Units from the selected video frames...",
        "completed_file": "Extraction completed for {name}!",
        "preview": "Preview of final data ({rows} rows) — {mode}:",
        "download": "Download Parquet",
        "visualizations": "Interactive visualizations",
        "chart_help_button": "About this chart",
        "chart_face_score_help": "Each point is a frame-level detection. The horizontal axis is time or frame; the vertical axis is FaceScore (0–1). The red line is your review threshold. A low score calls for image inspection; it does not tell you whether an AU or emotion is correct.",
        "chart_signals_help": "The horizontal axis is time or frame; the vertical axis is the model output. Choose AUs, emotions, or pose in the controls. Shaded bands mark stimuli. Compare trends, not unlike scales; these values are model estimates.",
        "chart_heatmap_help": "Each row is an AU and each column is a sampled frame. Color shows the model estimate; blank cells mean missing data, not zero. Shaded bands mark stimuli and help locate changes over time.",
        "chart_before_during_help": "Each pair of points is one recording: the mean for the 5 seconds before onset and the mean during the stimulus. Vertical distance shows the observed difference. Each recording counts once; this is descriptive, not causal evidence.",
        "chart_event_aligned_help": "The horizontal axis shows seconds from stimulus onset: 0 is onset and the shaded band is its duration. The vertical axis shows the selected signal. Each line is one recording; gaps indicate missing data. Compare shapes and differences across participants.",
        "chart_similarity_help": "Each cell compares two recordings using Pearson r for the temporal shape across 20 stimulus-relative phases. Values near 1 indicate similar trajectories; near -1, inverse trajectories. Only phases observed in both recordings count; this is not emotion or causality.",
        "visualizations_after_processing": "Upload a video and select **Extract data** to generate the output and its interactive visualizations.",
        "quality_tab": "Data quality",
        "signals_tab": "Signals over time",
        "heatmap_tab": "AU heatmap",
        "output_rows_metric": "Output rows",
        "frames_with_detections": "Frames with detections",
        "median_face_score": "Median FaceScore",
        "flagged_face_scores": "FaceScores below threshold",
        "confidence_threshold": "Flag FaceScore below",
        "quality_threshold_note": "Flags rows for review; it does not remove data.",
        "no_face_score": "This output does not contain FaceScore to assess detection confidence.",
        "signal_category": "Metric group",
        "au_group": "Action Units",
        "emotion_group": "Emotions",
        "pose_group": "Head pose",
        "select_signals": "Signals to display",
        "stimulus_comparison": "Before/during stimulus comparison",
        "stimulus_metric": "Metric to compare",
        "no_stimulus_comparison": "Add stimulus windows and process at least one video to compare periods.",
        "advanced_stimulus_analysis": "Advanced across-recording analysis",
        "event_aligned_tab": "Event-aligned trajectories",
        "similarity_tab": "Recording similarity",
        "no_event_aligned_data": "There is not enough data around this stimulus to plot the recordings.",
        "similarity_min_recordings": "At least two recordings with sufficient data in this interval are required.",
        "finished": "Processing complete.",
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

language = "en"


def t(key, **values):
    return TRANSLATIONS[language][key].format(**values)


with st.container(key="global-navigation"):
    nav_brand, nav_introduction, nav_extraction = st.columns([4, 2, 2.2], vertical_alignment="center")
    with nav_brand:
        st.markdown('<div class="global-nav-brand">FEDET</div>', unsafe_allow_html=True)
    with nav_introduction:
        if st.button(
            t("introduction"),
            type="primary" if page == "introduction" else "secondary",
            use_container_width=True,
            key="global_nav_introduction",
        ) and page != "introduction":
            st.session_state["app_page"] = "introduction"
            st.rerun()
    with nav_extraction:
        if st.button(
            t("go_workspace"),
            type="primary" if page == "workspace" else "secondary",
            use_container_width=True,
            key="global_nav_extraction",
        ) and page != "workspace":
            st.session_state["app_page"] = "workspace"
            st.rerun()


if page == "introduction":
    with open(os.path.join(os.path.dirname(__file__), "Au-Reference.jpg"), "rb") as image_file:
        au_reference_image = base64.b64encode(image_file.read()).decode("ascii")

    intro_guide = {
        "pt": """
        <nav class="guide-nav" aria-label="Navegação do guia">
            <h2>Neste guia</h2>
            <a href="#getting-started">Como começar</a>
            <a href="#preparing-videos">Preparar os vídeos</a>
            <a href="#understanding-output">Compreender o dataset de output</a>
            <a href="#action-units">Action Units (Unidades de Ação Facial)</a>
            <a href="#au-reference">Referência de Action Units</a>
            <a href="#emotions">Emoções e métricas faciais</a>
            <a href="#landmarks">Landmarks faciais e deteção do rosto</a>
            <a href="#sampling">Modos de amostragem</a>
            <a href="#metadata">Metadados experimentais e estímulos</a>
            <a href="#good-practices">Boas práticas e limitações</a>
        </nav>
        <div class="guide-content">
            <section class="guide-section" id="getting-started">
                <h2>Como começar</h2>
                <p>Esta interface transforma vídeos em dados estruturados sobre a atividade facial ao longo do tempo. Não é necessário programar: basta seguir as etapas na área de análise.</p>
                <ol>
                    <li>Carregue um ou mais vídeos nos formatos suportados.</li>
                    <li>Atribua um <strong>Subject ID (ID do Sujeito)</strong> a cada vídeo.</li>
                    <li>Escolha o modo de amostragem adequado à sua pergunta de investigação.</li>
                    <li>Opcionalmente, defina os intervalos dos estímulos experimentais.</li>
                    <li>Selecione <strong>Extract data (Extrair dados)</strong>, aguarde o processamento e descarregue cada ficheiro Parquet.</li>
                </ol>
                <div class="guide-note"><p>O resultado é um dataset em que cada linha representa uma observação facial num frame ou num intervalo de um segundo.</p></div>
            </section>

            <section class="guide-section" id="preparing-videos">
                <h2>Preparar os vídeos</h2>
                <p>Para obter deteções mais consistentes, use vídeos em que o rosto está visível durante a maior parte do tempo e ocupa uma área razoável da imagem.</p>
                <ul>
                    <li>Evite iluminação muito escura, contraluz forte e alterações bruscas de exposição.</li>
                    <li>Evite rostos muito pequenos, parcialmente ocultos ou excessivamente inclinados.</li>
                    <li>Verifique se o vídeo contém apenas os participantes relevantes ou tenha em conta que podem ser detetados vários rostos.</li>
                    <li>Mantenha uma qualidade e uma resolução semelhantes entre vídeos que serão comparados.</li>
                </ul>
                <div class="guide-note"><p>O Py-Feat faz previsões frame a frame. Por isso, uma deteção num frame não é uma garantia de que o mesmo rosto será identificado da mesma forma em todos os frames seguintes.</p></div>
            </section>

            <section class="guide-section" id="understanding-output">
                <h2>Compreender o dataset de output</h2>
                <p>O Parquet contém as previsões do Py-Feat e os identificadores acrescentados pela interface. O conjunto exato de colunas depende do detector e dos modelos disponíveis.</p>
                <ul>
                    <li><strong>subject_id:</strong> identificação atribuída ao participante no momento do upload.</li>
                    <li><strong>input_file:</strong> nome do vídeo de origem.</li>
                    <li><strong>frame:</strong> número do frame analisado no vídeo original.</li>
                    <li><strong>time_sec:</strong> tempo aproximado do frame em segundos, no modo Every frame.</li>
                    <li><strong>approx_time:</strong> representação legível do tempo, como 00:12.</li>
                    <li><strong>second, time_start_sec e time_end_sec:</strong> intervalo temporal criado quando os dados são agregados por segundo.</li>
                    <li><strong>Stimulus_1, Stimulus_2, etc.:</strong> uma coluna por estímulo; uma linha pode estar marcada em várias colunas quando os intervalos se sobrepõem.</li>
                    <li><strong>Action Units, emotions, landmarks e outras métricas:</strong> previsões faciais produzidas pelo modelo Py-Feat.</li>
                </ul>
                <div class="guide-note"><p>Leia sempre os nomes das colunas antes da análise. Nem todos os detectores produzem exatamente as mesmas categorias ou o mesmo número de landmarks.</p></div>
            </section>

            <section class="guide-section" id="action-units">
                <h2>Action Units (Unidades de Ação Facial)</h2>
                <p>Uma <strong>Action Unit (AU)</strong> descreve um movimento observável de uma região facial, de acordo com o <strong>Facial Action Coding System (FACS)</strong>. Por exemplo, uma AU pode representar a elevação de uma sobrancelha, o levantamento da bochecha ou a separação dos lábios.</p>
                <p>As AUs descrevem movimento facial; não são emoções. Uma expressão emocional pode envolver várias AUs ao mesmo tempo, e a mesma AU pode aparecer em diferentes contextos ou expressões.</p>
                <div class="guide-note"><p>Um valor elevado numa AU significa que o modelo encontrou maior evidência desse movimento facial. Não deve ser interpretado isoladamente como prova de uma emoção ou de uma intenção.</p></div>
            </section>

            <section class="guide-section" id="au-reference">
                <h2>Referência de Action Units</h2>
                <p>A referência oficial do Py-Feat relaciona cada AU com o nome FACS, a musculatura envolvida, expressões frequentemente associadas e os modelos que a suportam.</p>
                <ul>
                    <li><strong>AU1 - Inner Brow Raiser (Elevador interno da sobrancelha):</strong> movimento da parte interna das sobrancelhas.</li>
                    <li><strong>AU4 - Brow Lowerer (Abaixador da sobrancelha):</strong> aproximação e descida das sobrancelhas.</li>
                    <li><strong>AU6 - Cheek Raiser (Elevador da bochecha):</strong> elevação da bochecha e estreitamento da abertura dos olhos.</li>
                    <li><strong>AU12 - Lip Corner Puller (Puxador do canto dos lábios):</strong> elevação dos cantos da boca.</li>
                    <li><strong>AU15 - Lip Corner Depressor (Abaixador do canto dos lábios):</strong> descida dos cantos da boca.</li>
                    <li><strong>AU25 - Lips Part (Separação dos lábios):</strong> abertura dos lábios sem necessariamente envolver a mandíbula.</li>
                </ul>
                <p class="guide-links">Para a tabela completa e pesquisável, consulte a <a href="https://py-feat.org/pages/au_reference/" target="_blank">AU Reference oficial do Py-Feat</a>.</p>
                <div class="image-placeholder">Imagem recomendada: diagrama facial com as principais Action Units e as regiões musculares correspondentes.</div>
            </section>

            <section class="guide-section" id="emotions">
                <h2>Emoções e métricas faciais</h2>
                <p>Os modelos de emoção estimam categorias como <strong>Neutral, Happy, Sad, Surprise, Fear, Disgust</strong> e <strong>Anger</strong>. Estas estimativas são baseadas na aparência facial observada pelo modelo.</p>
                <p>As probabilidades ou valores apresentados devem ser tratados como indicadores para análise, e não como uma medição direta do estado emocional interno de uma pessoa. A interpretação deve considerar o contexto, a variabilidade individual e o desenho experimental.</p>
                <p>Dependendo do detector utilizado, também podem existir métricas como <strong>valence</strong> (valência) e <strong>arousal</strong> (ativação), <strong>gaze</strong> (direção do olhar) e <strong>head pose</strong> (orientação da cabeça). Apenas use estas variáveis quando estiverem presentes no dataset descarregado.</p>
            </section>

            <section class="guide-section" id="landmarks">
                <h2>Landmarks faciais e deteção do rosto</h2>
                <p><strong>Facial landmarks</strong> são pontos de referência colocados em posições específicas do rosto, como sobrancelhas, olhos, nariz, boca e contorno facial. Permitem descrever a geometria e o movimento do rosto ao longo do vídeo.</p>
                <p>A <strong>face detection</strong> localiza o rosto na imagem e pode fornecer uma caixa delimitadora, ou <strong>face box</strong>. Os landmarks e a face box são úteis para verificar se o rosto foi encontrado corretamente e para estudar movimentos faciais.</p>
                <div class="guide-note"><p>A numeração e a quantidade de landmarks dependem do modelo. Não compare diretamente coordenadas de modelos diferentes sem confirmar a sua convenção.</p></div>
                <div class="image-placeholder">Imagem recomendada: exemplo de um rosto com landmarks numerados e a face box assinalada.</div>
            </section>

            <section class="guide-section" id="sampling">
                <h2>Modos de amostragem</h2>
                <ul>
                    <li><strong>Every frame (A cada frame):</strong> preserva a maior resolução temporal e é indicado para fenómenos rápidos, mas produz datasets maiores e demora mais tempo.</li>
                    <li><strong>1 FPS - representative frame (Frame representativo):</strong> seleciona aproximadamente um frame por segundo. É mais rápido e compacto, mas pode não capturar movimentos breves.</li>
                    <li><strong>Custom sampling:</strong> define uma frequência-alvo em frames por segundo. A frequência real é aproximada e depende da taxa de frames do vídeo.</li>
                </ul>
                <div class="guide-note"><p>Escolha o modo com base na escala temporal da sua hipótese. Se a expressão relevante for breve, uma frequência customizada mais alta pode preservá-la.</p></div>
            </section>

            <section class="guide-section" id="metadata">
                <h2>Metadados experimentais e estímulos</h2>
                <p>Ative a opção de marcadores de estímulo quando quiser relacionar as deteções faciais com momentos específicos do protocolo experimental.</p>
                <ol>
                    <li>Indique quantos estímulos existem na sessão.</li>
                    <li>Defina o nome de cada estímulo.</li>
                    <li>Introduza os segundos de início e de fim relativamente ao início do vídeo.</li>
                    <li>Verifique se os intervalos correspondem ao protocolo; sobreposições são preservadas em colunas separadas.</li>
                </ol>
                <p>O dataset inclui uma coluna <strong>Stimulus_1</strong>, <strong>Stimulus_2</strong>, etc., para cada estímulo. Cada coluna contém o nome configurado dentro do respetivo intervalo; fora dele fica vazia. Numa sobreposição, as colunas dos dois estímulos são preenchidas.</p>
            </section>

            <section class="guide-section" id="good-practices">
                <h2>Boas práticas e limitações</h2>
                <ul>
                    <li>Faça uma inspeção visual de uma amostra dos vídeos e dos resultados antes da análise estatística.</li>
                    <li>Não trate uma previsão de emoção como prova definitiva de um estado emocional.</li>
                    <li>Registe o modo de amostragem, os intervalos de estímulo e qualquer decisão de exclusão de dados.</li>
                    <li>Tenha especial cuidado com oclusões, óculos, máscaras, iluminação, pose, baixa resolução e múltiplos rostos.</li>
                    <li>Evite comparar diretamente resultados obtidos com modelos ou versões diferentes sem documentar essa diferença.</li>
                    <li>Consulte a documentação oficial e as licenças dos modelos antes de reutilizar os resultados noutros contextos.</li>
                </ul>
                <div class="guide-note"><p>O Py-Feat recomenda verificar sempre as deteções. Modelos de visão computacional podem produzir resultados inesperados em vídeos reais, mesmo quando funcionam bem em benchmarks.</p></div>
                <p class="guide-links">Referências: <a href="https://py-feat.org/" target="_blank">Py-Feat</a> | <a href="https://py-feat.org/pages/models/" target="_blank">Models</a> | <a href="https://py-feat.org/pages/faqs/" target="_blank">FAQs and known issues</a></p>
            </section>
        </div>
        """,
        "en": """
        <nav class="guide-nav" aria-label="Guide navigation">
            <h2>In this guide</h2>
            <a href="#getting-started">Getting Started</a>
            <a href="#preparing-videos">Preparing Your Videos</a>
            <a href="#sampling">Sampling Modes</a>
            <a href="#metadata">Experimental Metadata and Stimuli</a>
            <a href="#landmarks">Facial Landmarks and Face Detection</a>
            <a href="#action-units">Action Units</a>
            <a href="#au-reference">AU Reference</a>
            <a href="#emotions">Emotions and Facial Metrics</a>
            <a href="#understanding-output">Understanding the Output Dataset</a>
            <a href="#interpretation"><span class="guide-nav-warning" aria-hidden="true">&#9888;</span>Interpreting the Data</a>
            <a href="#good-practices">Good Practices and Limitations</a>
        </nav>
        <div class="guide-content">
            <section class="guide-section" id="getting-started">
                <h2>Getting Started</h2>
                <p>This interface transforms videos into structured data about facial activity over time. No programming is required: follow the steps in the analysis workspace.</p>
                <ol>
                    <li>Upload one or more videos in a supported format.</li>
                    <li>Assign a <strong>Subject ID</strong> to each video.</li>
                    <li>Choose the sampling mode that fits your research question.</li>
                    <li>Optionally define the time windows for experimental stimuli.</li>
                    <li>Select <strong>Extract data</strong>, wait for processing, and download each Parquet file.</li>
                </ol>
                <div class="guide-note"><p>The result is a dataset where each row represents a facial observation at a frame or within a one-second interval.</p></div>
            </section>

            <section class="guide-section" id="preparing-videos">
                <h2>Preparing Your Videos</h2>
                <p>For more consistent detections, use videos where the face is visible for most of the recording and occupies a reasonable area of the image.</p>
                <ul>
                    <li>Avoid very dark lighting, strong backlighting, and abrupt exposure changes.</li>
                    <li>Avoid very small, partially occluded, or extremely rotated faces.</li>
                    <li>Check whether the video contains only the relevant participants, since multiple faces may be detected.</li>
                    <li>Keep quality and resolution similar across videos that will be compared.</li>
                </ul>
            </section>

            <section class="guide-section" id="sampling">
                <h2>Sampling Modes</h2>
                <ul>
                    <li><strong>Every frame:</strong> preserves the highest temporal resolution and is useful for fast phenomena, but creates larger datasets and takes longer.</li>
                    <li><strong>1 FPS - representative frame:</strong> selects approximately one frame per second. It is faster and more compact, but may miss brief movements.</li>
                    <li><strong>Custom sampling:</strong> sets a target frequency in frames per second. The actual frequency is approximate and depends on the video's frame rate.</li>
                </ul>
                <div class="guide-note"><p>Choose the mode based on the time scale of your hypothesis. If a relevant expression is brief, a higher custom frequency may preserve it.</p></div>
            </section>

            <section class="guide-section" id="metadata">
                <h2>Experimental Metadata and Stimuli</h2>
                <p>Enable stimulus markers when you want to relate facial detections to specific moments in the experimental protocol.</p>
                <ol>
                    <li>Indicate how many stimuli are in the session.</li>
                    <li>Define the name of each stimulus.</li>
                    <li>Enter the start and end seconds relative to the beginning of the video.</li>
                    <li>Check that the intervals match the protocol; overlaps are preserved in separate columns.</li>
                </ol>
                <p>The dataset includes one column, <strong>Stimulus_1</strong>, <strong>Stimulus_2</strong>, etc., for each stimulus. Each column contains its configured name inside that interval and is blank outside it. When intervals overlap, both stimulus columns are populated.</p>
            </section>

            <section class="guide-section" id="landmarks">
                <h2>Facial Landmarks and Face Detection</h2>
                <p><strong>Facial landmarks</strong> are reference points placed at specific facial locations, such as the eyebrows, eyes, nose, mouth, and facial outline. They describe facial geometry and movement over time.</p>
                <p><strong>Face detection</strong> locates the face in the image and may provide a bounding box, or <strong>face box</strong>. Landmarks and face boxes help verify whether the face was detected correctly and support the study of facial movement.</p>
            </section>

            <section class="guide-section" id="action-units">
                <h2>Action Units</h2>
                <p>An <strong>Action Unit (AU)</strong> describes an observable movement in a facial region according to the <strong>Facial Action Coding System (FACS)</strong>. For example, an AU may represent eyebrow raising, cheek raising, or lip separation.</p>
                <p>AUs describe facial movement; they are not emotions. An emotional expression can involve several AUs at once, and the same AU can occur in different contexts or expressions.</p>
                <div class="guide-note"><p>A higher AU value means that the model found stronger evidence for that facial movement. It should not be interpreted alone as proof of an emotion or intention.</p></div>
            </section>

            <section class="guide-section" id="au-reference">
                <h2>AU Reference</h2>
                <p>The official Py-Feat reference connects each AU with its FACS name, involved musculature, commonly associated expressions, and supporting models.</p>
                <ul>
                    <li><strong>AU1 - Inner Brow Raiser:</strong> movement of the inner part of the eyebrows.</li>
                    <li><strong>AU4 - Brow Lowerer:</strong> drawing the eyebrows together and downward.</li>
                    <li><strong>AU6 - Cheek Raiser:</strong> raising the cheeks and narrowing the eye opening.</li>
                    <li><strong>AU12 - Lip Corner Puller:</strong> raising the corners of the mouth.</li>
                    <li><strong>AU15 - Lip Corner Depressor:</strong> lowering the corners of the mouth.</li>
                    <li><strong>AU25 - Lips Part:</strong> separating the lips without necessarily involving the jaw.</li>
                </ul>
                <p class="guide-links">For the complete searchable table, see the <a href="https://py-feat.org/pages/au_reference/" target="_blank">official Py-Feat AU Reference</a>.</p>
                <img src="data:image/jpeg;base64,AU_REFERENCE_IMAGE" alt="Facial Action Unit reference diagram" style="display: block; max-width: min(100%, 520px); height: auto; margin: 1rem auto 0; border-radius: 7px;">
            </section>

            <section class="guide-section" id="emotions">
                <h2>Emotions and Facial Metrics</h2>
                <p>This application uses Py-Feat's default emotion model to estimate facial-expression categories such as <strong>Neutral, Happy, Sad, Surprise, Fear, Disgust</strong>, and <strong>Anger</strong>. These estimates are based on the facial appearance observed in each video frame.</p>
                <p>Probabilities or values should be treated as indicators for analysis, not as direct measurements of a person's internal emotional state. Interpretation should consider context, individual variability, and the experimental design.</p>
            </section>

            <section class="guide-section" id="understanding-output">
                <h2>Understanding the Output Dataset</h2>
                <p>The Parquet file contains Py-Feat predictions and identifiers added by this interface.</p>
                <ul>
                    <li><strong>subject_id:</strong> participant identifier assigned during upload.</li>
                    <li><strong>input_file:</strong> source video name.</li>
                    <li><strong>frame:</strong> frame number in the original video.</li>
                    <li><strong>time_sec:</strong> approximate frame time in seconds in Every frame mode.</li>
                    <li><strong>approx_time:</strong> readable time representation, such as 00:12.</li>
                    <li><strong>second, time_start_sec, and time_end_sec:</strong> time interval created when data are aggregated by second.</li>
                    <li><strong>Stimulus_1, Stimulus_2, etc.:</strong> one column per stimulus; a row can be marked in multiple columns when intervals overlap.</li>
                    <li><strong>Action Units, emotions, landmarks, and other metrics:</strong> facial predictions produced by the Py-Feat model.</li>
                </ul>
                <div class="guide-note"><p>Always check the column names before analysis.</p></div>
            </section>

            <section class="guide-section" id="interpretation">
                <h2>Interpreting the Data</h2>

                <h3>1. Action Units (AUs): Separating Signal from Noise</h3>
                <ul>
                    <li><strong>Empirical thresholds:</strong> Treat AU readings below <strong>0.10–0.15</strong> as possible baseline noise. Calibrate this range against your detector, video quality, and participant baseline; do not apply it as a universal cutoff.</li>
                    <li><strong>Anatomical co-occurrence:</strong> Look for anatomically plausible AU clusters. For example, AU6 with AU12 can support a smile interpretation. An isolated AU may reflect noise, a social gesture, or another movement.</li>
                    <li><strong>Baseline deviation:</strong> Compare peaks with the same participant's resting, neutral face. A change from that baseline is often more informative than the raw value.</li>
                    <li><strong>Temporal persistence:</strong> Filter isolated, single-frame spikes that may come from tracking artifacts. Look for movements lasting at least <strong>3–5 consecutive frames</strong> (roughly <strong>100–200 ms</strong>, depending on frame rate and sampling).</li>
                </ul>

                <h3>2. Emotion Probabilities: Value vs. Misinterpretation</h3>
                <ul>
                    <li><strong>Flat distributions:</strong> If several emotion scores fall between <strong>0.20 and 0.30</strong>, treat the result as ambiguous rather than assigning a clear emotion. A single dominant score, such as <strong>above 0.70</strong>, is a stronger model signal, not proof of a person's psychological state.</li>
                    <li><strong>Relative change:</strong> Focus on changes across conditions, such as a rise from <strong>0.05 to 0.45</strong>, rather than treating an absolute probability as a definitive measurement.</li>
                    <li><strong>Contextual synchrony:</strong> Align emotion scores with the exact stimulus timestamp. Compare changes before, during, and after the stimulus.</li>
                </ul>

                <h3>3. Advanced Tracking and Geometric Metrics (Critical Sanity Checks)</h3>
                <ul>
                    <li><strong>Face detection confidence (<code>Face_Score</code>):</strong> Flag or exclude data points below <strong>0.90–0.95</strong> for closer review. Choose a cutoff that fits your data and analysis goals.</li>
                    <li><strong>Head pose (<code>Pitch</code>, <code>Roll</code>, <code>Yaw</code>):</strong> Treat rotations above roughly <strong>15–20 degrees</strong> as a reason to inspect the frame. Larger rotations can distort visible facial geometry and affect AU estimates.</li>
                    <li><strong>Landmark jitter and variance:</strong> Look for sudden, implausible jumps in landmark coordinates. These often indicate tracking failure, not a micro-expression.</li>
                    <li><strong>Discontinuities and NaNs:</strong> Identify frames where the face is occluded or missing. Preserve these gaps as missing data; do not interpolate across them without a justified method.</li>
                </ul>

                <h3>Summary Checklist for Data Cleaning</h3>
                <div class="guide-table-wrap">
                    <table class="guide-table">
                        <thead>
                            <tr>
                                <th>Data Point</th>
                                <th>Discard as Noise if...</th>
                                <th>Keep as Valuable Signal if...</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>AU Values</strong></td>
                                <td>Readings stay below approximately <strong>0.10–0.15</strong>, appear as isolated spikes, or lack a plausible AU cluster.</td>
                                <td>Values rise above the participant's neutral baseline, form a plausible cluster, and persist across consecutive frames.</td>
                            </tr>
                            <tr>
                                <td><strong>Emotion Values</strong></td>
                                <td>Scores are spread across several categories around <strong>0.20–0.30</strong>, with no clear dominant category.</td>
                                <td>One category becomes strongly dominant, or its score changes meaningfully across the stimulus context. Treat it as a model estimate, not a psychological fact.</td>
                            </tr>
                            <tr>
                                <td><strong>Face_Score</strong></td>
                                <td>Confidence falls below your chosen review threshold, typically <strong>0.90–0.95</strong>.</td>
                                <td>Confidence meets your threshold and the face is visibly tracked well.</td>
                            </tr>
                            <tr>
                                <td><strong>Head Pose Angles</strong></td>
                                <td><code>Pitch</code>, <code>Roll</code>, or <code>Yaw</code> exceed roughly <strong>15–20 degrees</strong>, or change abruptly.</td>
                                <td>Pose remains stable enough for reliable facial visibility and comparison.</td>
                            </tr>
                            <tr>
                                <td><strong>Landmark Coordinates</strong></td>
                                <td>Coordinates jump unnaturally, become discontinuous, or are missing during occlusion.</td>
                                <td>Coordinates move smoothly and consistently with visible facial motion.</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </section>

            <section class="guide-section" id="good-practices">
                <h2>Good Practices and Limitations</h2>
                <ul>
                    <li>Visually inspect a sample of the videos and results before statistical analysis.</li>
                    <li>Do not treat an emotion prediction as definitive evidence of an emotional state.</li>
                    <li>Record the sampling mode, stimulus intervals, and any data-exclusion decisions.</li>
                    <li>Take extra care with occlusion, glasses, masks, lighting, pose, low resolution, and multiple faces.</li>
                    <li>Consult the official documentation and model licenses before reusing results in other contexts.</li>
                </ul>
                <div class="guide-note"><p>Py-Feat recommends checking detections carefully. Computer-vision models can produce unexpected results in real-world videos, even when they perform well on benchmarks.</p></div>
                <p class="guide-links">References: <a href="https://py-feat.org/" target="_blank">Py-Feat</a> | <a href="https://py-feat.org/pages/models/" target="_blank">Models</a> | <a href="https://py-feat.org/pages/faqs/" target="_blank">FAQs and known issues</a></p>
            </section>
        </div>
        """,
    }
    intro_guide["en"] = intro_guide["en"].replace("AU_REFERENCE_IMAGE", au_reference_image)

    st.markdown(
        f"""
        <section class="intro-hero">
            <h1>{t("intro_page_title")}</h1>
            <p class="hero-subtitle">{t("intro_page_body")}</p>
            <p class="hero-powered-by">{t("intro_page_powered_by")}</p>
        </section>

        """,
        unsafe_allow_html=True,
    )
    guide_navigation, guide_content = intro_guide[language].split('<div class="guide-content">', 1)
    with st.container(key="intro-guide-layout"):
        navigation_column, content_column = st.columns([0.72, 2.28], gap="large")
        with navigation_column:
            with st.container(key="intro-guide-sticky"):
                st.html(guide_navigation)
                with st.container(key="intro-nav-workspace-button"):
                    navigation_workspace_clicked = st.button(
                        t("go_workspace"),
                        type="primary",
                        use_container_width=True,
                        key="intro_workspace_navigation_button",
                    )
        with content_column:
            st.html(f'<div class="guide-content">{guide_content}')
    footer_workspace_clicked = st.button(
        t("go_workspace"),
        type="primary",
        use_container_width=True,
        key="intro_workspace_footer_button",
    )
    if navigation_workspace_clicked or footer_workspace_clicked:
        st.session_state["app_page"] = "workspace"
        st.rerun()
    st.stop()


sampling_labels = {
    t("all_frames"): "all_frames",
    t("fast_fps"): "fast_fps",
    t("custom_sampling"): "custom_fps",
}

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


def calculate_skip_frames(video_fps, target_fps):
    return max(1, int(round(float(video_fps) / float(target_fps))))


def estimate_processing_seconds(sampled_rows, sampling_mode, runtime_history):
    mode_history = runtime_history.get(sampling_mode)
    if mode_history and mode_history["rows"] > 0:
        seconds_per_row = mode_history["seconds"] / mode_history["rows"]
        return float(sampled_rows) * seconds_per_row

    measured_rows = sum(history["rows"] for history in runtime_history.values() if history["rows"] > 0)
    measured_seconds = sum(history["seconds"] for history in runtime_history.values() if history["rows"] > 0)
    if measured_rows > 0:
        return float(sampled_rows) * measured_seconds / measured_rows

    return float(sampled_rows) * BASELINE_SECONDS_PER_FRAME


def has_runtime_measurements(runtime_history):
    return any(history["rows"] > 0 for history in runtime_history.values())


def iter_with_progress(iterable, on_progress):
    total = len(iterable)
    for batch_number, item in enumerate(iterable, start=1):
        yield item
        on_progress(batch_number, total)


def report_video_progress(
    batch_number,
    total_batches,
    *,
    progress_bar,
    completed_rows,
    expected_rows,
    total_rows,
    video_started_at,
    current,
    total,
    name,
):
    inference_progress = batch_number / total_batches if total_batches else 0.0
    video_progress = min(inference_progress * 0.95, 0.95)
    work_done = completed_rows + expected_rows * video_progress
    overall_progress = min(work_done / max(total_rows, 1), 0.999)
    rows_observed = expected_rows * inference_progress
    elapsed = time.perf_counter() - video_started_at
    remaining_rows = max(total_rows - work_done, 0)
    remaining_seconds = elapsed * remaining_rows / rows_observed if rows_observed else 0
    remaining_total = max(0, int(round(remaining_seconds)))
    progress_bar.progress(
        overall_progress,
        text=t(
            "processing_video_progress",
            current=current,
            total=total,
            name=name,
            percent=video_progress * 100,
            minutes=remaining_total // 60,
            seconds=remaining_total % 60,
        ),
    )


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
    """Return per-frame rows or aggregate the 1 FPS representative samples."""
    if sampling_mode in {"all_frames", "custom_fps"}:
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

    for index, stimulus in enumerate(stimuli, start=1):
        start = float(stimulus["inicio"])
        end = float(stimulus["fim"])
        if end < start:
            start, end = end, start
        matches = row_times.ge(start) & row_times.lt(end)
        labels = pd.Series("", index=df.index, dtype="object")
        labels.loc[matches] = stimulus["nome"]
        df[f"Stimulus_{index}"] = labels
    return df


def dataframe_to_parquet_bytes(df):
    buffer = BytesIO()
    df.to_parquet(buffer, engine="pyarrow", compression="snappy", index=False)
    return buffer.getvalue()


def visualization_time(df):
    for column in ("time_sec", "time_start_sec", "frame"):
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce"), column
    return pd.Series(range(len(df)), index=df.index, dtype="float64"), "sample"


def visualization_columns(df, category):
    if category == "Action Units":
        return sorted(
            (column for column in df.columns if re.fullmatch(r"AU\d+", str(column))),
            key=lambda column: int(re.search(r"\d+", str(column)).group()),
        )
    if category == "Emotions":
        preferred = ["neutral", "happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    else:
        preferred = ["Pitch", "Roll", "Yaw"]
    return [column for column in preferred if column in df.columns]


def style_visualization(fig, height=340):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#202c33",
        plot_bgcolor="#202c33",
        font_color="#edf3f1",
        height=height,
        margin={"l": 20, "r": 20, "t": 42, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_xaxes(gridcolor="#394951", zerolinecolor="#536b73")
    fig.update_yaxes(gridcolor="#394951", zerolinecolor="#536b73")
    return fig


def show_chart_explanation(chart_key, description_key):
    with st.popover(
        t("chart_help_button"),
        icon=":material/info:",
        key=f"chart_help_{chart_key}",
    ):
        st.write(t(description_key))


def add_stimulus_spans(fig, stimuli):
    for stimulus in stimuli:
        start = float(stimulus["inicio"])
        end = float(stimulus["fim"])
        if end < start:
            start, end = end, start
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="#f09a52",
            opacity=0.12,
            line_width=0,
            annotation_text=stimulus["nome"],
            annotation_position="top left",
        )
    return fig


def compare_stimulus_windows(results, metric):
    records = []
    for result in results.values():
        df = result["dataframe"]
        if metric not in df.columns:
            continue
        times, _ = visualization_time(df)
        values = pd.to_numeric(df[metric], errors="coerce")
        for stimulus in result.get("stimuli", []):
            start = float(stimulus["inicio"])
            end = float(stimulus["fim"])
            if end < start:
                start, end = end, start
            before = values.loc[times.ge(start - 5) & times.lt(start)].mean()
            during = values.loc[times.ge(start) & times.lt(end)].mean()
            if pd.isna(before) or pd.isna(during):
                continue
            case = f"{result['subject_id']} · {df['input_file'].iloc[0]} · {stimulus['nome']}"
            for period, value in (("Before", before), ("During", during)):
                records.append({
                    "case": case,
                    "subject": result["subject_id"],
                    "stimulus": stimulus["nome"],
                    "period": period,
                    "value": value,
                })
    return pd.DataFrame.from_records(records)


def event_aligned_signals(results, metric, stimulus_name, padding=5.0):
    records = []
    for result in results.values():
        df = result["dataframe"]
        if metric not in df.columns or df.empty:
            continue
        stimulus = next(
            (item for item in result.get("stimuli", []) if item["nome"] == stimulus_name),
            None,
        )
        if stimulus is None:
            continue

        start = float(stimulus["inicio"])
        end = float(stimulus["fim"])
        if end < start:
            start, end = end, start
        times, time_column = visualization_time(df)
        values = pd.to_numeric(df[metric], errors="coerce")
        window = times.ge(start - padding) & times.le(end + padding)
        case = f"{result['subject_id']} · {df['input_file'].iloc[0]}"
        for relative_time, value in zip(times.loc[window] - start, values.loc[window]):
            if pd.isna(relative_time) or pd.isna(value):
                continue
            records.append({
                "relative_time": float(relative_time),
                "value": float(value),
                "recording": case,
                "time_column": time_column,
                "stimulus_duration": end - start,
            })
    return pd.DataFrame.from_records(records)


def pearson_correlation(left_values, right_values):
    if len(left_values) < 4:
        return None
    left_mean = sum(left_values) / len(left_values)
    right_mean = sum(right_values) / len(right_values)
    left_centered = [value - left_mean for value in left_values]
    right_centered = [value - right_mean for value in right_values]
    left_sum = sum(value * value for value in left_centered)
    right_sum = sum(value * value for value in right_centered)
    if left_sum == 0 or right_sum == 0:
        return None
    covariance = sum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered)
    )
    return covariance / math.sqrt(left_sum * right_sum)


def stimulus_similarity_matrix(results, metric, stimulus_name, phase_bins=20):
    recording_series = {}
    for result in results.values():
        df = result["dataframe"]
        if metric not in df.columns or df.empty:
            continue
        stimulus = next(
            (item for item in result.get("stimuli", []) if item["nome"] == stimulus_name),
            None,
        )
        if stimulus is None:
            continue

        start = float(stimulus["inicio"])
        end = float(stimulus["fim"])
        if end < start:
            start, end = end, start
        if end <= start:
            continue
        times, _ = visualization_time(df)
        values = pd.to_numeric(df[metric], errors="coerce")
        within_window = times.ge(start) & times.lt(end) & values.notna()
        phase_data = pd.DataFrame({
            "phase": (times.loc[within_window] - start) / (end - start),
            "value": values.loc[within_window],
        })
        if phase_data.empty:
            continue
        phase_data["phase_bin"] = (phase_data["phase"] * phase_bins).astype(int).clip(0, phase_bins - 1)
        binned = phase_data.groupby("phase_bin")["value"].mean().to_dict()
        if len(binned) < 4:
            continue
        case = f"{result['subject_id']} · {df['input_file'].iloc[0]}"
        recording_series[case] = binned

    labels = list(recording_series)
    correlations = []
    shared_bin_counts = []
    for row_label in labels:
        correlation_row = []
        count_row = []
        for column_label in labels:
            common_bins = sorted(set(recording_series[row_label]) & set(recording_series[column_label]))
            left_values = [recording_series[row_label][phase] for phase in common_bins]
            right_values = [recording_series[column_label][phase] for phase in common_bins]
            correlation_row.append(pearson_correlation(left_values, right_values))
            count_row.append(len(common_bins))
        correlations.append(correlation_row)
        shared_bin_counts.append(count_row)

    return (
        pd.DataFrame(correlations, index=labels, columns=labels, dtype="float64"),
        pd.DataFrame(shared_bin_counts, index=labels, columns=labels, dtype="int64"),
    )


st.subheader(t("upload"))
uploaded_files = st.file_uploader(t("upload_help"), type=['mp4', 'avi', 'mov'],
    accept_multiple_files=True)

video_subjects = {}
total_frames_to_process = 0
videos_info = {}
runtime_history = st.session_state.setdefault("processing_runtime_history", {})

if uploaded_files:
    st.markdown(t("subject_mapping"))
    for file in uploaded_files:
      
        fps, frames = get_video_metadata(file.name, file.read())
        videos_info[file.name] = {"fps": fps, "frames": frames}
        
        col_file, col_id = st.columns([1, 2])
        with col_file:
            st.write(t("file_info", name=file.name, frames=frames, fps=fps))
        with col_id:
            video_subjects[file.name] = st.text_input(t("subject_id"), key=f"id_{file.name}", placeholder=t("subject_placeholder"), label_visibility="collapsed")

with st.sidebar:
    st.header(t("controls"))
    
    sampling_rate = st.selectbox(
        t("sampling_label"),
        options=list(sampling_labels),
        index=1,
        key="sampling_selection",
        accept_new_options=False,
    )
    st.html(
        """
        <script>
        (() => {
            const lockSamplingInput = () => {
                const expandButton = document.querySelector('button[data-testid="stExpandSidebarButton"]');
                if (expandButton) expandButton.setAttribute("aria-label", "Open Extraction Controls");
                const input = document.querySelector(
                    '.st-key-sampling_selection [data-testid="stSelectbox"] input'
                );
                if (!input || input.dataset.noTextEntry === "true") return;
                input.readOnly = true;
                input.setAttribute("inputmode", "none");
                input.dataset.noTextEntry = "true";
                input.addEventListener("keydown", (event) => {
                    if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
                        event.preventDefault();
                        event.stopImmediatePropagation();
                    }
                }, true);
                input.addEventListener("beforeinput", (event) => {
                    event.preventDefault();
                    event.stopImmediatePropagation();
                }, true);
            };
            lockSamplingInput();
            if (window.samplingDropdownObserver) window.samplingDropdownObserver.disconnect();
            window.samplingDropdownObserver = new MutationObserver(lockSamplingInput);
            window.samplingDropdownObserver.observe(document.body, {
                childList: true,
                subtree: true,
            });
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )
    sampling_mode = sampling_labels[sampling_rate]
    custom_sampling_fps = 1.0
    if sampling_mode == "custom_fps":
        custom_sampling_fps = st.number_input(
            t("custom_sampling_fps"),
            min_value=0.1,
            value=1.0,
            step=0.1,
            key="custom_sampling_frequency",
        )
        st.caption(t("custom_sampling_help"))
    elif sampling_mode == "fast_fps":
        st.caption(t("sampling_help"))
    sampling_display = (
        f"{t('custom_sampling')} ({custom_sampling_fps:g} FPS)"
        if sampling_mode == "custom_fps"
        else sampling_rate
    )
    
    st.markdown("---")
    st.subheader(t("time_prediction"))
    
    if uploaded_files:
        frames_to_analyze = 0
        output_rows = 0
        for file_name, info in videos_info.items():
            vid_fps = info["fps"]
            vid_frames = info["frames"]

            if sampling_mode == "all_frames":
                skip = 1
                effective_rows = vid_frames
            else:
                target_fps = custom_sampling_fps if sampling_mode == "custom_fps" else 1.0
                skip = calculate_skip_frames(vid_fps, target_fps)
                effective_rows = int(math.ceil(vid_frames / skip)) if vid_frames else 0

            videos_info[file_name]["skip_frames"] = skip
            videos_info[file_name]["sampling_mode"] = sampling_mode
            videos_info[file_name]["effective_rows"] = effective_rows
            frames_to_analyze += effective_rows
            output_rows += effective_rows

        st.metric(label=t("frames_analyzed"), value=f"{frames_to_analyze}")
        st.metric(label=t("output_rows"), value=f"{output_rows}")
        estimated_seconds = estimate_processing_seconds(frames_to_analyze, sampling_mode, runtime_history)
        rounded_seconds = max(0, int(round(estimated_seconds)))
        minutes = rounded_seconds // 60
        seconds = rounded_seconds % 60
        st.metric(
            label=t("estimated_time"),
            value=f"{minutes}m {seconds}s",
        )
        if not has_runtime_measurements(runtime_history):
            st.caption(t("estimated_time_disclaimer"))
    else:
        st.info(t("upload_for_estimate"))

st.markdown("---")

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

st.session_state["stimuli_timestamps"] = estimulos_timestamps
stimuli_for_processing = st.session_state.get("stimuli_timestamps", [])
st.markdown("---")


st.subheader(t("processing"))
processed_results = st.session_state.setdefault("processed_results", {})
if not processed_results:
    st.info(t("visualizations_after_processing"))

if st.button(t("start_extraction"), type="primary"):
    
    missing_ids = [vid for vid, subj_id in video_subjects.items() if not subj_id.strip()]
    
    if not uploaded_files:
        st.error(t("missing_video"))
    elif missing_ids:
        st.warning(t("missing_ids"))
    else:
        st.success(t("validated"))
        processed_results.clear()
        
        with st.spinner(t("loading")):
            detector = load_detector()
        
       
        progress_bar = st.progress(0, text=t("preparing"))
        total_files = len(uploaded_files)
        completed_rows = 0
        
        for idx, file in enumerate(uploaded_files):
            subject_id = video_subjects[file.name]
            skip_frames = videos_info[file.name]["skip_frames"]
            sampling_mode = videos_info[file.name].get("sampling_mode", sampling_mode)
            expected_rows = videos_info[file.name]["effective_rows"]

            progress_text = t("processing_video", current=idx + 1, total=total_files, name=file.name, subject=subject_id)
            progress_bar.progress(idx / total_files, text=progress_text)
            video_started_at = time.perf_counter()
            progress_callback = partial(
                report_video_progress,
                progress_bar=progress_bar,
                completed_rows=completed_rows,
                expected_rows=expected_rows,
                total_rows=frames_to_analyze,
                video_started_at=video_started_at,
                current=idx + 1,
                total=total_files,
                name=file.name,
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                file.seek(0)
                temp_video.write(file.read())
                temp_video_path = temp_video.name

            try:
                with st.spinner(t("extracting")):
                    with DETECT_VIDEO_PROGRESS_LOCK:
                        original_tqdm = feat_detector.tqdm
                        feat_detector.tqdm = lambda iterable: iter_with_progress(iterable, progress_callback)
                        try:
                            with torch.no_grad():
                                video_prediction = detector.detect_video(
                                    temp_video_path,
                                    skip_frames=skip_frames,
                                    batch_size=4,
                                )
                        finally:
                            feat_detector.tqdm = original_tqdm
                    df_raw = video_prediction.to_pandas() if hasattr(video_prediction, 'to_pandas') else video_prediction

                    output_df = prepare_frame_dataframe(df_raw, videos_info[file.name]["fps"], sampling_mode)
                    if stimuli_for_processing:
                        output_df = apply_stimulus_metadata(output_df, stimuli_for_processing)
                    output_df["subject_id"] = subject_id
                    output_df["input_file"] = file.name

                    parquet_bytes = dataframe_to_parquet_bytes(output_df)
                    processed_results[file.name] = {
                        "subject_id": subject_id,
                        "dataframe": output_df,
                        "parquet_bytes": parquet_bytes,
                        "file_name": f"{subject_id}_{os.path.splitext(file.name)[0]}_pyfeat.parquet",
                        "expected_rows": expected_rows,
                        "stimuli": [dict(stimulus) for stimulus in stimuli_for_processing],
                    }
                    elapsed_seconds = time.perf_counter() - video_started_at
                    mode_history = runtime_history.setdefault(sampling_mode, {"rows": 0, "seconds": 0.0})
                    mode_history["rows"] += expected_rows
                    mode_history["seconds"] += elapsed_seconds
                    completed_rows += expected_rows
                    progress_bar.progress(
                        min(completed_rows / max(frames_to_analyze, 1), 1.0),
                        text=t(
                            "processing_video_progress",
                            current=idx + 1,
                            total=total_files,
                            name=file.name,
                            percent=100,
                            minutes=0,
                            seconds=0,
                        ),
                    )

            finally:
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
        
        
        progress_bar.progress(1.0, text=t("finished"))
        st.rerun()

for result in processed_results.values():
    result_df = result["dataframe"]
    result_input = result_df["input_file"].iloc[0] if not result_df.empty else result["file_name"]
    result_key = result["file_name"]
    expected_rows = result.get("expected_rows", len(result_df))
    stimuli = result.get("stimuli", [])
    time_values, time_column = visualization_time(result_df)
    time_label = "Time (seconds)" if time_column in {"time_sec", "time_start_sec"} else "Frame"

    st.success(t("completed_file", name=result_input))
    st.write(t("preview", rows=result_df.shape[0], mode=sampling_display))

    with st.expander(t("visualizations"), expanded=True):
        quality_tab, signals_tab, heatmap_tab = st.tabs([
            t("quality_tab"),
            t("signals_tab"),
            t("heatmap_tab"),
        ])

        with quality_tab:
            face_scores = (
                pd.to_numeric(result_df["FaceScore"], errors="coerce")
                if "FaceScore" in result_df.columns
                else None
            )
            threshold = st.slider(
                t("confidence_threshold"),
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.05,
                key=f"confidence_threshold_{result_key}",
            )
            detected_frames = result_df["frame"].nunique() if "frame" in result_df.columns else len(result_df)
            frame_coverage = min(detected_frames / expected_rows, 1.0) if expected_rows else 0.0
            quality_columns = st.columns(4)
            quality_columns[0].metric(t("output_rows_metric"), f"{len(result_df):,}")
            quality_columns[1].metric(
                t("frames_with_detections"),
                f"{detected_frames:,} / {expected_rows:,}",
                delta=f"{frame_coverage:.0%}",
            )
            quality_columns[2].metric(
                t("median_face_score"),
                f"{face_scores.median():.3f}" if face_scores is not None and face_scores.notna().any() else "N/A",
            )
            flagged_scores = int(face_scores.lt(threshold).sum()) if face_scores is not None else 0
            quality_columns[3].metric(
                t("flagged_face_scores"),
                f"{flagged_scores:,}" if face_scores is not None else "N/A",
            )
            st.caption(t("quality_threshold_note"))

            if face_scores is None:
                st.info(t("no_face_score"))
            else:
                confidence_data = pd.DataFrame({"time": time_values, "FaceScore": face_scores}).dropna()
                if len(confidence_data) > 3000:
                    confidence_data = confidence_data.iloc[::math.ceil(len(confidence_data) / 3000)]
                confidence_fig = px.scatter(
                    confidence_data,
                    x="time",
                    y="FaceScore",
                    title="Face detection confidence over time",
                    labels={"time": time_label},
                    hover_data={"FaceScore": ":.3f"},
                )
                confidence_fig.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color="#ef5350",
                    annotation_text=f"Review threshold: {threshold:.2f}",
                )
                style_visualization(confidence_fig)
                st.plotly_chart(
                    confidence_fig,
                    width="stretch",
                    key=f"confidence_chart_{result_key}",
                )
                show_chart_explanation(f"quality_{result_key}", "chart_face_score_help")

        with signals_tab:
            signal_categories = [
                (t("au_group"), "Action Units"),
                (t("emotion_group"), "Emotions"),
                (t("pose_group"), "Head pose"),
            ]
            available_categories = [
                (label, category)
                for label, category in signal_categories
                if visualization_columns(result_df, category)
            ]
            if not available_categories:
                st.info(t("no_face_score"))
            else:
                selected_label = st.radio(
                    t("signal_category"),
                    options=[label for label, _ in available_categories],
                    horizontal=True,
                    key=f"signal_category_{result_key}",
                )
                category = dict(available_categories)[selected_label]
                signal_columns = visualization_columns(result_df, category)
                default_signals = signal_columns[:2] if category == "Action Units" else signal_columns[:1]
                selected_signals = st.multiselect(
                    t("select_signals"),
                    options=signal_columns,
                    default=default_signals,
                    key=f"signal_columns_{result_key}",
                )
                if selected_signals:
                    signal_data = result_df[selected_signals].apply(pd.to_numeric, errors="coerce").copy()
                    signal_data["time"] = time_values.to_numpy()
                    signal_data = signal_data.dropna(subset=["time"])
                    if len(signal_data) > 5000:
                        signal_data = signal_data.iloc[::math.ceil(len(signal_data) / 5000)]
                    signal_fig = px.line(
                        signal_data,
                        x="time",
                        y=selected_signals,
                        title="Selected model outputs over time",
                        labels={"time": time_label, "value": "Model output"},
                    )
                    add_stimulus_spans(signal_fig, stimuli)
                    style_visualization(signal_fig, height=390)
                    st.plotly_chart(
                        signal_fig,
                        width="stretch",
                        key=f"signals_chart_{result_key}",
                    )
                    show_chart_explanation(f"signals_{result_key}", "chart_signals_help")

        with heatmap_tab:
            au_columns = visualization_columns(result_df, "Action Units")
            if not au_columns:
                st.info(t("no_face_score"))
            else:
                heatmap_data = result_df[au_columns].apply(pd.to_numeric, errors="coerce")
                heatmap_times = time_values
                if len(heatmap_data) > 1500:
                    step = math.ceil(len(heatmap_data) / 1500)
                    heatmap_data = heatmap_data.iloc[::step]
                    heatmap_times = heatmap_times.iloc[::step]
                heatmap_fig = go.Figure(
                    data=go.Heatmap(
                        z=heatmap_data.to_numpy(dtype=float).T,
                        x=heatmap_times,
                        y=au_columns,
                        colorscale="Viridis",
                        colorbar={"title": "AU"},
                        hovertemplate="%{y}<br>Time: %{x:.2f}<br>Value: %{z:.3f}<extra></extra>",
                    )
                )
                add_stimulus_spans(heatmap_fig, stimuli)
                heatmap_fig.update_layout(
                    title="Action Unit activity across time",
                    xaxis_title=time_label,
                    yaxis_title="Action Unit",
                )
                style_visualization(heatmap_fig, height=470)
                st.plotly_chart(
                    heatmap_fig,
                    width="stretch",
                    key=f"au_heatmap_{result_key}",
                )
                show_chart_explanation(f"heatmap_{result_key}", "chart_heatmap_help")

    st.dataframe(result_df.head(50), use_container_width=True)
    st.download_button(
        label=t("download"),
        data=result["parquet_bytes"],
        file_name=result["file_name"],
        mime="application/vnd.apache.parquet",
        key=f"download_{result['file_name']}",
    )

results_with_stimuli = {
    file_name: result
    for file_name, result in processed_results.items()
    if result.get("stimuli")
}
if results_with_stimuli:
    with st.expander(t("stimulus_comparison"), expanded=True):
        available_columns = set.union(*(
            set(result["dataframe"].columns)
            for result in results_with_stimuli.values()
        ))
        available_columns = list(available_columns)
        signal_categories = [
            (t("au_group"), "Action Units"),
            (t("emotion_group"), "Emotions"),
            (t("pose_group"), "Head pose"),
        ]
        category_options = [
            (label, category)
            for label, category in signal_categories
            if set(visualization_columns(pd.DataFrame(columns=available_columns), category))
        ]
        if category_options:
            selected_category_label = st.radio(
                t("signal_category"),
                options=[label for label, _ in category_options],
                horizontal=True,
                key="stimulus_comparison_category",
            )
            selected_category = dict(category_options)[selected_category_label]
            common_metric_columns = visualization_columns(
                pd.DataFrame(columns=available_columns), selected_category
            )
            selected_metric = st.selectbox(
                t("stimulus_metric"),
                options=common_metric_columns,
                key="stimulus_comparison_metric",
                accept_new_options=False,
            )
            comparison_data = compare_stimulus_windows(results_with_stimuli, selected_metric)
            if comparison_data.empty:
                st.info(t("no_stimulus_comparison"))
            else:
                comparison_fig = px.line(
                    comparison_data,
                    x="period",
                    y="value",
                    color="stimulus",
                    line_group="case",
                    markers=True,
                    hover_data=["subject", "case"],
                    category_orders={"period": ["Before", "During"]},
                    title=f"{selected_metric}: before vs during stimulus",
                    labels={"value": selected_metric, "period": "Period", "stimulus": "Stimulus"},
                )
                comparison_fig.update_xaxes(categoryorder="array", categoryarray=["Before", "During"])
                style_visualization(comparison_fig)
                st.plotly_chart(
                    comparison_fig,
                    width="stretch",
                    key="stimulus_comparison_chart",
                )
                show_chart_explanation("stimulus_comparison", "chart_before_during_help")
        else:
            st.info(t("no_stimulus_comparison"))

    with st.expander(t("advanced_stimulus_analysis"), expanded=True):
        event_tab, similarity_tab = st.tabs([
            t("event_aligned_tab"),
            t("similarity_tab"),
        ])
        all_result_columns = list(set.union(*(
            set(result["dataframe"].columns)
            for result in results_with_stimuli.values()
        )))
        result_schema = pd.DataFrame(columns=all_result_columns)
        available_categories = [
            (label, category)
            for label, category in [
                (t("au_group"), "Action Units"),
                (t("emotion_group"), "Emotions"),
                (t("pose_group"), "Head pose"),
            ]
            if visualization_columns(result_schema, category)
        ]
        stimulus_names = sorted({
            stimulus["nome"]
            for result in results_with_stimuli.values()
            for stimulus in result.get("stimuli", [])
        })

        if available_categories and stimulus_names:
            with event_tab:
                event_category_label = st.radio(
                    t("signal_category"),
                    options=[label for label, _ in available_categories],
                    horizontal=True,
                    key="event_aligned_category",
                )
                event_category = dict(available_categories)[event_category_label]
                event_metrics = visualization_columns(result_schema, event_category)
                event_control_col, event_metric_col = st.columns(2)
                with event_control_col:
                    event_stimulus = st.selectbox(
                        t("stimulus_selector"),
                        options=stimulus_names,
                        key="event_aligned_stimulus",
                        accept_new_options=False,
                    )
                with event_metric_col:
                    event_metric = st.selectbox(
                        t("stimulus_metric"),
                        options=event_metrics,
                        key="event_aligned_metric",
                        accept_new_options=False,
                    )
                event_data = event_aligned_signals(
                    results_with_stimuli,
                    event_metric,
                    event_stimulus,
                )
                if event_data.empty:
                    st.info(t("no_event_aligned_data"))
                else:
                    event_fig = px.line(
                        event_data,
                        x="relative_time",
                        y="value",
                        color="recording",
                        line_group="recording",
                        hover_data={"relative_time": ":.2f", "value": ":.3f"},
                        title=f"{event_metric} around {event_stimulus}",
                        labels={
                            "relative_time": "Time from stimulus onset (seconds)",
                            "value": event_metric,
                            "recording": "Recording",
                        },
                    )
                    duration = float(event_data["stimulus_duration"].median())
                    event_fig.add_vrect(
                        x0=0,
                        x1=duration,
                        fillcolor="#f09a52",
                        opacity=0.12,
                        line_width=0,
                        annotation_text=event_stimulus,
                        annotation_position="top left",
                    )
                    event_fig.add_vline(x=0, line_dash="dash", line_color="#f09a52")
                    style_visualization(event_fig, height=410)
                    st.plotly_chart(
                        event_fig,
                        width="stretch",
                        key="event_aligned_chart",
                    )
                    show_chart_explanation("event_aligned", "chart_event_aligned_help")

            with similarity_tab:
                similarity_category_label = st.radio(
                    t("signal_category"),
                    options=[label for label, _ in available_categories],
                    horizontal=True,
                    key="trajectory_similarity_category",
                )
                similarity_category = dict(available_categories)[similarity_category_label]
                similarity_metrics = visualization_columns(result_schema, similarity_category)
                similarity_control_col, similarity_metric_col = st.columns(2)
                with similarity_control_col:
                    similarity_stimulus = st.selectbox(
                        t("stimulus_selector"),
                        options=stimulus_names,
                        key="trajectory_similarity_stimulus",
                        accept_new_options=False,
                    )
                with similarity_metric_col:
                    similarity_metric = st.selectbox(
                        t("stimulus_metric"),
                        options=similarity_metrics,
                        key="trajectory_similarity_metric",
                        accept_new_options=False,
                    )
                similarity_data, shared_bins = stimulus_similarity_matrix(
                    results_with_stimuli,
                    similarity_metric,
                    similarity_stimulus,
                )
                if len(similarity_data) < 2:
                    st.info(t("similarity_min_recordings"))
                else:
                    similarity_fig = go.Figure(
                        data=go.Heatmap(
                            z=similarity_data.to_numpy(),
                            x=similarity_data.columns,
                            y=similarity_data.index,
                            customdata=shared_bins.to_numpy(),
                            colorscale="RdBu",
                            reversescale=True,
                            zmin=-1,
                            zmax=1,
                            colorbar={"title": "Pearson r"},
                            hovertemplate=(
                                "Recording A: %{y}<br>Recording B: %{x}"
                                "<br>Pearson r: %{z:.3f}<br>Shared phases: %{customdata}"
                                "<extra></extra>"
                            ),
                        )
                    )
                    similarity_fig.update_layout(
                        title=f"{similarity_metric} trajectory similarity during {similarity_stimulus}",
                        xaxis_title="Recording",
                        yaxis_title="Recording",
                    )
                    style_visualization(similarity_fig, height=500)
                    st.plotly_chart(
                        similarity_fig,
                        width="stretch",
                        key="trajectory_similarity_chart",
                    )
                    show_chart_explanation("trajectory_similarity", "chart_similarity_help")