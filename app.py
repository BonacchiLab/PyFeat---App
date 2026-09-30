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

        .guide-layout {
            max-width: 980px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 220px minmax(0, 1fr);
            gap: 1.5rem;
            align-items: start;
        }

        .guide-nav {
            position: sticky;
            top: 1.5rem;
            background: #18262c;
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 1rem;
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
            .guide-layout {
                grid-template-columns: 1fr;
            }

            .guide-nav {
                position: static;
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
                padding: 2rem 1rem 3rem;
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
                    <li><strong>stimulus:</strong> marcador experimental atribuído às linhas dentro dos intervalos definidos.</li>
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
                    <li><strong>1 FPS - exact mean (Média exata):</strong> analisa os frames do segundo e calcula uma média. Reduz o volume de dados, mas suaviza alterações rápidas.</li>
                </ul>
                <div class="guide-note"><p>Escolha o modo com base na escala temporal da sua hipótese. Se a expressão relevante pode durar apenas alguns frames, a média por segundo pode ocultá-la.</p></div>
            </section>

            <section class="guide-section" id="metadata">
                <h2>Metadados experimentais e estímulos</h2>
                <p>Ative a opção de marcadores de estímulo quando quiser relacionar as deteções faciais com momentos específicos do protocolo experimental.</p>
                <ol>
                    <li>Indique quantos estímulos existem na sessão.</li>
                    <li>Defina o nome de cada estímulo.</li>
                    <li>Introduza os segundos de início e de fim relativamente ao início do vídeo.</li>
                    <li>Verifique se os intervalos correspondem ao protocolo e se não existem sobreposições involuntárias.</li>
                </ol>
                <p>A coluna <strong>stimulus</strong> identifica o estímulo associado a cada linha. As linhas fora dos intervalos definidos ficam sem marcador.</p>
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
            <a href="#understanding-output">Understanding the Output Dataset</a>
            <a href="#action-units">Action Units</a>
            <a href="#au-reference">AU Reference</a>
            <a href="#emotions">Emotions and Facial Metrics</a>
            <a href="#landmarks">Facial Landmarks and Face Detection</a>
            <a href="#sampling">Sampling Modes</a>
            <a href="#metadata">Experimental Metadata and Stimuli</a>
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
                <div class="guide-note"><p>Py-Feat makes frame-by-frame predictions. A detection in one frame does not guarantee that the same face will be identified in exactly the same way in every later frame.</p></div>
            </section>

            <section class="guide-section" id="understanding-output">
                <h2>Understanding the Output Dataset</h2>
                <p>The Parquet file contains Py-Feat predictions and identifiers added by this interface. The exact set of columns depends on the detector and the available models.</p>
                <ul>
                    <li><strong>subject_id:</strong> participant identifier assigned during upload.</li>
                    <li><strong>input_file:</strong> source video name.</li>
                    <li><strong>frame:</strong> frame number in the original video.</li>
                    <li><strong>time_sec:</strong> approximate frame time in seconds in Every frame mode.</li>
                    <li><strong>approx_time:</strong> readable time representation, such as 00:12.</li>
                    <li><strong>second, time_start_sec, and time_end_sec:</strong> time interval created when data are aggregated by second.</li>
                    <li><strong>stimulus:</strong> experimental marker assigned to rows inside the defined windows.</li>
                    <li><strong>Action Units, emotions, landmarks, and other metrics:</strong> facial predictions produced by the Py-Feat model.</li>
                </ul>
                <div class="guide-note"><p>Always check the column names before analysis. Different detectors do not necessarily produce the same categories or number of landmarks.</p></div>
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
                <div class="image-placeholder">Recommended image: a facial diagram showing the main Action Units and their corresponding muscle regions.</div>
            </section>

            <section class="guide-section" id="emotions">
                <h2>Emotions and Facial Metrics</h2>
                <p>Emotion models estimate categories such as <strong>Neutral, Happy, Sad, Surprise, Fear, Disgust</strong>, and <strong>Anger</strong>. These estimates are based on the facial appearance observed by the model.</p>
                <p>Probabilities or values should be treated as indicators for analysis, not as direct measurements of a person's internal emotional state. Interpretation should consider context, individual variability, and the experimental design.</p>
                <p>Depending on the detector, outputs may also include <strong>valence</strong>, <strong>arousal</strong>, <strong>gaze</strong>, and <strong>head pose</strong>. Use these variables only when they are present in the downloaded dataset.</p>
            </section>

            <section class="guide-section" id="landmarks">
                <h2>Facial Landmarks and Face Detection</h2>
                <p><strong>Facial landmarks</strong> are reference points placed at specific facial locations, such as the eyebrows, eyes, nose, mouth, and facial outline. They describe facial geometry and movement over time.</p>
                <p><strong>Face detection</strong> locates the face in the image and may provide a bounding box, or <strong>face box</strong>. Landmarks and face boxes help verify whether the face was detected correctly and support the study of facial movement.</p>
                <div class="guide-note"><p>The number and numbering of landmarks depend on the model. Do not directly compare coordinates from different models without checking their conventions.</p></div>
                <div class="image-placeholder">Recommended image: an example face with numbered landmarks and the face box highlighted.</div>
            </section>

            <section class="guide-section" id="sampling">
                <h2>Sampling Modes</h2>
                <ul>
                    <li><strong>Every frame:</strong> preserves the highest temporal resolution and is useful for fast phenomena, but creates larger datasets and takes longer.</li>
                    <li><strong>1 FPS - representative frame:</strong> selects approximately one frame per second. It is faster and more compact, but may miss brief movements.</li>
                    <li><strong>1 FPS - exact mean:</strong> analyzes the frames in each second and calculates a mean. It reduces data volume, but smooths rapid changes.</li>
                </ul>
                <div class="guide-note"><p>Choose the mode based on the time scale of your hypothesis. If the relevant expression may last only a few frames, a one-second mean may hide it.</p></div>
            </section>

            <section class="guide-section" id="metadata">
                <h2>Experimental Metadata and Stimuli</h2>
                <p>Enable stimulus markers when you want to relate facial detections to specific moments in the experimental protocol.</p>
                <ol>
                    <li>Indicate how many stimuli are in the session.</li>
                    <li>Define the name of each stimulus.</li>
                    <li>Enter the start and end seconds relative to the beginning of the video.</li>
                    <li>Check that the intervals match the protocol and that there are no unintended overlaps.</li>
                </ol>
                <p>The <strong>stimulus</strong> column identifies the stimulus associated with each row. Rows outside the defined intervals remain unlabelled.</p>
            </section>

            <section class="guide-section" id="good-practices">
                <h2>Good Practices and Limitations</h2>
                <ul>
                    <li>Visually inspect a sample of the videos and results before statistical analysis.</li>
                    <li>Do not treat an emotion prediction as definitive evidence of an emotional state.</li>
                    <li>Record the sampling mode, stimulus intervals, and any data-exclusion decisions.</li>
                    <li>Take extra care with occlusion, glasses, masks, lighting, pose, low resolution, and multiple faces.</li>
                    <li>Avoid directly comparing results from different models or versions without documenting the difference.</li>
                    <li>Consult the official documentation and model licenses before reusing results in other contexts.</li>
                </ul>
                <div class="guide-note"><p>Py-Feat recommends checking detections carefully. Computer-vision models can produce unexpected results in real-world videos, even when they perform well on benchmarks.</p></div>
                <p class="guide-links">References: <a href="https://py-feat.org/" target="_blank">Py-Feat</a> | <a href="https://py-feat.org/pages/models/" target="_blank">Models</a> | <a href="https://py-feat.org/pages/faqs/" target="_blank">FAQs and known issues</a></p>
            </section>
        </div>
        """,
    }

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
    st.html(f"<div class=\"guide-layout\">{intro_guide[language]}</div>")
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
processed_results = st.session_state.setdefault("processed_results", {})

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

                    parquet_bytes = dataframe_to_parquet_bytes(output_df)
                    processed_results[file.name] = {
                        "subject_id": subject_id,
                        "dataframe": output_df,
                        "parquet_bytes": parquet_bytes,
                        "file_name": f"{subject_id}_{os.path.splitext(file.name)[0]}_pyfeat.parquet",
                    }

            finally:
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
        
        # FIMMMMM da barra 
        progress_bar.progress(1.0, text=t("finished"))
        st.info(t("finished"))

for result in processed_results.values():
    st.success(t("completed_file", name=result["dataframe"]["input_file"].iloc[0]))
    st.write(t("preview", rows=result["dataframe"].shape[0], mode=sampling_rate))
    st.dataframe(result["dataframe"].head(50), use_container_width=True)
    st.download_button(
        label=t("download"),
        data=result["parquet_bytes"],
        file_name=result["file_name"],
        mime="application/vnd.apache.parquet",
        key=f"download_{result['file_name']}",
    )