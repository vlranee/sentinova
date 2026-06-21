import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(
    page_title="Sentinova",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""
<style>
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding: 0 !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }
[data-testid="stSidebar"] { display: none; }
iframe { border: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model (sekali saja, di-cache) ─────────────────────────────────────
MODEL_NAME = "rantirann/sentinova-indobert"

@st.cache_resource(show_spinner="Memuat model IndoBERT…")
def load_resources():
    from utils.model_loader import load_model
    return load_model(MODEL_NAME)

tokenizer, model, cfg = load_resources()

# ── Load & patch HTML — fetch ke /api/predict (relative, satu origin) ──────
HTML_PATH = Path(__file__).parent / "sentinova.html"
html_content = HTML_PATH.read_text(encoding="utf-8")
html_content = html_content.replace(
    "window.FLASK_API_URL = params.get('api_url') || 'http://localhost:5000';",
    "window.FLASK_API_URL = '/api';"
)

components.html(html_content, height=920, scrolling=True)


# ─────────────────────────────────────────────────────────────────────────
# ASGI entry point — WAJIB bernama `app` di top-level module supaya
# Streamlit Cloud (ASGI-only sejak 1.5x) bisa mendeteksinya.
# ─────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.routing import Mount
from streamlit.starlette import App as StreamlitApp

api = FastAPI()
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str

@api.get("/health")
def health():
    return {"status": "ok"}

@api.post("/predict")
def predict_endpoint(req: PredictRequest):
    from utils.model_loader import predict
    from utils.preprocessing import preprocess_text
    clean = preprocess_text(req.text)
    label, confidence, all_scores = predict(clean, tokenizer, model, cfg)
    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "scores": {k: round(v, 4) for k, v in all_scores.items()},
    }

# Streamlit UI tetap di "/", endpoint FastAPI ter-mount di "/api/*"
app = StreamlitApp(
    __file__,
    routes=[Mount("/api", app=api)],
)
