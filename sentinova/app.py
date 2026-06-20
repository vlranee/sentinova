"""
Sentinova — Streamlit Community Cloud entry point
"""
import threading
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

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_NAME = "rantirann/sentinova-indobert"

@st.cache_resource(show_spinner="Memuat model IndoBERT…")
def load_resources():
    from utils.model_loader import load_model
    return load_model(MODEL_NAME)

tokenizer, model, cfg = load_resources()

# ── FastAPI di background thread ──────────────────────────────────────────────
API_PORT = 8502
API_URL  = f"http://localhost:{API_PORT}"

@st.cache_resource
def start_api_server(_tokenizer, _model, _cfg):
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    # Nama variabel 'sentinova_app' bukan 'app' — hindari konflik dengan uvicorn
    sentinova_app = FastAPI()
    sentinova_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    class PredictRequest(BaseModel):
        text: str

    @sentinova_app.get("/")
    def health():
        return {"status": "ok"}

    @sentinova_app.post("/predict")
    def predict_endpoint(req: PredictRequest):
        from utils.model_loader import predict
        from utils.preprocessing import preprocess_text
        clean = preprocess_text(req.text)
        label, confidence, all_scores = predict(clean, _tokenizer, _model, _cfg)
        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "scores": {k: round(v, 4) for k, v in all_scores.items()},
        }

    config = uvicorn.Config(sentinova_app, host="0.0.0.0", port=API_PORT, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    return API_URL

api_url = start_api_server(tokenizer, model, cfg)

# Tunggu server siap
import time, requests as _req
for _ in range(20):
    try:
        if _req.get(f"{api_url}/", timeout=1).status_code == 200:
            break
    except Exception:
        time.sleep(0.5)

# ── Load & patch HTML ─────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent / "sentinova.html"
html_content = HTML_PATH.read_text(encoding="utf-8")

html_content = html_content.replace(
    "window.FLASK_API_URL = params.get('api_url') || 'http://localhost:5000';",
    f"window.FLASK_API_URL = '{api_url}';"
)

# ── Render — pakai st.components.v1.html (masih works, warning bisa diabaikan) ──
components.html(html_content, height=920, scrolling=True)