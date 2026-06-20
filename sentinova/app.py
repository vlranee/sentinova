"""
Sentinova — Streamlit Community Cloud entry point
Arsitektur: FastAPI (background thread, port 8502) + Streamlit HTML wrapper
"""
import threading
import json
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

# ── Load model (cached, sekali saja) ─────────────────────────────────────────
MODEL_NAME = "rantirann/sentinova-indobert"

@st.cache_resource(show_spinner="Memuat model IndoBERT dari Hugging Face…")
def load_resources():
    from utils.model_loader import load_model
    return load_model(MODEL_NAME)

tokenizer, model, cfg = load_resources()

# ── Jalankan FastAPI di background thread ────────────────────────────────────
API_PORT = 8502
API_URL  = f"http://localhost:{API_PORT}"

@st.cache_resource
def start_api_server():
    """Start FastAPI server in daemon thread — dipanggil sekali via cache."""
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    class PredictRequest(BaseModel):
        text: str

    @app.get("/")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
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

    config = uvicorn.Config(app, host="0.0.0.0", port=API_PORT, log_level="error")
    server = uvicorn.Server(config)

    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    return API_URL

api_url = start_api_server()

# Tunggu server siap (max 10s, biasanya < 1s karena sudah ada di cache)
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

# Patch: hardcode API URL ke localhost:8502
html_content = html_content.replace(
    "window.FLASK_API_URL = params.get('api_url') || 'http://localhost:5000';",
    f"window.FLASK_API_URL = '{api_url}';"
)

# ── Render ────────────────────────────────────────────────────────────────────
components.html(html_content, height=920, scrolling=True)
