# Sentinova — Deteksi Penipuan Bahasa Indonesia

Aplikasi deteksi penipuan berbasis IndoBERT fine-tuned pada tweet X (Twitter) berbahasa Indonesia.

## Stack
- **Frontend**: HTML + Chart.js + Lucide Icons (embedded dalam Streamlit)
- **Backend**: FastAPI (background thread) + Streamlit
- **Model**: IndoBERT fine-tuned — [`rantirann/sentinova-indobert`](https://huggingface.co/rantirann/sentinova-indobert)

## Struktur
```
sentinova/
├── app.py                  ← Streamlit entry point
├── sentinova.html          ← Frontend HTML
├── requirements.txt
├── packages.txt
├── .streamlit/
│   └── config.toml
└── utils/
    ├── __init__.py
    ├── model_loader.py     ← Load dari HF Hub + inferensi
    └── preprocessing.py   ← Preprocessing identik dengan NB2b
```

## Deploy ke Streamlit Community Cloud
1. Push repo ini ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` sebagai entrypoint
4. Deploy — model otomatis diunduh dari Hugging Face saat cold start (~2 menit pertama)

## Catatan Performa
- Model `model.safetensors` ≈ 498 MB — Streamlit Cloud punya RAM ~1GB, cukup untuk CPU inference
- Cold start pertama ~2-3 menit (download model dari HF Hub)
- Setelah itu `@st.cache_resource` menjaga model tetap di memory
