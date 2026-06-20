import json
import logging
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "max_length"     : 128,
    "best_threshold" : 0.25,
    "temperature"    : 1.8265306122448979,
    "id2label"       : {"0": "NON_FRAUD", "1": "FRAUD"},
}


def load_model(model_name_or_path: str):
    """
    Load tokenizer & model dari Hugging Face Hub atau folder lokal.
    Jika model_name_or_path adalah HF repo (bukan path folder yang ada),
    config inferensi diambil dari training_config.json di repo tersebut.
    """
    import os

    is_local = os.path.isdir(model_name_or_path)

    # Coba load training_config.json dari HF Hub
    config = dict(_DEFAULTS)
    if not is_local:
        try:
            from huggingface_hub import hf_hub_download
            cfg_path = hf_hub_download(repo_id=model_name_or_path, filename="training_config.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            config.update({
                "max_length"     : raw.get("hyperparameters", {}).get("max_length", _DEFAULTS["max_length"]),
                "best_threshold" : raw.get("best_threshold", _DEFAULTS["best_threshold"]),
                "temperature"    : raw.get("temperature", _DEFAULTS["temperature"]),
                "id2label"       : {str(k): v for k, v in raw.get("id2label", _DEFAULTS["id2label"]).items()},
            })
            logger.info(f"Config dari HF Hub: threshold={config['best_threshold']}, "
                        f"temperature={config['temperature']:.4f}")
        except Exception as e:
            logger.warning(f"Gagal load training_config.json dari HF Hub: {e} — pakai default.")
    else:
        import os as _os
        cfg_path = _os.path.join(model_name_or_path, "training_config.json")
        if _os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            config.update({
                "max_length"     : raw.get("hyperparameters", {}).get("max_length", _DEFAULTS["max_length"]),
                "best_threshold" : raw.get("best_threshold", _DEFAULTS["best_threshold"]),
                "temperature"    : raw.get("temperature", _DEFAULTS["temperature"]),
                "id2label"       : {str(k): v for k, v in raw.get("id2label", _DEFAULTS["id2label"]).items()},
            })

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    config["device"] = device

    logger.info(f"Model '{model_name_or_path}' dimuat di: {device}")
    return tokenizer, model, config


def predict(text: str, tokenizer, model, config: dict):
    """
    Inferensi satu teks.
    Input: teks sudah dipreprocess (dari preprocessing.preprocess_text).
    """
    device      = config["device"]
    max_length  = config["max_length"]
    temperature = config.get("temperature", 1.0)
    threshold   = config.get("best_threshold", 0.5)
    id2label    = config["id2label"]

    inputs = tokenizer(
        text,
        truncation     = True,
        padding        = "max_length",
        max_length     = max_length,
        return_tensors = "pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.softmax(logits / temperature, dim=-1)[0].cpu().numpy()

    fraud_id   = next((int(k) for k, v in id2label.items() if v == "FRAUD"), 1)
    prob_fraud = float(probs[fraud_id])

    pred_id    = fraud_id if prob_fraud >= threshold else (1 - fraud_id)
    label      = id2label[str(pred_id)]
    confidence = float(probs[pred_id])
    all_scores = {id2label[str(i)]: float(p) for i, p in enumerate(probs)}

    return label, confidence, all_scores
