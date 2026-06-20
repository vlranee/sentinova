import re
import unicodedata


# Emoji range — identik dengan NB2b cell 6
_EMOJI_RE = re.compile(
    '[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
    '\U0001F680-\U0001F6FF\U0001FA00-\U0001FA6F'
    '\U0001F1E0-\U0001F1FF\U00002702-\U000027B0'
    '\U0000200D\U0000FE0F]+',
    flags=re.UNICODE,
)

# Normalisasi slang domain fraud — identik dengan NB2b cell 8
_BERT_NORM = {
    "tf"      : "transfer",
    "trf"     : "transfer",
    "wd"      : "withdraw",
    "depo"    : "deposit",
    "wa"      : "whatsapp",
    "tele"    : "telegram",
    "phising" : "phishing",
    "fishing" : "phishing",
}
_NORM_PATTERN = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(_BERT_NORM, key=len, reverse=True)) + r")(?!\w)",
    re.IGNORECASE | re.UNICODE,
)


def _normalize(text: str) -> str:
    def _rep(m):
        return _BERT_NORM.get(m.group(0).lower(), m.group(0))
    return _NORM_PATTERN.sub(_rep, text)


def preprocess_text(text: str) -> str:
    """
    Preprocessing identik dengan NB2b (clean_text_bert → normalize_bert).
    Urutan operasi dijaga sama persis dengan notebook agar hasil inferensi konsisten.
    Tidak melakukan stemming/stopword — tokenizer BERT menangani subword.
    """
    # NFC normalization
    t = unicodedata.normalize("NFC", str(text))

    # Ganti URL dengan token [URL]
    t = re.sub(r"https?://\S+|www\.\S+", "[URL]", t)

    # Anonimisasi mention
    t = re.sub(r"@\w+", "@pengguna", t)

    # Hapus tanda pagar, pertahankan kata
    t = re.sub(r"#(\w+)", r"\1", t)

    # Decode HTML entities
    for ent, rep in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                     ("&quot;", '"'), ("&#39;", "'")]:
        t = t.replace(ent, rep)

    # Hapus karakter kontrol
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", t)

    # Hapus emoji (ganti spasi, bukan hapus total — sama dengan notebook)
    t = _EMOJI_RE.sub(" ", t)

    # Normalisasi slang
    t = _normalize(t)

    # Lowercase
    t = t.lower()

    # Bersihkan whitespace berlebih
    t = re.sub(r"\s+", " ", t).strip()

    return t
