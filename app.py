%%writefile app.py
import streamlit as st
import requests
import urllib.parse
from pymongo import MongoClient
from PIL import Image
import pytesseract
import torch
from transformers import pipeline

# --- 1. CONFIGURATION & API KEYS ---
MONGODB_URI = "mongodb+srv://Mani:Mani1@cluster0.sxgntvn.mongodb.net/" #[cite: 1]
OPENROUTER_API_KEY = "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" #[cite: 1]
SERPER_API_KEY = "8468cefefc4741189cb2898e6d444003952f7286"
GOOGLE_FACT_KEY = "AIzaSyC2Cm0BsuLZG-ybMrkulhyyo4p5s3ZBkNs"

MODEL_A = "google/gemma-3-27b-it:free" #[cite: 1]
MODEL_B = "google/gemma-3-12b-it:free" #[cite: 1]

# --- 2. DATABASE SETUP ---
try:
    client = MongoClient(MONGODB_URI) #[cite: 1]
    db = client['OmniVerifier_2026']
    logs_col = db['Verification_Logs']
except Exception as e:
    st.error(f"Database Connection Failed: {e}")

# --- 3. HUGGING FACE MODELS ---
DEVICE = 0 if torch.cuda.is_available() else -1

@st.cache_resource
def load_models():
    t_mod = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection", device=DEVICE)
    i_mod = pipeline("image-classification", model="umm-maybe/AI-image-detector", device=DEVICE)
    # Fixed Whisper loading for long audio
    a_mod = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=DEVICE)
    return t_mod, i_mod, a_mod

fake_detector, img_detector, audio_transcriber = load_models()

# --- 4. CORE VERIFICATION LOGIC ---

def get_fact_check(query):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={urllib.parse.quote(query)}&key={GOOGLE_FACT_KEY}"
    try:
        res = requests.get(url, timeout=5).json()
        if "claims" in res:
            c = res["claims"][0]['claimReview'][0]
            return f"Fact-Check: {c['textualRating']} by {c['publisher']['name']}"
        return "No Professional Fact-Check Record."
    except: return "Fact-Check API Offline."

def get_search_evidence(query):
    evidence = []
    try:
        g_url = "https://google.serper.dev/search"
        g_res = requests.post(g_url, headers={'X-API-KEY': SERPER_API_KEY}, json={"q": query, "num": 10}, timeout=8).json()
        evidence.extend([item.get('snippet', '') for item in g_res.get('organic', [])])
    except: pass
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            evidence.extend([r['body'] for r in ddgs.text(query, max_results=5)])
    except: pass
    return " | ".join(evidence)

def ai_reasoning_hybrid(text, evidence, fact_data, bert_res):
    prompt = f"Verify this content: {text}\nEvidence: {evidence}\nFactCheck: {fact_data}\nBERT Style: {bert_res}\nVerdict (REAL/FAKE):"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"} #[cite: 1]
    try:
        payload = {"model": MODEL_A, "messages": [{"role": "user", "content": prompt}]} #[cite: 1]
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=12)
        return response.json()['choices'][0]['message']['content']
    except:
        try:
            payload = {"model": MODEL_B, "messages": [{"role": "user", "content": prompt}]} #[cite: 1]
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=12)
            return response.json()['choices'][0]['message']['content']
        except:
            return "AI reasoning engine unavailable."

def process_unified_verification(text_to_verify):
    if not text_to_verify or len(text_to_verify.strip()) < 10:
        return "Insufficient text.", "", "Insufficient text."
    b_res = fake_detector(text_to_verify[:512])[0]
    b_label = f"{b_res['label']} ({b_res['score']:.1%})"
    f_data = get_fact_check(text_to_verify)
    e_data = get_search_evidence(text_to_verify)
    analysis = ai_reasoning_hybrid(text_to_verify, e_data, f_data, b_label)
    try:
        logs_col.insert_one({"query": text_to_verify[:150], "verdict": analysis, "year": 2026}) #[cite: 1]
    except: pass
    return b_label, f_data, analysis

# --- 5. UI FRONTEND ---
st.set_page_config(page_title="OmniVerifier Ultra", layout="wide")
st.title("🛡️ OmniVerifier Ultra")

tab1, tab2, tab3 = st.tabs(["📝 Text", "🖼️ Image", "🎙️ Audio"])

with tab1:
    u_input = st.text_area("Paste News Headline:", height=150)
    if st.button("Deep Verify Text"):
        with st.spinner("Analyzing..."):
            b_l, f_d, verdict = process_unified_verification(u_input)
            st.metric("BERT Style", b_l)
            st.info(f"**Fact-Check:** {f_d}")
            st.markdown(f"###  Verdict\n{verdict}")

with tab2:
    img_file = st.file_uploader("Upload Image:")
    if img_file:
        img = Image.open(img_file)
        st.image(img, width=400)
        if st.button("Deep Scan Image"):
            res = img_detector(img)
            ai_score = next(item['score'] for item in res if item['label'] == 'artificial')
            st.write(f"**AI Content:** {ai_score:.2%}")
            ocr_text = pytesseract.image_to_string(img).strip()
            if ocr_text:
                st.info(f"Detected Text: {ocr_text}")
                _, _, v = process_unified_verification(ocr_text)
                st.success(f"**Verdict on Image Text:** {v}")

with tab3:
    aud_file = st.file_uploader("Upload Audio:")
    if aud_file:
        if st.button("Deep Scan Audio"):
            with open("temp.wav", "wb") as f: f.write(aud_file.getbuffer())
            with st.spinner("Transcribing..."):
                # FIXED: Added return_timestamps=True for 30s+ audio
                trans = audio_transcriber("temp.wav", return_timestamps=True)["text"]
                st.write(f"**Transcript:** {trans}")
                with st.spinner("Verifying transcript..."):
                    _, _, v = process_unified_verification(trans)
                    st.success(f"**Verdict on Audio Transcript:** {v}")
