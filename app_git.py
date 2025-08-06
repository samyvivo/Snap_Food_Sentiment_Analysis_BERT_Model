import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
import zipfile
import io
import pickle

@st.cache_resource
def load_model():
    url = "https://huggingface.co/samyhusy/Snap_food_Sentiment_analysis/resolve/main/snapfood_model.zip"
    response = requests.get(url)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall("model_dir")

    MODEL_PATH = "model_dir/snapfood_model"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ------------------------
# تنظیمات صفحه
# ------------------------
st.set_page_config(page_title="تحلیل احساسات اسنپ‌فود", layout="wide")


# تابع تحلیل احساسات
def sentiment_analysis(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    return predicted_class, confidence


st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Vazirmatn', sans-serif;
    }

    h1, h2, h3, h4, h5, h6, p, div, span {
        font-family: 'Vazirmatn', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)


# ------------------------
# طراحی صفحه با دو ستون
# ------------------------
col1, col2 = st.columns([2, 1])

# ستون ۱ → تصویر بزرگ
with col1:
    st.image("Pizaa.png")

# ستون ۲ → فرم و نتیجه
with col2:
    st.image("snapfood_logo.png", width=150)
    st.markdown(
        """
        <h2 style='text-align: right; direction: rtl; font-family: B Nazanin;'>
        تحلیل احساسی نظرات کاربران اسنپ‌فود (مدل BERT)
        </h2>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <p style='text-align: right; direction: rtl; font-size:18px;'>
        متن نظر خود را وارد کنید تا مدل آن را به صورت مثبت یا منفی تحلیل کند.
        </p>
        """,
        unsafe_allow_html=True
    )

    user_input = st.text_area("📝 متن نظر شما:", key="input", height=150)

    if st.button("تحلیل احساس"):
        if user_input.strip() == "":
            st.warning("⚠️ لطفاً یک متن وارد کنید.")
        else:
            label, confidence = sentiment_analysis(user_input)
            sentiment = "😊 مثبت" if label == 1 else "😠 منفی"
            st.success(f"نتیجه پیش‌بینی: {sentiment}")
            st.info(f"درصد اطمینان مدل: {confidence:.2%}")
