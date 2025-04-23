import re
import joblib
import pandas as pd
from underthesea import word_tokenize, text_normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from deep_translator import GoogleTranslator
import streamlit as st

# Giao diện Streamlit
emoji_pattern = re.compile(
    "["  # Các emoji cần xóa
    u"\U0001F600-\U0001F64F"  # Mặt cười, cảm xúc
    u"\U0001F300-\U0001F5FF"  # Biểu tượng thiên nhiên
    u"\U0001F680-\U0001F6FF"  # Biểu tượng giao thông
    u"\U0001F1E0-\U0001F1FF"  # Cờ quốc gia
    u"\U00002702-\U000027B0"  # Ký hiệu dạng hình
    u"\U0001F926-\U0001F937"  # Hành động, cử chỉ
    u"\U00010000-\U0010FFFF"  # Các ký tự bổ sung
    u"\u200d"  # Nối ký tự (zero width joiner)
    u"\u2640-\u2642"  # Ký hiệu giới tính
    u"\u2600-\u2B55"  # Biểu tượng đa dạng khác
    u"\u23cf"  # Biểu tượng đẩy đĩa
    u"\u23e9"  # Tua nhanh
    u"\u231a"  # Đồng hồ
    u"\u3030"  # Dấu ngoằn ngoèo
    u"\ufe0f"  # Bộ chọn kiểu hiển thị
    "]+", flags=re.UNICODE
)

short_word_dict = {
    "ko": "không", "kg": "không", "khong": "không", "k": "không", "kh": "không",
    "cx": "cũng", "mik": "mình", "mn": "mọi người", "bt": "bình thường",
    "nv": "nhân viên", "sp": "sản phẩm", "đc": "được", "dc": "được",
    "đk": "điều khoản", "đt": "điện thoại", "j": "gì", "vs": "với",
    "hok": "không", "lun": "luôn", "z": "gì", "zậy": "gì vậy", "thik": "thích",
    "hum": "hôm", "wa": "qua", "m": "mình", "mk": "mình", "bn": "bạn", "ok": "ổn"
}

def translate_to_vietnamese(text):
    try:
        return GoogleTranslator(source='auto', target='vi').translate(text)
    except Exception as e:
        return text  # Nếu lỗi thì trả lại nguyên văn


def clean_data(text):
    text = text.lower()
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text_normalize(text)
    words = text.split()
    words = [short_word_dict.get(word, word) for word in words]
    text = ' '.join(words)
    text = word_tokenize(text, format="text")
    return text

def train_model(df):
    vectorizer = CountVectorizer()
    model = LogisticRegression()

    # Chia dữ liệu train/test
    train_sentences, test_sentences, train_labels, test_labels = train_test_split(
        df['clean_content'],
        df['title'],
        test_size=0.1,
        random_state=42
    )

    # Vector hóa
    train_features = vectorizer.fit_transform(train_sentences)
    model.fit(train_features, train_labels)

    return model, vectorizer

@st.cache_resource
def load_data():
    sheet_id = "1GTo2SGUjDUA4T1mBaUwIkiOhCmfcRaIygP-5KTb5Gl0"
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
    df = pd.read_csv(csv_url)
    df = df.fillna('')  # Thay thế NaN bằng chuỗi rỗng trong toàn bộ DataFrame
    return df

def save_model(model, vectorizer, model_file='sentiment_model.pkl', vectorizer_file='count_vectorizer.pkl'):
    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)

def load_model(model_file='sentiment_model.pkl', vectorizer_file='count_vectorizer.pkl'):
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
    return model, vectorizer
