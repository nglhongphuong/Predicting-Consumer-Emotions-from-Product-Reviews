import streamlit as st
import pandas as pd
import joblib
import re
from underthesea import word_tokenize, text_normalize

# ===== Thiết lập giao diện =====
st.set_page_config(page_title="Dự Đoán Cảm Xúc Theo File", page_icon="📄", layout="wide")
st.title("📄 Dự Đoán Cảm Xúc Từ File Bình Luận Khách Hàng")

st.markdown("""
#### 👉 Tải lên file `.csv` chứa cột `content` (bình luận).
Sau đó, hệ thống sẽ:
- Làm sạch dữ liệu
- Dự đoán cảm xúc theo mô hình đã huấn luyện (Page 1)
- Trả về file `.csv` đã được gán nhãn để bạn tải về
""")

# ===== Load model và vectorizer đã huấn luyện ở Page 1 =====
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("count_vectorizer.pkl")

# ===== Bộ lọc & xử lý văn bản =====
emoji_pattern = re.compile(
    "[" + "".join([
        u"\U0001F600-\U0001F64F", u"\U0001F300-\U0001F5FF", u"\U0001F680-\U0001F6FF",
        u"\U0001F1E0-\U0001F1FF", u"\U00002702-\U000027B0", u"\U000024C2-\U0001F251",
        u"\U0001F926-\U0001F937", u"\U00010000-\U0010FFFF", u"\u200d", u"\u2640-\u2642",
        u"\u2600-\u2B55", u"\u23cf", u"\u23e9", u"\u231a", u"\u3030", u"\ufe0f"
    ]) + "]+", flags=re.UNICODE
)

short_word_dict = {
    "ko": "không", "kg": "không", "khong": "không", "k": "không", "kh": "không",
    "cx": "cũng", "mik": "mình", "mn": "mọi người", "bt": "bình thường",
    "nv": "nhân viên", "sp": "sản phẩm", "đc": "được", "dc": "được",
    "đk": "điều khoản", "đt": "điện thoại", "j": "gì", "vs": "với",
    "hok": "không", "lun": "luôn", "z": "gì", "zậy": "gì vậy", "thik": "thích",
    "hum": "hôm", "wa": "qua", "m": "mình", "mk": "mình", "bn": "bạn", "ok": "ổn"
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text_normalize(text)
    words = text.split()
    words = [short_word_dict.get(word, word) for word in words]
    text = ' '.join(words)
    text = word_tokenize(text, format="text")
    return text

# ===== Giao diện upload file =====
uploaded_file = st.file_uploader("📤 Tải lên file CSV chứa cột `content`", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if 'content' not in df.columns:
            st.error("❌ File cần có cột tên là `content`.")
        else:
            with st.spinner("🔄 Đang xử lý và dự đoán..."):
                df['clean_content'] = df['content'].apply(clean_text)
                features = vectorizer.transform(df['clean_content'])
                df['sentiment'] = model.predict(features)

                # Cho tải xuống
                st.success("✅ Xử lý thành công! Nhấn nút dưới để tải file kết quả.")
                st.download_button(
                    label="📥 Tải file kết quả",
                    data=df.to_csv(index=False).encode('utf-8-sig'),
                    file_name="ket_qua_du_doan.csv",
                    mime="text/csv"
                )

                st.dataframe(df[['content', 'sentiment']])
    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý file: {e}")
