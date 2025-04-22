import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from underthesea import word_tokenize # Bộ ngữ nghĩa
from underthesea import text_normalize #Chuẩn hóa văn bản
import re

# Giao diện Streamlit
st.set_page_config(page_title="Dự Đoán Cảm Xúc Khách Hàng", page_icon="😊", layout="wide")
st.title("Dự Đoán Cảm Xúc Khách Hàng Từ Bình Luận Sản Phẩm")

# Phần hướng dẫn
st.markdown("""
    ### Nhập bình luận của bạn dưới đây và chúng tôi sẽ dự đoán cảm xúc của khách hàng.
    **Mô hình của chúng tôi phân loại thành 2 nhãn:** 
    - **Hài lòng**
    - **Không hài lòng**
""")


#Danh sách emoji cần xóa
emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # Mặt cười, cảm xúc
        u"\U0001F300-\U0001F5FF"  # Biểu tượng thiên nhiên, đối tượng
        u"\U0001F680-\U0001F6FF"  # Biểu tượng giao thông
        u"\U0001F1E0-\U0001F1FF"  # Cờ quốc gia
        u"\U00002702-\U000027B0"  # Ký hiệu dạng hình
        u"\U000024C2-\U0001F251"  # Ký hiệu đóng khung
        u"\U0001F926-\U0001F937"  # Hành động, cử chỉ
        u"\U00010000-\U0010FFFF" # Các ký tự bổ sung
        u"\u200d"                 # Nối ký tự (zero width joiner)
        u"\u2640-\u2642"          # Ký hiệu giới tính
        u"\u2600-\u2B55"          # Biểu tượng đa dạng khác
        u"\u23cf"                 # Biểu tượng đẩy đĩa
        u"\u23e9"                 # Tua nhanh
        u"\u231a"                 # Đồng hồ
        u"\u3030"                 # Dấu ngoằn ngoèo
        u"\ufe0f"                 # Bộ chọn kiểu hiển thị
        "]+", flags=re.UNICODE
)
#Load danh sách các từ viết tắt hay sử dụng
short_word_dict = {
    "ko": "không",
    "kg": "không",
    "khong": "không",
    "k": "không",
    "kh": "không",
    "cx": "cũng",
    "mik": "mình",
    "mn": "mọi người",
    "bt": "bình thường",
    "nv": "nhân viên",
    "sp": "sản phẩm",
    "đc": "được",
    "dc": "được",
    "đk": "điều khoản",
    "đt": "điện thoại",
    "j": "gì",
    "vs": "với",
    "hok": "không",
    "lun": "luôn",
    "z": "gì",
    "zậy": "gì vậy",
    "thik": "thích",
    "hum": "hôm",
    "wa": "qua",
    "m": "mình",
    "mk": "mình",
    "bn": "bạn",
    "ok": "ổn"
}

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

# Hàm huấn luyện mô hình
def train_model(df):
    # Vectorizer và mô hình
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

# Sử dụng st.cache_resource cho việc lưu mô hình và vectorizer
@st.cache_resource
def load_data():
    sheet_id = "1GTo2SGUjDUA4T1mBaUwIkiOhCmfcRaIygP-5KTb5Gl0"
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
    df = pd.read_csv(csv_url)
    df = df.fillna('')  # Thay thế NaN bằng chuỗi rỗng trong toàn bộ DataFrame
    return df
# Tải dữ liệu
df_balanced_2_label = load_data()

# Nếu mô hình chưa được huấn luyện và lưu, ta sẽ huấn luyện và lưu mô hình
model_file = 'sentiment_model.pkl'
vectorizer_file = 'count_vectorizer.pkl'

# Nếu mô hình đã tồn tại, tải mô hình và vectorizer đã lưu
try:
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
except FileNotFoundError:
    # Nếu chưa có mô hình, huấn luyện và lưu lại
    model, vectorizer = train_model(df_balanced_2_label)
    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)

# Nhập dữ liệu văn bản từ người dùng
user_input = st.text_area("Nhập bình luận của khách hàng:")

if st.button("Dự đoán cảm xúc"):
    if user_input:
        user_input = clean_data(user_input)
        # Vector hóa văn bản người dùng nhập
        user_vector = vectorizer.transform([user_input])
        # Dự đoán cảm xúc
        prediction = model.predict(user_vector)

        # Hiển thị kết quả
        if prediction[0] == "cực kỳ hài lòng":
            st.success("👍 Khách hàng **hài lòng** với sản phẩm.")
        else:
            st.error("👎 Khách hàng **không hài lòng** với sản phẩm.")
    else:
        st.warning("Vui lòng nhập bình luận để dự đoán cảm xúc.")
