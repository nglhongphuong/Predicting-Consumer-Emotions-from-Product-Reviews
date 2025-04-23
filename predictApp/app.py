import streamlit as st
import sys
import joblib
import utils.func

st.set_page_config(
    page_title="Predicting Customer Emotions From Product Reviews",
    page_icon="💬",
    layout="wide"
)
#========== LOAD MÔ HÌNH ĐẦU TIÊN =======================
df_balanced_2_label = utils.func.load_data()

# Nếu mô hình chưa được huấn luyện và lưu, ta sẽ huấn luyện và lưu mô hình
model_file = 'sentiment_model.pkl'
vectorizer_file = 'count_vectorizer.pkl'

# Nếu mô hình đã tồn tại, tải mô hình và vectorizer đã lưu
try:
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
except FileNotFoundError:
    # Nếu chưa có mô hình, huấn luyện và lưu lại
    model, vectorizer = utils.func.train_model(df_balanced_2_label)
    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)


#============================================================
st.markdown("""
    <style>
        .title { text-align: center; font-size: 40px; font-weight: bold; color: #ff6347; margin-bottom: 10px; }
        .subtitle { text-align: center; font-size: 22px; color: white; margin-bottom: 40px; }
        .section { background-color: #f9f9f9; padding: 30px; border-radius: 12px;
                   box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); margin: 0 auto; max-width: 900px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">💬 Predicting Customer Emotions From Product Reviews 💬</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Dự đoán cảm xúc Cực kỳ hài lòng / Không hài lòng từ bình luận sản phẩm để nâng cao chất lượng dịch vụ!</div>', unsafe_allow_html=True)

st.markdown("### 📌 Giới thiệu ứng dụng")
st.markdown("""
Ứng dụng này giúp bạn phân tích bình luận của khách hàng để nhận diện cảm xúc ẩn sau từng câu chữ:

- 😄 **Phân loại cảm xúc**: Chỉ với 2 nhãn — **Hài lòng** và **Không hài lòng**.
- 📈 **Hỗ trợ doanh nghiệp**: Nắm bắt nhanh phản hồi khách hàng, cải thiện dịch vụ và sản phẩm.
- 🧠 **Ứng dụng NLP tiếng Việt**: Phân tích và xử lý ngôn ngữ tự nhiên trong bình luận tiếng Việt từ Tiki.
""")

st.markdown("### 👩‍💻 Công nghệ áp dụng")
st.markdown("""
- 🤖 **Logistic Regression (Machine Learning truyền thống)**: Mô hình đơn giản nhưng hiệu quả, đạt **88% độ chính xác**.
- 📦 **Dữ liệu thực tế từ Tiki**: Bình luận sản phẩm tiếng Việt được crawl và xử lý.
- 🧹 **Tiền xử lý văn bản tiếng Việt**: Chuẩn hóa, tách từ...
- 🌐 **Streamlit**: Giao diện người dùng đơn giản, thân thiện, trực quan.
""")

st.markdown("### 🔗 Liên kết")
st.markdown("""
- 📂 **Mã nguồn**: [GitHub Repository](https://github.com/nglhongphuong/Predicting-Consumer-Emotions-from-Product-Reviews)
- 📅 **Phiên bản**: 1.0 | 🕓 Ngày tạo: Tháng 4/2025
""")

# Nút điều hướng đến các trang chính
st.markdown('<div class="button-container">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    if st.button("📝 Dự đoán cảm xúc từ một bình luận"):
        st.switch_page("pages/1_Customer_Emotion_Predictor.py")

with col2:
    if st.button("📄 Dự đoán cảm xúc từ file CSV"):
        st.switch_page("pages/2_Labeling_data.py")

st.write(f"Python version: {sys.version}")
st.markdown('</div>', unsafe_allow_html=True)
