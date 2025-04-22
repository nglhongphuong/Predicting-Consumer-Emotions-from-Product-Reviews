import streamlit as st

st.set_page_config(
    page_title="Predicting Customer Emotions From Product Reviews",
    page_icon="💬",
    layout="wide"
)

st.markdown("""
    <style>
        .title { text-align: center; font-size: 40px; font-weight: bold; color: #ff6347; margin-bottom: 10px; }
        .subtitle { text-align: center; font-size: 22px; color: #444; margin-bottom: 40px; }
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

st.markdown('</div>', unsafe_allow_html=True)
