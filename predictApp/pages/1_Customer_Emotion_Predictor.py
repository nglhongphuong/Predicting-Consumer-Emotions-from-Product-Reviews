import streamlit as st
import utils.func
import joblib
import os

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


# ===== Load model và vectorizer đã huấn luyện ở Page 1 =====
if os.path.exists("sentiment_model.pkl"):
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("count_vectorizer.pkl")
else:
    st.error("Model chưa được tạo. Vui lòng chạy trang page_1 để huấn luyện trước.")


# Nhập dữ liệu văn bản từ người dùng
user_input = st.text_area("Nhập bình luận của khách hàng:")

if st.button("Dự đoán cảm xúc"):
    if user_input:
        user_input = utils.func.clean_data(utils.func.translate_to_vietnamese(user_input))
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
