import streamlit as st
import pandas as pd
import joblib
import os
import utils.func

# ===== Thiết lập giao diện =====
st.set_page_config(page_title="Dự Đoán Cảm Xúc Theo File", page_icon="📄", layout="wide")
st.title("📄 Dự Đoán Cảm Xúc Từ File Bình Luận Khách Hàng")

st.markdown("""
#### 👉 Tải lên file `.csv,.xlsx, .txt` chứa cột `content` (bình luận).
Sau đó, hệ thống sẽ:
- Làm sạch dữ liệu
- Dự đoán cảm xúc theo mô hình đã huấn luyện (Page 1)
- Trả về file `.csv,.xlsx, .txt` đã được gán nhãn để bạn tải về
""")

# ===== Load model và vectorizer đã huấn luyện ở Page 1 =====
if os.path.exists("sentiment_model.pkl"):
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("count_vectorizer.pkl")
else:
    st.error("Model chưa được tạo. Vui lòng chạy trang page_1 để huấn luyện trước.")

# ===== Giao diện upload file =====
uploaded_file = st.file_uploader("📤 Tải lên file chứa cột `content` (.csv, .xlsx, .txt)",
                                 type=["csv", "xlsx", "xls", "txt"])

if uploaded_file is not None:
    try:
        # Đọc file dựa trên đuôi file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            df = pd.read_csv(uploaded_file, delimiter="\t")  # hoặc delimiter="," nếu cần
        else:
            st.error("❌ Định dạng file không được hỗ trợ.")
            st.stop()

        # Kiểm tra cột 'content'
        if 'content' not in df.columns:
            st.error("❌ File cần có cột tên là `content`.")
        else:
            with st.spinner("🔄 Đang xử lý và dự đoán..."):
                # Làm sạch dữ liệu và hiển thị kết quả
                df['clean_content'] = df['content'].apply(
                    lambda x: utils.func.clean_data(utils.func.translate_to_vietnamese(x)))

                # Hiển thị nội dung sau khi làm sạch
                st.subheader("💬 Nội dung sau khi làm sạch:")
                st.dataframe(df[['content', 'clean_content']])

                # Tiến hành dự đoán cảm xúc
                features = vectorizer.transform(df['clean_content'])
                df['sentiment'] = model.predict(features)

                st.success("✅ Xử lý thành công! Nhấn nút dưới để tải file kết quả.")
                st.download_button(
                    label="📥 Tải file kết quả",
                    data=df.to_csv(index=False).encode('utf-8-sig'),
                    file_name="ket_qua_du_doan.csv",
                    mime="text/csv"
                )

                # Hiển thị bảng kết quả dự đoán
                st.dataframe(df[['content', 'sentiment']])
    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý file: {e}")