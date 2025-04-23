# Predicting Customer Emotions From Product Reviews 💬

**A Midterm Project for the Data Analytics Course — April 2025**

---

## 📌 Overview

**Predicting Customer Emotions From Product Reviews** is a midterm project developed in April 2025 for the Data Analytics course. The application aims to automatically classify customer emotions (Satisfied / Unsatisfied) based on their product reviews written in Vietnamese. This helps businesses improve service quality and better understand customer feedback.

---

## 💡 Features

- 😄 **Emotion Classification**: Binary classification - *Satisfied* or *Unsatisfied*.
- 🧠 **Vietnamese NLP**: Analyze customer feedback in Vietnamese using NLP techniques.
- 📊 **Real-world Dataset**: Collected from product reviews on Tiki.vn.
- 🌐 **Interactive Web App**: Built with Streamlit for real-time prediction.

---

## 🧰 Technologies Used

- **Python 3.11.12**
- **Logistic Regression** for machine learning (88% accuracy).
- **Pandas, Scikit-learn** for data processing and modeling.
- **Streamlit** for building a user-friendly web interface.
- **Under-the-hood NLP**: Custom Vietnamese text preprocessing with normalization and word segmentation.

---

## 🚀 Live Demo

You can try the deployed application here:  
🔗 [https://predicting-constomer-emotion.streamlit.app/](https://predicting-constomer-emotion.streamlit.app/)

---
## ⚙️ Installation

To run this project locally, follow these steps:

```bash
# 1. Clone the repository
git clone https://github.com/nglhongphuong/Predicting-Consumer-Emotions-from-Product-Reviews.git

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```
---
## 📁 Project Structure
```
Your-Folder-Name/
├── app.py                      # Main entrypoint for Streamlit app
├── utils/
│   ├── __init__.py
│   └── func.py              
└── pages/
    ├── 1_Page_one.py          
    └── 2_Page_two.py
```
📚 [Official Docs – Upgrade Python on Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python)
