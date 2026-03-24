# 💳 Credit Risk Prediction App

## 📌 Overview
โปรเจคนี้เป็นการใช้ Machine Learning เพื่อทำนายว่าลูกค้าจะมีโอกาสเบี้ยวหนี้หรือไม่  
โดยมีเป้าหมายเพื่อช่วยในการตัดสินใจอนุมัติสินเชื่อ

---

## 🎯 Problem Statement
ธนาคารไม่สามารถรู้ล่วงหน้าว่าลูกค้าคนใดจะไม่ชำระหนี้  
จึงต้องใช้ Machine Learning เพื่อช่วยประเมินความเสี่ยง

---

## 📊 Dataset
- Credit Card Default Dataset
- จำนวนข้อมูล: ~30,000 แถว
- Features:
  - LIMIT_BAL (วงเงิน)
  - AGE (อายุ)
  - PAY_0 (ประวัติการชำระ)
  - BILL_AMT1-6
  - PAY_AMT1-6
- Target:
  - default.payment.next.month (0 = ไม่เบี้ยว, 1 = เบี้ยว)

---

## 🔍 Exploratory Data Analysis (EDA)
- พบว่า dataset มี class imbalance
- ลูกค้าที่เคยค้างชำระมีโอกาสเบี้ยวสูง
- วงเงินมีผลต่อความเสี่ยง

---

## 🤖 Model Development
โมเดลที่ใช้:
- Logistic Regression
- Random Forest
- XGBoost (เลือกใช้)

### 🎯 Evaluation Metric
- Recall (เน้นจับลูกค้าที่มีความเสี่ยง)

---

## 📈 Results
- XGBoost ให้ผลลัพธ์ดีที่สุด
- สามารถทำนายความเสี่ยงได้อย่างมีประสิทธิภาพ

---

## 🌐 Deployment
แอปถูก deploy บน Streamlit Cloud

👉 [Click to open app](https://your-app-link.streamlit.app)

---

## 🖥️ How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py