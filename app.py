import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- CONFIG & ASSETS ---
st.set_page_config(
    page_title="Credit Risk AI Analyzer",
    page_icon="💳",
    layout="centered"
)

@st.cache_resource
def load_model():
    # ใช้ cache เพื่อไม่ให้โหลดโมเดลใหม่ทุกครั้งที่กดปุ่ม
    return joblib.load("model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("ไม่สามารถโหลดโมเดลได้ กรุณาตรวจสอบไฟล์ model.pkl")
    st.stop()

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("💳 ระบบวิเคราะห์ความเสี่ยงเครดิต (Credit Risk AI)")
st.info("เครื่องมือช่วยตัดสินใจเบื้องต้นในการประเมินความสามารถในการชำระหนี้ของลูกค้า โดยใช้โมเดล Machine Learning")

# --- MAIN INPUT SECTION ---
with st.expander("ℹ️ คำอธิบายข้อมูลที่ต้องระบุ (Feature Glossary)", expanded=False):
    st.markdown("""
    * **วงเงินเครดิต:** จำนวนเงินกู้สูงสุดที่ได้รับอนุมัติ
    * **สถานะการจ่ายล่าสุด:** ค่า 0 = จ่ายตรงเวลา, 1 = ค้างชำระ 1 เดือน, 2 = ค้างชำระ 2 เดือน (ติดลบคือจ่ายก่อนกำหนด)
    * **การศึกษา:** 1=ป.โท/เอก, 2=ป.ตรี, 3=มัธยมปลาย, 4=อื่นๆ
    * **สถานะสมรส:** 1=แต่งงานแล้ว, 2=โสด, 3=อื่นๆ
    """)

with st.container():
    st.subheader("📥 ข้อมูลโปรไฟล์ลูกค้า")
    col1, col2 = st.columns(2)
    
    with col1:
        limit_bal = st.number_input("💰 วงเงินเครดิต (TWD)", min_value=1000, max_value=1000000, value=50000, step=500)
        age = st.number_input("🎂 อายุ (ปี)", min_value=18, max_value=100, value=30)
        pay_0 = st.selectbox("📊 สถานะการจ่ายเดือนล่าสุด", 
                             options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                             help="เลือกตามประวัติการชำระเงินจริง")

    with col2:
        edu_map = {1: "ปริญญาโท/เอก", 2: "ปริญญาตรี", 3: "มัธยมปลาย", 4: "อื่นๆ"}
        education = st.selectbox("🎓 ระดับการศึกษา", options=list(edu_map.keys()), format_func=lambda x: edu_map[x])
        
        mar_map = {1: "แต่งงานแล้ว", 2: "โสด", 3: "อื่นๆ"}
        marriage = st.selectbox("💍 สถานะการสมรส", options=list(mar_map.keys()), format_func=lambda x: mar_map[x])

# --- PREDICTION LOGIC ---
st.markdown("---")
if st.button("🚀 เริ่มวิเคราะห์ความเสี่ยง", use_container_width=True):
    
    # 1. Input Validation (ป้องกันค่าที่ไม่สมเหตุสมผลเพิ่มเติมนอกจาก Slider/Number Input)
    if limit_bal < 5000 and age < 20:
        st.warning("⚠️ ข้อมูลดูมีความขัดแย้งกัน (วงเงินต่ำมากและอายุน้อย) โปรดตรวจสอบอีกครั้ง")

    # 2. Prepare Data
    try:
        columns = model.feature_names_in_
        # สร้าง DataFrame โดยตั้งค่าเริ่มต้นเป็น 0
        input_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
        
        # แมปค่าเข้ากับ Feature Names
        input_df['LIMIT_BAL'] = limit_bal
        input_df['AGE'] = age
        input_df['PAY_0'] = pay_0
        input_df['EDUCATION'] = education
        input_df['MARRIAGE'] = marriage
        
        # 3. Predict
        prob = model.predict_proba(input_df)[0][1]
        risk_level = "สูง" if prob > 0.6 else "ปานกลาง" if prob > 0.3 else "ต่ำ"
        color = "red" if prob > 0.6 else "orange" if prob > 0.3 else "green"

        # 4. Display Results
        st.subheader("🔍 ผลการวิเคราะห์")
        
        m1, m2 = st.columns(2)
        m1.metric("ระดับความเสี่ยง", risk_level)
        m2.metric("โอกาสในการผิดนัดชำระหนี้", f"{prob*100:.2f}%")
        
        st.write(f"**Confidence Score:**")
        st.progress(float(prob))
        
        if prob > 0.6:
            st.error(f"🚨 ระบบประเมินว่าลูกค้าท่านนี้มีความเสี่ยง **{risk_level}** ในการผิดนัดชำระหนี้")
        elif prob > 0.3:
            st.warning(f"⚠️ ลูกค้ามีความเสี่ยง **{risk_level}** ควรขอเอกสารค้ำประกันเพิ่มเติม")
        else:
            st.success(f"✅ ลูกค้ามีความเสี่ยง **{risk_level}** อยู่ในเกณฑ์ที่เชื่อถือได้")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการคำนวณ: {e}")

# --- FOOTER & DISCLAIMER ---
st.markdown("---")
st.markdown("""
<div style="font-size: 0.8rem; color: gray; line-height: 1.5;">
    <strong>⚠️ ข้อจำกัดความรับผิดชอบ (Disclaimer):</strong><br>
    ผลลัพธ์จาก AI นี้เป็นเพียงการคาดคะเนทางสถิติจากข้อมูลในอดีตเท่านั้น ไม่ใช่คำแนะนำทางการเงินหรือการรับประกันผลลัพธ์ 
    การตัดสินใจขั้นสุดท้ายควรผ่านการพิจารณาจากเจ้าหน้าที่สินเชื่อและนโยบายของสถาบันการเงิน
</div>
""", unsafe_allow_html=True)