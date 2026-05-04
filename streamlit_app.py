import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.linear_model import LinearRegression
import plotly.express as px
import json

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="VMP Digital Strategy", layout="wide")

# --- KẾT NỐI GOOGLE SHEETS ---
def get_ss_client():
    scope = ['https://www.googleapis.com/auth/spreadsheets']
    # Lấy thông tin bảo mật từ Streamlit Secrets
    creds_dict = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    return gspread.authorize(creds)

@st.cache_data(ttl=600)
def load_data():
    gc = get_ss_client()
    url = "https://docs.google.com/spreadsheets/d/19LEh9xAFKCHa63_sKO_tGZHyZZ23xriv0VEuDsURd2g/edit#gid=1147209162"
    sh = gc.open_by_url(url)
    worksheet = sh.worksheet("Dữ liệu data mềm")
    data = worksheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])
    
    # Tiền xử lý dữ liệu (Giống 100% bản Colab của bạn)
    df['Day submit'] = pd.to_datetime(df['Day submit'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Day submit']).sort_values('Day submit')
    df['Touchpoints'] = df.groupby('Email')['Email'].transform('count')
    df['Is_New'] = ~df.duplicated(subset=['Email'], keep='first')
    df['Status'] = df['Is_New'].map({True: 'Mới', False: 'Cũ'})
    return df

# --- LOGIC DỰ BÁO ---
def calculate_target(df_group):
    # Lấy số liệu theo tháng
    df_group['Month'] = df_group['Day submit'].dt.month
    ts = df_group.groupby('Month').size().reindex(range(1, 13), fill_value=0)
    y = ts.values.astype(float)
    
    # Dự báo đơn giản (Smoothing)
    try:
        model = SimpleExpSmoothing(y).fit(smoothing_level=0.3, optimized=False)
        pred = model.forecast(1)[0]
    except:
        pred = np.mean(y[-3:]) if len(y) > 0 else 10
        
    # Mục tiêu +15% như logic cũ của bạn
    target = int(max(pred * 1.15, 10))
    return target

# --- GIAO DIỆN WEB ---
st.title("🚀 VMP B2B Digital Strategy Dashboard")
st.markdown("---")

try:
    df = load_data()
    
    # Sidebar lọc
    st.sidebar.header("Bộ lọc chiến dịch")
    group_choice = st.sidebar.selectbox("Chọn nhóm dữ liệu", ["Tổng hợp", "Event", "Ebook"])
    
    if group_choice != "Tổng hợp":
        df_display = df[df['Nhóm Form'].str.contains(group_choice, case=False, na=False)]
    else:
        df_display = df

    # Hiển thị KPI
    target_val = calculate_target(df_display)
    actual_val = len(df_display[df_display['Day submit'].dt.year == 2026]) # Giả định năm hiện tại
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng Data (Filtered)", len(df_display))
    c2.metric("Mục tiêu dự báo tháng tới", f"{target_val} Data")
    c3.metric("Tiến độ thực hiện", f"{(actual_val/target_val*100 if target_val>0 else 0):.1f}%")

    # Biểu đồ xu hướng
    st.subheader(f"Biểu đồ tăng trưởng: {group_choice}")
    df_chart = df_display.groupby(df_display['Day submit'].dt.date).size().reset_index(name='Số lượng')
    fig = px.area(df_chart, x='Day submit', y='Số lượng', color_discrete_sequence=['#38bdf8'])
    st.plotly_chart(fig, use_container_width=True)

    # Bảng dữ liệu
    st.subheader("📂 Chi tiết data khách hàng")
    st.dataframe(df_display, use_container_width=True)

except Exception as e:
    st.error(f"Đang chờ cấu hình Secrets hoặc lỗi kết nối: {e}")
    st.info("Vui lòng đảm bảo bạn đã cấp quyền cho Email Service Account trong Google Sheets.")
