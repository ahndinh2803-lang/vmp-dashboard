import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN DARK MODE & CSS TÙY CHỈNH
# ==========================================
st.set_page_config(page_title="VMP Digital Dashboard", layout="wide")

st.markdown("""
    <style>
    /* Nền tối toàn trang */
    .stApp { background-color: #0f172a; color: #f8fafc; }
    
    /* Tùy chỉnh các thẻ chỉ số (Cards) */
    .custom-card {
        background-color: #1e293b;
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #334155;
        margin-bottom: 10px;
    }
    
    /* Thanh tiến độ mini dưới Card */
    .mini-progress-bg {
        background-color: #334155;
        height: 6px;
        border-radius: 3px;
        margin-top: 10px;
        width: 100%;
    }
    .mini-progress-fill {
        height: 6px;
        border-radius: 3px;
    }
    
    /* Màu sắc theo trạng thái */
    .text-success { color: #10b981; }
    .text-danger { color: #ef4444; }
    .text-warning { color: #f59e0b; }
    
    /* Làm đẹp Tab giống ảnh mẫu */
    .stTabs [data-baseweb="tab-list"] { background-color: #1e293b; border-radius: 10px; padding: 5px; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; }
    .stTabs [aria-selected="true"] { background-color: #38bdf8 !important; color: white !important; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. HÀM KẾT NỐI & XỬ LÝ DATA (GIỮ NGUYÊN LOGIC)
# ==========================================
def get_ss_client():
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], 
            scopes=['https://www.googleapis.com/auth/spreadsheets'])
    return gspread.authorize(creds)

@st.cache_data(ttl=600)
def load_data():
    gc = get_ss_client()
    url = "https://docs.google.com/spreadsheets/d/19LEh9xAFKCHa63_sKO_tGZHyZZ23xriv0VEuDsURd2g/edit#gid=1147209162"
    sh = gc.open_by_url(url)
    data = sh.worksheet("Dữ liệu data mềm").get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])
    df['Day submit'] = pd.to_datetime(df['Day submit'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Day submit'])
    df['Month'] = df['Day submit'].dt.month
    df['Year'] = df['Day submit'].dt.year
    return df

try:
    df_main = load_data()
except Exception as e:
    st.error(f"Lỗi: {e}")
    st.stop()

# ==========================================
# 3. GIAO DIỆN TỔNG QUAN (CHẾ ĐỘ THẺ BO TRÒN + PROGRESS)
# ==========================================
st.sidebar.title("🚀 VMP DIGITAL")
nav = st.sidebar.radio("Menu", ["📊 Tổng quan", "🎯 Mục tiêu"])

if nav == "📊 Tổng quan":
    st.title("📊 Tổng quan Overview")
    
    # Giả định Target tháng hiện tại (Tháng 4/2026) dựa trên ảnh mẫu
    targets = {"Tổng": 188, "Ebook": 10, "Event": 182}
    
    df_now = df_main[(df_main['Year'] == 2026) & (df_main['Month'] == 4)]
    actuals = {
        "Tổng": len(df_now),
        "Ebook": len(df_now[df_now['Nhóm Form'].str.contains('Ebook', na=False, case=False)]),
        "Event": len(df_now[df_now['Nhóm Form'].str.contains('Event', na=False, case=False)])
    }

    cols = st.columns(3)
    for i, label in enumerate(["Tổng", "Ebook", "Event"]):
        with cols[i]:
            act, tgt = actuals[label], targets[label]
            ratio = min(act / tgt, 1.0) if tgt > 0 else 0
            color = "#10b981" if act >= tgt else "#ef4444"
            
            st.markdown(f"""
                <div class="custom-card">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-size: 0.8rem; font-weight: bold; color: #94a3b8;">TỔNG SL DATA {label.upper()}</span>
                        <span style="font-size: 0.7rem; color: #10b981;">▲ Tăng trưởng</span>
                    </div>
                    <div style="font-size: 2.5rem; font-weight: bold; margin: 10px 0;">{act}</div>
                    <div style="color: {color}; font-size: 0.8rem; font-weight: bold;">{round(act/tgt*100, 1) if tgt > 0 else 0}%</div>
                    <div class="mini-progress-bg">
                        <div class="mini-progress-fill" style="width: {ratio*100}%; background-color: {color};"></div>
                    </div>
                    <div style="text-align: right; font-size: 0.7rem; color: #94a3b8; margin-top: 5px;">Mục tiêu T4: {tgt} Data</div>
                </div>
                """, unsafe_allow_html=True)

    # Biểu đồ xu hướng (Style Dark)
    st.markdown("### Biểu đồ Xu hướng Data theo NGÀY")
    df_trend = df_now.copy()
    df_trend['Ngày'] = df_trend['Day submit'].dt.date
    trend_data = df_trend.groupby(['Ngày', 'Nhóm Form']).size().reset_index(name='Số lượng')
    fig = px.line(trend_data, x='Ngày', y='Số lượng', color='Nhóm Form', template="plotly_dark")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. TRANG MỤC TIÊU (PROGRESS BAR TRONG BẢNG)
# ==========================================
elif nav == "🎯 Mục tiêu":
    st.header("🎯 Theo dõi Mục tiêu 2026")
    t1, t2, t3 = st.tabs(["🚀 Mục tiêu Chung", "📊 Mục tiêu Event", "📘 Mục tiêu Ebook"])
    
    with t1:
        st.markdown("""<div style="background-color: #334155; padding: 10px; border-radius: 8px; font-size: 0.8rem; color: #f59e0b;">
            💡 Phương pháp: <b>Trung bình động (MA)</b> - Mục tiêu năm nay phải lớn hơn thực đạt năm trước.
        </div>""", unsafe_allow_html=True)
        
        # Tạo bảng mục tiêu giả định theo ảnh mẫu để hiển thị Progress Bar
        data_rows = [
            {"Tháng": "Tháng 1", "2025": 0, "2026 Tgt": 10, "2026 Act": 90},
            {"Tháng": "Tháng 2", "2025": 93, "2026 Tgt": 106, "2026 Act": 38},
            {"Tháng": "Tháng 3", "2025": 396, "2026 Tgt": 455, "2026 Act": 124},
            {"Tháng": "Tháng 4", "2025": 25, "2026 Tgt": 188, "2026 Act": 206},
        ]
        
        # Hiển thị bảng thủ công để có Progress Bar giống mẫu
        cols_h = st.columns([2, 2, 2, 2, 3])
        cols_h[0].write("**Tháng**")
        cols_h[1].write("**Thực đạt 2025**")
        cols_h[2].write("**Mục tiêu 2026**")
        cols_h[3].write("**Thực đạt 2026**")
        cols_h[4].write("**Tỷ lệ đạt (%)**")
        
        st.markdown("---")
        for row in data_rows:
            c = st.columns([2, 2, 2, 2, 3])
            c[0].write(row["Tháng"])
            c[1].write(str(row["2025"]))
            c[2].markdown(f"<span style='color: #f59e0b; font-weight: bold;'>{row['2026 Tgt']}</span>", unsafe_allow_html=True)
            
            # Đổi màu số thực đạt nếu chưa đạt mục tiêu
            status_color = "#10b981" if row["2026 Act"] >= row["2026 Tgt"] else "#ef4444"
            c[3].markdown(f"<span style='color: {status_color};'>{row['2026 Act']}</span>", unsafe_allow_html=True)
            
            # Progress Bar trong bảng
            ratio = min(row["2026 Act"] / row["2026 Tgt"], 1.0) if row["2026 Tgt"] > 0 else 0
            c[4].markdown(f"""
                <div style="font-size: 0.8rem; color: {status_color};">{round(ratio*100, 1)}%</div>
                <div class="mini-progress-bg">
                    <div class="mini-progress-fill" style="width: {ratio*100}%; background-color: {status_color};"></div>
                </div>
                """, unsafe_allow_html=True)
