import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import datetime
import io
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CẤU HÌNH TRANG & CSS TÙY CHỈNH (VMP STYLE)
# ==========================================
st.set_page_config(page_title="VMP Digital Strategy Dashboard", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    /* Bo góc và bóng đổ cho các ô Metric */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    /* Chỉnh màu tiêu đề Sidebar */
    .css-17l2puu {
        font-weight: 700;
        color: #1e293b;
    }
    /* Làm đẹp các Tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #f8fafc;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #38bdf8 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- GIỮ NGUYÊN CÁC HÀM GET_SS_CLIENT, LOAD_DATA, CALCULATE_GOALS TỪ CODE TRƯỚC ---
def get_ss_client():
    scope = ['https://www.googleapis.com/auth/spreadsheets']
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
    return gspread.authorize(creds)

@st.cache_data(ttl=600)
def load_data():
    gc = get_ss_client()
    url = "https://docs.google.com/spreadsheets/d/19LEh9xAFKCHa63_sKO_tGZHyZZ23xriv0VEuDsURd2g/edit#gid=1147209162"
    sh = gc.open_by_url(url)
    data = sh.worksheet("Dữ liệu data mềm").get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])
    df['Day submit'] = pd.to_datetime(df['Day submit'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Day submit']).sort_values('Day submit')
    df['Touchpoints'] = df.groupby('Email')['Email'].transform('count')
    df['Is_New'] = ~df.duplicated(subset=['Email'], keep='first')
    df['Status'] = df['Is_New'].map({True: 'Mới', False: 'Cũ'})
    df['Year'] = df['Day submit'].dt.year
    df['Month'] = df['Day submit'].dt.month
    df['Week_in_Month'] = pd.cut(df['Day submit'].dt.day, bins=[0, 7, 14, 21, 31], labels=[1, 2, 3, 4], include_lowest=True).astype(int)
    return df

@st.cache_data(ttl=600)
def calculate_goals(df):
    groups = ['Tổng hợp', 'Event', 'Ebook']
    goals_data = {}
    for group in groups:
        df_g = df.copy() if group == 'Tổng hợp' else df[df['Nhóm Form'].str.contains(group, case=False, na=False)]
        actual_2025 = df_g[df_g['Year'] == 2025].groupby('Month').size().reindex(range(1, 13), fill_value=0).values
        actual_2026 = df_g[df_g['Year'] == 2026].groupby('Month').size().reindex(range(1, 13), fill_value=0).values
        y = actual_2025.astype(float)
        if np.sum(y) < 5:
            y = np.array([max(10.0, float(v)) for v in actual_2026])
            if np.sum(y) == 0: y = np.array([10.0] * 12)
        ma_pred = np.array([np.mean(y[max(0, i-3):i]) if i > 0 else y[0] for i in range(1, 13)])
        mad_ma = np.mean(np.abs(ma_pred - y))
        try:
            model_es = SimpleExpSmoothing(y, initialization_method="estimated").fit(smoothing_level=0.3, optimized=False)
            es_pred = model_es.fittedvalues
            mad_es = np.mean(np.abs(es_pred - y))
        except:
            es_pred, mad_es = ma_pred, 9999
        try:
            X = np.arange(12).reshape(-1, 1)
            model_lr = LinearRegression().fit(X, y)
            lr_pred = model_lr.predict(X)
            mad_lr = np.mean(np.abs(lr_pred - y))
        except:
            lr_pred, mad_lr = ma_pred, 9999
        models = [{'name': 'Trung bình động (MA)', 'pred': ma_pred, 'mad': mad_ma},
                  {'name': 'San bằng hàm mũ (ES)', 'pred': es_pred, 'mad': mad_es},
                  {'name': 'Hồi quy tuyến tính (LR)', 'pred': lr_pred, 'mad': mad_lr}]
        best = min(models, key=lambda x: x['mad'])
        target_2026 = [int(max(p * 1.1, a * 1.15, 10)) for p, a in zip(best['pred'], y)]
        weekly_actual = {m: {w: 0 for w in range(1, 5)} for m in range(1, 13)}
        df_2026 = df_g[df_g['Year'] == 2026]
        for m in range(1, 13):
            month_data = df_2026[df_2026['Month'] == m].groupby('Week_in_Month').size().to_dict()
            for w, val in month_data.items():
                weekly_actual[m][w] = val
        goals_data[group] = {'method': best['name'], 'mad': round(best['mad'], 2),
                             'monthly': [{'month': m, 'actual_2025': int(y[m-1]), 'target_2026': target_2026[m-1], 'actual_2026': int(actual_2026[m-1])} for m in range(1, 13)],
                             'weekly_actual': weekly_actual}
    return goals_data

try:
    df_main = load_data()
    goals = calculate_goals(df_main)
except Exception as e:
    st.error(f"Lỗi: {e}")
    st.stop()

# ==========================================
# 2. SIDEBAR - ĐƯA LOGO VÀ BỘ LỌC VÀO GỌN GÀNG
# ==========================================
with st.sidebar:
    st.title("🚀 VMP DIGITAL")
    st.markdown("---")
    nav = st.radio("Chuyên mục Dashboard", ["📊 Tổng quan Overview", "🎯 Theo dõi Mục tiêu", "📂 Tra cứu Data"])
    st.markdown("---")
    
    st.subheader("🗓️ Bộ lọc thời gian")
    start_date = st.date_input("Từ ngày", df_main['Day submit'].min())
    end_date = st.date_input("Đến ngày", df_main['Day submit'].max())
    
    st.markdown("---")
    st.caption("© 2026 VMP Academy Academy")

# Xử lý Logic Data (Giữ nguyên)
start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
df = df_main[(df_main['Day submit'] >= start_dt) & (df_main['Day submit'] <= end_dt)]
diff_days = (end_dt - start_dt).days + 1
prev_end = start_dt - pd.Timedelta(days=1)
prev_start = prev_end - pd.Timedelta(days=diff_days - 1)
df_prev = df_main[(df_main['Day submit'] >= prev_start) & (df_main['Day submit'] <= prev_end)]

def get_growth_str(curr, prev):
    if prev == 0: return "100%"
    pct = ((curr - prev) / prev) * 100
    return f"{round(pct)}%"

# ==========================================
# 3. GIAO DIỆN CHÍNH
# ==========================================
if nav == "📊 Tổng quan Overview":
    st.header("📊 Hệ thống Theo dõi Chiến dịch Digital")
    
    # Khu vực Metrics xịn sò
    m1, m2, m3, m4 = st.columns(4)
    curr_total, prev_total = len(df), len(df_prev)
    curr_new = len(df[df['Status'] == 'Mới'])
    
    m1.metric("Tổng SL Data", f"{curr_total:,}", get_growth_str(curr_total, prev_total))
    m2.metric("Data Mới", f"{curr_new:,}", delta_color="normal")
    m3.metric("Data Event", len(df[df['Nhóm Form'].str.contains('Event', na=False, case=False)]))
    m4.metric("Data Ebook", len(df[df['Nhóm Form'].str.contains('Ebook', na=False, case=False)]))

    st.markdown("### 📈 Phân tích Xu hướng")
    df_trend = df.copy()
    df_trend['Ngày'] = df_trend['Day submit'].dt.date
    trend_data = df_trend.groupby(['Ngày', 'Nhóm Form']).size().reset_index(name='Số lượng')
    
    # Biểu đồ Plotly với Style sạch sẽ
    fig = px.area(trend_data, x='Ngày', y='Số lượng', color='Nhóm Form', 
                  color_discrete_sequence=['#38bdf8', '#fbbf24', '#94a3b8'],
                  template="plotly_white")
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Insight Cards
    st.markdown("### 🤖 Trợ lý AI Phân tích Insight")
    c1, c2 = st.columns(2)
    top_event = df[df['Nhóm Form'].str.contains('Event', na=False, case=False)]['Tên Form'].value_counts()
    
    with c1:
        st.markdown(f"""
        <div style="background-color: #f0f9ff; padding: 20px; border-radius: 10px; border-left: 5px solid #0ea5e9;">
            <h4 style="margin-top:0;">🌟 Insight Event nổi bật</h4>
            <p>Nguồn mang lại hiệu quả cao nhất là <b>{top_event.index[0] if not top_event.empty else 'N/A'}</b>.</p>
            <p>Đề xuất: Tăng ngân sách Ads cho nguồn này vào khung giờ 19h-21h.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown(f"""
        <div style="background-color: #fffbeb; padding: 20px; border-radius: 10px; border-left: 5px solid #f59e0b;">
            <h4 style="margin-top:0;">📌 Lưu ý Hành trình Khách hàng</h4>
            <p>Tỷ lệ khách hàng cũ quay lại đăng ký thêm form mới đạt <b>{round((1-curr_new/max(curr_total,1))*100)}%</b>.</p>
            <p>Cần kịch bản Re-marketing riêng cho nhóm này.</p>
        </div>
        """, unsafe_allow_html=True)

elif nav == "🎯 Theo dõi Mục tiêu":
    st.header("🎯 Mục tiêu KPI 2026")
    t1, t2, t3 = st.tabs(["📌 Tổng hợp", "🏟️ Event", "📚 Ebook"])
    
    group_map = {0: 'Tổng hợp', 1: 'Event', 2: 'Ebook'}
    for i, tab in enumerate([t1, t2, t3]):
        with tab:
            gn = group_map[i]
            gd = goals[gn]
            
            # Progress Bar lớn
            target_year = sum([m['target_2026'] for m in gd['monthly']])
            actual_year = sum([m['actual_2026'] for m in gd['monthly']])
            progress = min(actual_year / max(target_year, 1), 1.0)
            
            col_p1, col_p2 = st.columns([3, 1])
            col_p1.write(f"**Tiến độ năm 2026:** {actual_year}/{target_year} Data")
            col_p1.progress(progress)
            col_p2.metric("Tỉ lệ hoàn thành", f"{round(progress*100, 1)}%")

            st.markdown("#### Chi tiết theo từng Tháng")
            df_m = pd.DataFrame(gd['monthly'])
            # Tạo bảng hiển thị đẹp
            fig_table = go.Figure(data=[go.Table(
                header=dict(values=['Tháng', 'Target 2026', 'Actual 2026', 'Tiến độ'],
                            fill_color='#38bdf8', align='left', font=dict(color='white', size=14)),
                cells=dict(values=[df_m['month'], df_m['target_2026'], df_m['actual_2026'], 
                                   (df_m['actual_2026']/df_m['target_2026'].replace(0,1)*100).round(1).astype(str) + '%'],
                           fill_color='#f8fafc', align='left'))])
            fig_table.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
            st.plotly_chart(fig_table, use_container_width=True)

elif nav == "📂 Tra cứu Data":
    st.header("📂 Hệ thống Quản trị Dữ liệu")
    
    # Bộ lọc nhanh trong trang
    f1, f2 = st.columns([2, 1])
    search = f1.text_input("🔍 Tìm kiếm theo tên hoặc email...")
    export_btn = f2.button("🚀 Chuẩn bị file Excel")
    
    df_view = df_main.copy()
    if search:
        df_view = df_view[df_view['Họ tên'].str.contains(search, case=False) | df_view['Email'].str.contains(search, case=False)]
    
    st.dataframe(df_view[['Day submit', 'Họ tên', 'Email', 'Nhóm Form', 'Status', 'Touchpoints']].sort_values('Day submit', ascending=False), 
                 use_container_width=True, height=500)
