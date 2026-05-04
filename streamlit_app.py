import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.linear_model import LinearRegression
import plotly.express as px
import datetime
import io
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CẤU HÌNH TRANG & CSS DARK MODE (Tối ưu bo tròn & Màu sắc)
# ==========================================
st.set_page_config(page_title="VMP Digital Strategy Dashboard", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    /* Nền tối và font chữ */
    .stApp { background-color: #0f172a; color: #f8fafc; }
    
    /* Tùy chỉnh Sidebar */
    [data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid #334155; }
    
    /* Thẻ Card bo tròn */
    .custom-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        margin-bottom: 15px;
    }
    
    /* Thanh tiến độ tùy chỉnh */
    .progress-container {
        background-color: #334155;
        border-radius: 10px;
        height: 8px;
        width: 100%;
        margin-top: 10px;
        overflow: hidden;
    }
    .progress-bar-fill { height: 100%; border-radius: 10px; }
    
    /* Màu sắc trạng thái */
    .status-success { background-color: #10b981; } /* Xanh */
    .status-fail { background-color: #ef4444; }    /* Đỏ */
    
    /* Làm đẹp các Tab */
    .stTabs [data-baseweb="tab-list"] { background-color: #1e293b; border-radius: 10px; padding: 5px; }
    .stTabs [aria-selected="true"] { background-color: #38bdf8 !important; color: white !important; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. HÀM KẾT NỐI DỮ LIỆU (Giữ nguyên từ code của bạn)
# ==========================================
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
        goals_data[group] = {
            'method': best['name'], 'mad': round(best['mad'], 2),
            'monthly': [{'month': m, 'actual_2025': int(y[m-1]), 'target_2026': target_2026[m-1], 'actual_2026': int(actual_2026[m-1])} for m in range(1, 13)],
            'weekly_actual': weekly_actual
        }
    return goals_data

def get_growth_str(curr, prev):
    if prev == 0: return "▲ 100%" if curr > 0 else "0%"
    pct = ((curr - prev) / prev) * 100
    color = "#10b981" if pct >= 0 else "#ef4444"
    icon = "▲" if pct >= 0 else "▼"
    return f'<span style="color: {color};">{icon} {abs(round(pct))}% vs kỳ trước</span>'

try:
    df_main = load_data()
    goals = calculate_goals(df_main)
except Exception as e:
    st.error(f"Lỗi: {e}"); st.stop()

# ==========================================
# 3. SIDEBAR (Giữ nguyên thanh chỉnh ngày)
# ==========================================
with st.sidebar:
    st.title("VMP Digital")
    nav = st.radio("Điều hướng", ["📊 Tổng quan Overview", "🎯 Thiết lập Mục tiêu", "📂 Hành trình Data"])
    st.markdown("---")
    st.header("Lọc dữ liệu (Tổng quan)")
    start_date = st.date_input("Từ ngày", df_main['Day submit'].min())
    end_date = st.date_input("Đến ngày", df_main['Day submit'].max())
    
    # Logic lọc
    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    df = df_main[(df_main['Day submit'] >= start_dt) & (df_main['Day submit'] <= end_dt)]
    df_prev = df_main[(df_main['Day submit'] >= start_dt - pd.Timedelta(days=(end_dt-start_dt).days+1)) & 
                      (df_main['Day submit'] <= start_dt - pd.Timedelta(days=1))]

# ==========================================
# 4. GIAO DIỆN HIỂN THỊ
# ==========================================
if nav == "📊 Tổng quan Overview":
    st.title("📊 Tổng quan Overview")
    
    # Tính toán Metrics
    c1, c2, c3 = st.columns(3)
    metrics = [
        {"label": "TỔNG SL DATA", "curr": len(df), "prev": len(df_prev), "tgt": goals['Tổng hợp']['monthly'][datetime.datetime.now().month-1]['target_2026']},
        {"label": "DATA EBOOK", "curr": len(df[df['Nhóm Form'].str.contains('Ebook', na=False, case=False)]), 
         "prev": len(df_prev[df_prev['Nhóm Form'].str.contains('Ebook', na=False, case=False)]), "tgt": 10},
        {"label": "DATA EVENT", "curr": len(df[df['Nhóm Form'].str.contains('Event', na=False, case=False)]), 
         "prev": len(df_prev[df_prev['Nhóm Form'].str.contains('Event', na=False, case=False)]), "tgt": 182}
    ]
    
    cols = [c1, c2, c3]
    for i, m in enumerate(metrics):
        ratio = min(m['curr'] / m['tgt'], 1.0) if m['tgt'] > 0 else 0
        color_class = "status-success" if m['curr'] >= m['tgt'] else "status-fail"
        cols[i].markdown(f"""
            <div class="custom-card">
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8;">
                    <b>{m['label']}</b> {get_growth_str(m['curr'], m['prev'])}
                </div>
                <div style="font-size: 2.2rem; font-weight: bold; margin: 10px 0;">{m['curr']}</div>
                <div style="font-size: 0.8rem; font-weight: bold; color: {'#10b981' if m['curr']>=m['tgt'] else '#ef4444'};">
                    {round(m['curr']/m['tgt']*100, 1) if m['tgt'] > 0 else 0}%
                </div>
                <div class="progress-container"><div class="progress-bar-fill {color_class}" style="width: {ratio*100}%;"></div></div>
                <div style="text-align: right; font-size: 0.7rem; color: #64748b; margin-top: 5px;">Mục tiêu T4: {m['tgt']} Data</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### Biểu đồ Xu hướng")
    df_trend = df.copy(); df_trend['Ngày'] = df_trend['Day submit'].dt.date
    trend_data = df_trend.groupby(['Ngày', 'Nhóm Form']).size().reset_index(name='Số lượng')
    fig = px.line(trend_data, x='Ngày', y='Số lượng', color='Nhóm Form', markers=True, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

elif nav == "🎯 Thiết lập Mục tiêu":
    st.title("🎯 Thiết lập & Theo dõi Mục tiêu 2026")
    tabs = st.tabs(['📌 Mục tiêu Chung', '📊 Mục tiêu Event', '📘 Mục tiêu Ebook'])
    group_map = {0: 'Tổng hợp', 1: 'Event', 2: 'Ebook'}
    
    for i, tab in enumerate(tabs):
        with tab:
            g_data = goals[group_map[i]]
            st.markdown(f'<div style="background-color: #1e293b; padding: 10px; border-radius: 8px; color: #fbbf24; font-size: 0.85rem;">💡 Phương pháp: {g_data["method"]}</div>', unsafe_allow_html=True)
            
            st.subheader("Bảng 1: Theo dõi Mục tiêu 2026 theo Tháng")
            # Hiển thị bảng kèm Progress Bar thủ công
            h1, h2, h3, h4, h5 = st.columns([1.5, 2, 2, 2, 3])
            h1.write("**Tháng**"); h2.write("**Thực đạt 2025**"); h3.write("**Mục tiêu 2026**"); h4.write("**Thực đạt 2026**"); h5.write("**Tỷ lệ đạt (%)**")
            st.markdown("---")
            
            for m in g_data['monthly']:
                c1, c2, c3, c4, c5 = st.columns([1.5, 2, 2, 2, 3])
                c1.write(f"Tháng {m['month']}")
                c2.write(str(m['actual_2025']))
                c3.markdown(f"<span style='color: #fbbf24; font-weight: bold;'>{m['target_2026']}</span>", unsafe_allow_html=True)
                
                status_color = "#10b981" if m['actual_2026'] >= m['target_2026'] else "#ef4444"
                c4.markdown(f"<span style='color: {status_color}; font-weight: bold;'>{m['actual_2026']}</span>", unsafe_allow_html=True)
                
                ratio = min(m['actual_2026'] / m['target_2026'], 1.0) if m['target_2026'] > 0 else 0
                color_class = "status-success" if m['actual_2026'] >= m['target_2026'] else "status-fail"
                c5.markdown(f"""
                    <div style="font-size: 0.8rem; color: {status_color};">{round(m['actual_2026']/m['target_2026']*100, 1) if m['target_2026']>0 else 0}%</div>
                    <div class="progress-container"><div class="progress-bar-fill {color_class}" style="width: {ratio*100}%;"></div></div>
                    """, unsafe_allow_html=True)

elif nav == "📂 Hành trình Data":
    st.title("📂 Hành trình Khách hàng")
    c1, c2, c3 = st.columns(3)
    search = c1.text_input("🔍 Tìm Tên/Email")
    grp = c2.selectbox("📌 Nhóm Form", ["All", "Event", "Ebook", "Tiềm năng"])
    sts = c3.selectbox("🚦 Trạng thái", ["All", "Mới", "Cũ"])
    
    df_table = df_main.copy()
    if search: df_table = df_table[df_table['Họ tên'].str.contains(search, case=False, na=False) | df_table['Email'].str.contains(search, case=False, na=False)]
    if grp != "All": df_table = df_table[df_table['Nhóm Form'].str.contains(grp, case=False, na=False)]
    if sts != "All": df_table = df_table[df_table['Status'] == sts]
    
    df_table['Day submit'] = df_table['Day submit'].dt.strftime('%d/%m/%Y')
    st.dataframe(df_table[['Day submit', 'Họ tên', 'Email', 'Nhóm Form', 'Status', 'Touchpoints']], use_container_width=True, height=600)
