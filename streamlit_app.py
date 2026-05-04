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
# 1. CẤU HÌNH TRANG & KẾT NỐI DỮ LIỆU
# ==========================================
st.set_page_config(page_title="VMP Digital Strategy Dashboard", page_icon="📈", layout="wide")

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
            
        models = [
            {'name': 'Trung bình động (MA)', 'pred': ma_pred, 'mad': mad_ma},
            {'name': 'San bằng hàm mũ (ES)', 'pred': es_pred, 'mad': mad_es},
            {'name': 'Hồi quy tuyến tính (LR)', 'pred': lr_pred, 'mad': mad_lr}
        ]
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
    if prev == 0: return f"▲ 100%" if curr > 0 else "- Không đổi"
    pct = ((curr - prev) / prev) * 100
    if pct > 0: return f"▲ {round(pct)}% vs kỳ trước"
    elif pct < 0: return f"▼ {abs(round(pct))}% vs kỳ trước"
    return "- Không đổi"

try:
    df_main = load_data()
    goals = calculate_goals(df_main)
except Exception as e:
    st.error(f"Lỗi kết nối dữ liệu: {e}")
    st.stop()

# ==========================================
# 2. MENU & BỘ LỌC CHUNG
# ==========================================
st.sidebar.title("VMP Digital")
st.sidebar.caption("B2B Strategy Dashboard")

nav = st.sidebar.radio("Điều hướng", ["📊 Tổng quan Overview", "🎯 Thiết lập Mục tiêu", "📂 Hành trình Data"])
st.sidebar.markdown("---")

st.sidebar.header("Lọc dữ liệu (Tổng quan)")
min_date, max_date = df_main['Day submit'].min(), df_main['Day submit'].max()
start_date = st.sidebar.date_input("Từ ngày", min_date)
end_date = st.sidebar.date_input("Đến ngày", max_date)

start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)
df = df_main[(df_main['Day submit'] >= start_dt) & (df_main['Day submit'] <= end_dt)]

diff_days = (end_dt - start_dt).days + 1
prev_end = start_dt - pd.Timedelta(days=1)
prev_start = prev_end - pd.Timedelta(days=diff_days - 1)
df_prev = df_main[(df_main['Day submit'] >= prev_start) & (df_main['Day submit'] <= prev_end)]

# Xuất Excel
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    df.to_excel(writer, index=False, sheet_name='Data')
st.sidebar.download_button(label="⬇ Xuất Excel Data Lọc", data=buffer, file_name="VMP_Data.xlsx", mime="application/vnd.ms-excel")

# ==========================================
# 3. GIAO DIỆN CHÍNH
# ==========================================
if nav == "📊 Tổng quan Overview":
    st.title("📊 Tổng quan Overview")
    
    c1, c2, c3 = st.columns(3)
    curr_total, prev_total = len(df), len(df_prev)
    curr_ebook = len(df[df['Nhóm Form'].str.contains('Ebook', na=False, case=False)])
    prev_ebook = len(df_prev[df_prev['Nhóm Form'].str.contains('Ebook', na=False, case=False)])
    curr_event = len(df[df['Nhóm Form'].str.contains('Event', na=False, case=False)])
    prev_event = len(df_prev[df_prev['Nhóm Form'].str.contains('Event', na=False, case=False)])
    
    c1.metric("Tổng SL Data", curr_total, get_growth_str(curr_total, prev_total))
    c2.metric("Data Ebook", curr_ebook, get_growth_str(curr_ebook, prev_ebook))
    c3.metric("Data Event", curr_event, get_growth_str(curr_event, prev_event))
    
    st.markdown("### Biểu đồ Xu hướng")
    df_trend = df.copy()
    df_trend['Ngày'] = df_trend['Day submit'].dt.date
    trend_data = df_trend.groupby(['Ngày', 'Nhóm Form']).size().reset_index(name='Số lượng')
    fig = px.line(trend_data, x='Ngày', y='Số lượng', color='Nhóm Form', markers=True, color_discrete_sequence=['#38bdf8', '#fbbf24', '#94a3b8'])
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 🤖 Insight Tổng quan")
    ic1, ic2 = st.columns(2)
    top_event = df[df['Nhóm Form'].str.contains('Event', na=False, case=False)]['Tên Form'].value_counts()
    top_ebook = df[df['Nhóm Form'].str.contains('Ebook', na=False, case=False)]['Tên Form'].value_counts()
    
    with ic1:
        st.info(f"**INSIGHT EVENT** (Tỷ trọng: {round(curr_event/max(curr_total,1)*100)}%)\n\n"
                f"Nguồn tốt nhất: **{top_event.index[0] if not top_event.empty else 'Chưa có'}** ({top_event.values[0] if not top_event.empty else 0} Data)")
    with ic2:
        st.warning(f"**INSIGHT EBOOK** (Tỷ trọng: {round(curr_ebook/max(curr_total,1)*100)}%)\n\n"
                   f"Nguồn tốt nhất: **{top_ebook.index[0] if not top_ebook.empty else 'Chưa có'}** ({top_ebook.values[0] if not top_ebook.empty else 0} Data)")

elif nav == "🎯 Thiết lập Mục tiêu":
    st.title("🎯 Thiết lập & Theo dõi Mục tiêu 2026")
    
    tabs = st.tabs(['📌 Mục tiêu Chung', '📊 Mục tiêu Event', '📘 Mục tiêu Ebook'])
    group_map = {0: 'Tổng hợp', 1: 'Event', 2: 'Ebook'}
    
    for i, tab in enumerate(tabs):
        with tab:
            g_name = group_map[i]
            g_data = goals[g_name]
            st.caption(f"💡 Phương pháp dự báo: **{g_data['method']}** (MAD: {g_data['mad']})")
            
            st.subheader("Bảng 1: Mục tiêu theo Tháng")
            df_monthly = pd.DataFrame(g_data['monthly'])
            df_monthly['Tiến độ'] = (df_monthly['actual_2026'] / df_monthly['target_2026'].replace(0, 1) * 100).round(1).astype(str) + "%"
            df_monthly.columns = ['Tháng', 'Thực đạt 2025', 'Mục tiêu 2026', 'Thực đạt 2026', 'Tiến độ (%)']
            st.dataframe(df_monthly, use_container_width=True)
            
            st.subheader("Bảng 2: Theo dõi Tiến độ")
            view_level = st.radio("Cấp độ xem", ["Theo Quý", "Theo Tháng"], horizontal=True, key=f"view_{i}")
            
            if view_level == "Theo Quý":
                q = st.selectbox("Chọn Quý", [1, 2, 3, 4], key=f"q_{i}")
                months = [q*3-2, q*3-1, q*3]
                t_target = sum([g_data['monthly'][m-1]['target_2026'] for m in months])
                t_actual = sum([g_data['monthly'][m-1]['actual_2026'] for m in months])
                
                st.markdown(f"**TỔNG QUAN QUÝ {q}/2026** - Mục tiêu: **{t_target}** | Đạt: **{t_actual}**")
                
                track_data = []
                for m in months:
                    tgt = g_data['monthly'][m-1]['target_2026']
                    act = g_data['monthly'][m-1]['actual_2026']
                    track_data.append({"Thời gian": f"Tháng {m}", "Mục tiêu": tgt, "Thực đạt": act, "Tiến độ (%)": round(act/max(tgt,1)*100, 1)})
                st.table(track_data)
                
            else:
                m = st.selectbox("Chọn Tháng", list(range(1, 13)), key=f"m_{i}")
                tgt = g_data['monthly'][m-1]['target_2026']
                act = g_data['monthly'][m-1]['actual_2026']
                st.markdown(f"**TỔNG THÁNG {m}/2026** - Mục tiêu: **{tgt}** | Đạt: **{act}**")
                
                w_target = int(np.ceil(tgt / 4))
                w_data = []
                for w in range(1, 5):
                    w_act = g_data['weekly_actual'][m].get(w, 0)
                    w_data.append({"Thời gian": f"Tuần {w}", "Mục tiêu": w_target, "Thực đạt": w_act, "Tiến độ (%)": round(w_act/max(w_target,1)*100, 1)})
                st.table(w_data)

            st.markdown("### 🤖 Phân tích AI & Ghi chú của Planner")
            a1, a2 = st.columns(2)
            with a1:
                # Logic text AI đơn giản hóa tương đương bản JS
                st.success(f"✅ Đã thu về {t_actual if view_level=='Theo Quý' else act} Data.")
                st.warning(f"⚠️ Cần thu thập thêm để đạt mốc {t_target if view_level=='Theo Quý' else tgt}.")
                st.info("⚙️ Hành động: Theo dõi Real-time, đẩy mạnh Paid Ads Facebook/LinkedIn, A/B Testing.")
            with a2:
                st.text_area("📝 Đánh giá & Ghi chú thực tế nội bộ", height=150, key=f"note_{i}", placeholder="Nhập kế hoạch hành động vào đây...")

elif nav == "📂 Hành trình Data":
    st.title("📂 Hành trình Khách hàng")
    
    col1, col2, col3 = st.columns(3)
    search = col1.text_input("🔍 Tìm Tên/Email")
    grp = col2.selectbox("📌 Nhóm Form", ["All", "Event", "Ebook", "Tiềm năng"])
    sts = col3.selectbox("🚦 Trạng thái", ["All", "Mới", "Cũ"])
    
    df_table = df_main.copy()
    if search:
        df_table = df_table[df_table['Họ tên'].str.contains(search, case=False, na=False) | df_table['Email'].str.contains(search, case=False, na=False)]
    if grp != "All":
        df_table = df_table[df_table['Nhóm Form'].str.contains(grp, case=False, na=False)]
    if sts != "All":
        df_table = df_table[df_table['Status'] == sts]
        
    df_table['Day submit'] = df_table['Day submit'].dt.strftime('%d/%m/%Y')
    st.dataframe(df_table[['Day submit', 'Họ tên', 'Email', 'Nhóm Form', 'Status', 'Touchpoints']], use_container_width=True, height=600)
