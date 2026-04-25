import streamlit as st
import cv2
import os
import uuid
import numpy as np
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64

from db import (
    init_db, db_available, save_to_db,
    get_history, get_total_count, get_stats,
    get_confidence_data, get_daily_counts,
    search_similar, get_image_display
)
from agent_core import ImageAgent, Pipeline

# =========================================================
# CUSTOM CSS
# =========================================================
st.set_page_config(layout="wide", page_title="AI Image Detector")

st.markdown("""
<style>
    /* ===== GLOBAL ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #c4c4e0 !important;
    }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio div {
        color: #c4c4e0 !important;
        font-weight: 400;
    }
    [data-testid="stSidebar"] .stRadio [aria-checked="true"] label,
    [data-testid="stSidebar"] .stRadio [aria-checked="true"] {
        color: #ffffff !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
        padding: 0.5rem 0;
    }
    .sidebar-subtitle {
        font-size: 0.75rem;
        color: #8888aa;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #3a3a6e, transparent);
        margin: 1rem 0;
    }
    .sidebar-status {
        font-size: 0.8rem;
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        background: rgba(255,255,255,0.05);
    }
    .sidebar-status.online {
        border-left: 3px solid #51cf66;
    }
    .sidebar-status.offline {
        border-left: 3px solid #ff6b6b;
    }

    /* ===== PAGE TITLE ===== */
    .page-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a3e;
        letter-spacing: -0.5px;
        margin-bottom: 0.25rem;
    }
    .page-desc {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    /* ===== BADGE ===== */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-fake { background: #fff0f0; color: #e03131; border: 1px solid #ffc9c9; }
    .badge-real { background: #ebfbee; color: #2f9e44; border: 1px solid #b2f2bb; }
    .badge-high { background: #fff0f0; color: #e03131; border: 1px solid #ffc9c9; }
    .badge-medium { background: #fff9db; color: #e67700; border: 1px solid #ffe066; }
    .badge-low { background: #fff4e6; color: #d9480f; border: 1px solid #ffd8a8; }
    .badge-safe { background: #ebfbee; color: #2f9e44; border: 1px solid #b2f2bb; }

    /* ===== RESULT CARD ===== */
    .result-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    .result-item {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
    .result-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #868e96;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }
    .result-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1a1a3e;
    }

    /* ===== HISTORY CARD ===== */
    .history-card {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        overflow: hidden;
        transition: box-shadow 0.2s;
    }
    .history-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .history-meta {
        padding: 0.75rem 1rem;
        border-top: 1px solid #f1f3f5;
        font-size: 0.8rem;
        color: #868e96;
    }

    /* ===== SIMILAR CARD ===== */
    .similarity-bar {
        width: 100%;
        height: 4px;
        background: #e9ecef;
        border-radius: 2px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .similarity-fill {
        height: 100%;
        border-radius: 2px;
        background: linear-gradient(90deg, #339af0, #5c7cfa);
    }

    /* ===== CHAT BUBBLE ===== */
    .chat-bubble {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        font-size: 0.95rem;
        color: #343a40;
        line-height: 1.6;
    }

    /* ===== SECTION HEADER ===== */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a3e;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }

    /* ===== FILTER BAR ===== */
    .filter-bar {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
        padding: 0.75rem 1rem;
        background: #f8f9fa;
        border-radius: 10px;
    }
    .filter-info {
        font-size: 0.85rem;
        color: #868e96;
        font-weight: 500;
    }

    /* ===== METRIC STYLE OVERRIDE ===== */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #1a1a3e !important;
    }

    /* ===== HIDE DEFAULT STREAMLIT ELEMENTS ===== */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stToolbar"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# SETUP
# =========================================================
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

try:
    init_db()
    _db_ok = True
except Exception:
    _db_ok = False

@st.cache_resource
def load_system():
    pipeline = Pipeline()
    agent = ImageAgent(pipeline)
    return agent

agent = load_system()


# =========================================================
# NAVIGATION
# =========================================================
st.markdown("""<div style="padding: 0.25rem 0;">
    <div class="sidebar-title">AI Image Detector</div>
    <div class="sidebar-subtitle">Deepfake Detection System</div>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

page = st.sidebar.radio(
    "Menu",
    ["Detector", "Lich su kiem tra", "Dashboard", "Tim anh tuong tu"],
    label_visibility="collapsed"
)

st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

if _db_ok:
    st.sidebar.markdown(
        '<div class="sidebar-status online">Database Connected</div>',
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        '<div class="sidebar-status offline">Database Offline</div>',
        unsafe_allow_html=True
    )


# =========================================================
# HELPERS
# =========================================================
def render_db_image(record, width='stretch'):
    source = get_image_display(record)
    if source is None:
        st.warning("Anh khong ton tai")
        return
    st.image(source, width=width)

def badge(label_type, text):
    return f'<span class="badge badge-{label_type}">{text}</span>'

def result_card(label_text, value):
    return f'''<div class="result-item">
        <div class="result-label">{label_text}</div>
        <div class="result-value">{value}</div>
    </div>'''

def img_to_base64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


# =========================================================
# PAGE 1: DETECTOR
# =========================================================
def detector_page():
    st.markdown('<div class="page-title">Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Upload anh de phat hien fake/real, xem Grad-CAM va chat voi AI</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Chon anh", type=["jpg", "png", "jpeg"])

    if not uploaded:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 0; color: #868e96;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">&#128247;</div>
            <div style="font-size: 1rem;">Chon anh de bat dau phat hien</div>
        </div>""", unsafe_allow_html=True)
        return

    # Reset khi upload ảnh mới
    upload_hash = uploaded.file_id
    if st.session_state.get("upload_hash") != upload_hash:
        st.session_state["upload_hash"] = upload_hash
        st.session_state["res"] = None
        st.session_state["image_path"] = None

    # Lưu ảnh vào uploads/
    if st.session_state["image_path"] is None:
        ext = os.path.splitext(uploaded.name)[1]
        filename = f"{uuid.uuid4().hex}{ext}"
        perm_path = os.path.join(UPLOADS_DIR, filename)

        with open(perm_path, 'wb') as f:
            f.write(uploaded.read())
        with open(perm_path, 'rb') as f:
            image_bytes = f.read()

        st.session_state["image_path"] = perm_path
        st.session_state["image_bytes"] = image_bytes
        st.session_state["filename"] = filename

    image_path = st.session_state["image_path"]

    # Predict (chỉ 1 lần)
    if st.session_state["res"] is None:
        with st.spinner("Dang phan tich..."):
            st.session_state["res"] = agent.predict(
                image_path,
                image_bytes=st.session_state.get("image_bytes"),
                filename=st.session_state.get("filename")
            )

    res = st.session_state["res"]
    r = res["raw"]

    # Ảnh + GradCAM
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Anh goc</div>', unsafe_allow_html=True)
        st.image(image_path, width='stretch')

    with col2:
        st.markdown('<div class="section-header">Grad-CAM</div>', unsafe_allow_html=True)
        cam = r["cam"]
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            heatmap = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
            st.image(overlay, width='stretch')
        else:
            st.warning("Khong doc duoc anh")

    # Kết quả
    label_cls = "fake" if r["label"] == "fake" else "real"
    risk_cls = r["risk"].lower()

    st.markdown(f"""
    <div class="result-grid">
        {result_card("Ket qua", f'{badge(label_cls, r["label"].upper())}')}
        {result_card("Confidence", f'{r["confidence"]:.2%}')}
        {result_card("Risk Level", f'{badge(risk_cls, r["risk"])}')}
    </div>""", unsafe_allow_html=True)

    # Chat
    st.markdown('<div class="section-header">Chat voi AI</div>', unsafe_allow_html=True)
    user_input = st.text_input("Nhap cau hoi...")

    if user_input:
        response = agent.chat(user_input, res)
        st.markdown(f'<div class="chat-bubble">{response["text"]}</div>', unsafe_allow_html=True)

        if "giống" in user_input.lower():
            paths = r.get("similar_paths", [])
            lbls = r.get("similar_labels", [])
            if paths:
                st.markdown('<div class="section-header">Anh tuong tu (tu dataset)</div>', unsafe_allow_html=True)
                cols = st.columns(5)
                for i in range(min(5, len(paths))):
                    with cols[i]:
                        if os.path.exists(paths[i]):
                            st.image(paths[i], width='stretch')
                        if i < len(lbls):
                            lbl_cls = "fake" if lbls[i] == "fake" else "real"
                            st.markdown(badge(lbl_cls, lbls[i].upper()), unsafe_allow_html=True)


# =========================================================
# PAGE 2: LỊCH SỬ KIỂM TRA
# =========================================================
def history_page():
    st.markdown('<div class="page-title">Lich su kiem tra</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Xem lai nhung anh da tung upload va ket qua du doan</div>', unsafe_allow_html=True)

    if not _db_ok:
        st.error("Database khong kha dung. Kiem tra PostgreSQL dang chay.")
        return

    total = get_total_count()

    # Filters
    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        filter_label = st.selectbox("Loc theo label", ["Tat ca", "fake", "real"])
    with col_f2:
        per_page = st.selectbox("Anh / trang", [10, 20, 50])

    page_num = st.number_input("Trang", min_value=1, max_value=max(1, (total // per_page) + 1), value=1)
    offset = (page_num - 1) * per_page

    history = get_history(limit=per_page, offset=offset)
    if filter_label != "Tat ca":
        history = [h for h in history if h['label'] == filter_label]

    st.markdown(f'<div class="filter-bar"><span class="filter-info">Tong: {total} anh &nbsp;|&nbsp; Hien thi: {len(history)}</span></div>', unsafe_allow_html=True)

    if not history:
        st.info("Chua co lich su kiem tra nao.")
        return

    cols = st.columns(3)

    for i, record in enumerate(history):
        with cols[i % 3]:
            label_cls = "fake" if record['label'] == 'fake' else "real"
            risk_cls = record['risk'].lower()
            fname = record.get('filename') or os.path.basename(record.get('path', 'unknown'))

            st.markdown(f'''
            <div class="history-card">
                <div class="history-meta" style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        {badge(label_cls, record["label"].upper())}
                        {badge(risk_cls, record["risk"])}
                    </div>
                    <span>{record["confidence"]:.1%}</span>
                </div>
            </div>''', unsafe_allow_html=True)

            render_db_image(record)

            st.markdown(f'''<div class="history-meta">
                {record["created_at"]} &nbsp;|&nbsp; {fname}
            </div>''', unsafe_allow_html=True)


# =========================================================
# PAGE 3: DASHBOARD
# =========================================================
def dashboard_page():
    st.markdown('<div class="page-title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Thong ke tong quan ve hoat dong detect</div>', unsafe_allow_html=True)

    if not _db_ok:
        st.error("Database khong kha dung.")
        return

    stats = get_stats()
    label_stats = stats['label_stats']
    risk_stats = stats['risk_stats']
    conf_stats = stats['conf_stats']

    total = sum(v['count'] for v in label_stats.values())
    fake_count = label_stats.get('fake', {}).get('count', 0)
    real_count = label_stats.get('real', {}).get('count', 0)

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tong anh", total)
    c2.metric("Fake", fake_count, f"{fake_count/max(total,1):.1%}")
    c3.metric("Real", real_count, f"{real_count/max(total,1):.1%}")
    c4.metric("Avg Confidence", f"{conf_stats['avg']:.2%}")

    st.markdown("---")

    # Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<div class="section-header">Real vs Fake</div>', unsafe_allow_html=True)
        if total > 0:
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.patch.set_facecolor('white')
            wedges, texts, autotexts = ax.pie(
                [fake_count, real_count],
                labels=['Fake', 'Real'],
                colors=['#ff6b6b', '#51cf66'],
                explode=(0.03, 0.03),
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 11, 'fontweight': 500}
            )
            for t in autotexts:
                t.set_fontweight('bold')
            ax.set_title('Phan bo Real / Fake', fontsize=13, fontweight='600', pad=15)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Chua co du lieu.")

    with chart_col2:
        st.markdown('<div class="section-header">Risk Level</div>', unsafe_allow_html=True)
        risk_order = ['SAFE', 'LOW', 'MEDIUM', 'HIGH']
        risk_colors = {'SAFE': '#51cf66', 'LOW': '#ffd43b', 'MEDIUM': '#ff922b', 'HIGH': '#ff6b6b'}
        risk_counts = [risk_stats.get(r, 0) for r in risk_order]
        bar_colors = [risk_colors[r] for r in risk_order]

        if any(risk_counts):
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.patch.set_facecolor('white')
            bars = ax.bar(risk_order, risk_counts, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.6)
            for bar, count in zip(bars, risk_counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                           str(count), ha='center', fontweight='bold', fontsize=11)
            ax.set_title('Phan bo Risk Level', fontsize=13, fontweight='600', pad=15)
            ax.set_ylabel('So anh', fontsize=11)
            ax.set_ylim(0, max(risk_counts) + 2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Chua co du lieu.")

    st.markdown("---")

    # Histogram
    st.markdown('<div class="section-header">Phan bo Confidence</div>', unsafe_allow_html=True)
    conf_data = get_confidence_data()
    fake_conf = conf_data.get('fake', [])
    real_conf = conf_data.get('real', [])

    if fake_conf or real_conf:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('white')
        bins = np.linspace(0, 1, 21)
        if fake_conf:
            ax.hist(fake_conf, bins=bins, alpha=0.7, color='#ff6b6b',
                   label=f'Fake (n={len(fake_conf)})', edgecolor='white')
        if real_conf:
            ax.hist(real_conf, bins=bins, alpha=0.7, color='#51cf66',
                   label=f'Real (n={len(real_conf)})', edgecolor='white')
        ax.set_xlabel('Confidence', fontsize=11)
        ax.set_ylabel('So anh', fontsize=11)
        ax.set_title('Confidence Distribution by Label', fontsize=13, fontweight='600', pad=15)
        ax.legend(loc='best', fontsize=10)
        ax.set_xlim(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Chua co du lieu.")

    # Timeline
    st.markdown('<div class="section-header">Hoat dong theo ngay</div>', unsafe_allow_html=True)
    daily = get_daily_counts(days=30)

    if daily:
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('white')
        dates = [d['date'] for d in daily]
        fake_daily = [d['fake'] for d in daily]
        real_daily = [d['real'] for d in daily]
        x_pos = range(len(dates))

        ax.bar(x_pos, fake_daily, color='#ff6b6b', label='Fake', alpha=0.8)
        ax.bar(x_pos, real_daily, bottom=fake_daily, color='#51cf66', label='Real', alpha=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(dates, rotation=45, ha='right', fontsize=9)
        ax.set_title('Predictions per Day (30 days)', fontsize=13, fontweight='600', pad=15)
        ax.set_ylabel('So anh', fontsize=11)
        ax.legend(loc='best', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Chua co du lieu theo ngay.")

    # Detail table
    st.markdown("---")
    st.markdown('<div class="section-header">Chi tiet theo Label</div>', unsafe_allow_html=True)

    if label_stats:
        detail_data = []
        for label_name in ['fake', 'real']:
            if label_name in label_stats:
                s = label_stats[label_name]
                detail_data.append({
                    'Label': label_name.upper(),
                    'So anh': s['count'],
                    'Ty le': f"{s['count']/max(total,1):.1%}",
                    'Avg Confidence': f"{s['avg_conf']:.2%}",
                })
        st.table(detail_data)


# =========================================================
# PAGE 4: TIM ANH TUONG TU
# =========================================================
def similar_search_page():
    st.markdown('<div class="page-title">Tim anh tuong tu</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Upload anh de tim kiem trong database nhung anh co dac trung tuong tu</div>', unsafe_allow_html=True)

    if not _db_ok:
        st.error("Database khong kha dung.")
        return

    query_file = st.file_uploader("Upload anh tim kiem", type=["jpg", "png", "jpeg"], key="similar_upload")

    if not query_file:
        return

    col_q, col_r = st.columns([1, 2])
    with col_q:
        st.markdown('<div class="section-header">Anh tim kiem</div>', unsafe_allow_html=True)
        st.image(query_file, width='stretch')

    # Temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(query_file.read())
        tmp_path = tmp.name

    with st.spinner("Dang trich xuat dac trung..."):
        query_emb = agent.pipeline.get_embedding(tmp_path)

    if query_emb is None:
        st.error("Khong the trich xuat embedding.")
        os.unlink(tmp_path)
        return

    k = st.slider("So ket qua", min_value=1, max_value=20, value=5)

    with st.spinner("Dang tim kiem..."):
        results = search_similar(query_emb, k=k)

    os.unlink(tmp_path)

    with col_r:
        if not results:
            st.info("Database chua co anh nao. Upload anh qua trang Detector truoc.")
        else:
            st.markdown(f'<div class="section-header">Tim thay {len(results)} anh tuong tu</div>', unsafe_allow_html=True)
            cols = st.columns(min(5, len(results)))

            for i, rec in enumerate(results):
                with cols[i % len(cols)]:
                    label_cls = "fake" if rec['label'] == 'fake' else "real"
                    sim_pct = rec['similarity'] * 100
                    fname = rec.get('filename') or 'unknown'

                    st.markdown(f'''
                    <div style="background:#fff; border:1px solid #e9ecef; border-radius:12px; overflow:hidden;">
                        <div style="padding:0.5rem 0.75rem; display:flex; justify-content:space-between; align-items:center;">
                            {badge(label_cls, rec["label"].upper())}
                            <span style="font-size:0.8rem; font-weight:600; color:#339af0;">{sim_pct:.1f}%</span>
                        </div>
                    </div>''', unsafe_allow_html=True)

                    render_db_image(rec, width=200)

                    st.markdown(f'''<div class="similarity-bar">
                        <div class="similarity-fill" style="width:{min(sim_pct, 100)}%"></div>
                    </div>
                    <div style="font-size:0.78rem; color:#868e96; padding: 0 0.25rem;">
                        {fname} &nbsp;|&nbsp; Conf: {rec["confidence"]:.1%}
                    </div>''', unsafe_allow_html=True)


# =========================================================
# ROUTER
# =========================================================
if page == "Detector":
    detector_page()
elif page == "Lich su kiem tra":
    history_page()
elif page == "Dashboard":
    dashboard_page()
elif page == "Tim anh tuong tu":
    similar_search_page()
