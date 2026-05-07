import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import catboost as cb
from sklearn.base import BaseEstimator, ClassifierMixin

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="HeartCare AI - Dự Đoán Tim Mạch",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TÙY CHỈNH ---
# Đã xóa các màu nền fix cứng (white, #f8f9fa) để Streamlit tự đổi màu theo Dark/Light Mode
st.markdown("""
<style>
    .stMetric [data-testid="stMetricValue"] { font-size: 2.5rem; text-align: center; }
    [data-testid="stMetricLabel"] { justify-content: center; font-size: 1.1rem; }
    .risk-high { color: #ff4b4b; font-weight: bold; text-align: center; margin-top: -10px; margin-bottom: 20px;}
    .risk-low { color: #00c853; font-weight: bold; text-align: center; margin-top: -10px; margin-bottom: 20px;}
    html { scroll-behavior: smooth; }
</style>
""", unsafe_allow_html=True)

# --- 1. ĐỊNH NGHĨA CLASS WRAPPER & HÀM FEATURE ---
class CatBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, iterations=200, learning_rate=0.1, depth=6,
                 cat_features=None, random_state=42, verbose=0,
                 border_count=32, one_hot_max_size=10):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.cat_features = cat_features
        self.random_state = random_state
        self.verbose = verbose
        self.border_count = border_count
        self.one_hot_max_size = one_hot_max_size

    def fit(self, X, y):
        try:
            device = 'GPU' if cb.get_gpu_device_count() > 0 else 'CPU'
        except:
            device = 'CPU'
        self.model_ = cb.CatBoostClassifier(
            iterations=self.iterations, learning_rate=self.learning_rate,
            depth=self.depth, cat_features=self.cat_features,
            random_state=self.random_state, verbose=self.verbose,
            task_type=device, border_count=self.border_count,
            one_hot_max_size=self.one_hot_max_size, thread_count=-1
        )
        self.model_.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X): return self.model_.predict(X)
    def predict_proba(self, X): return self.model_.predict_proba(X)

def create_features(df):
    df = df.copy()
    df['age_risk'] = pd.cut(df['Age'], bins=[0, 40, 55, 65, 120], labels=[0, 1, 2, 3]).astype(int)
    df['bp_chol_interaction'] = df['BP'] * df['Cholesterol'] / 10000
    df['hr_reserve_ratio'] = df['Max HR'] / (220 - df['Age']).replace(0, 1)
    df['stress_score'] = df['ST depression'] / (df['hr_reserve_ratio'] + 1e-6)
    df['silent_ischemia_flag'] = ((df['Chest pain type'] == 4) & (df['Exercise angina'] == 1)).astype(int)
    return df

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_models():
    path = "saved_models"
    if not os.path.exists(path): return None, None
    try:
        xgb = joblib.load(os.path.join(path, "xgb_model.pkl"))
        lgb = joblib.load(os.path.join(path, "lgb_model.pkl"))
        cb_model = joblib.load(os.path.join(path, "cb_model.pkl"))
        meta = joblib.load(os.path.join(path, "metadata.pkl"))
        return (xgb, lgb, cb_model), meta
    except Exception as e:
        st.error(f"Lỗi load model: {e}")
        return None, None

models, metadata = load_models()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
    st.title("HeartCare AI")
    st.markdown("---")
    st.info("👨‍⚕️ **Hệ thống hỗ trợ sàng lọc**\nỨng dụng sử dụng AI để đánh giá nguy cơ tim mạch dựa trên 13 chỉ số lâm sàng.")
    
    st.markdown("---")
    st.subheader("📊 Tùy chọn hiển thị")
    show_details = st.checkbox("Hiển thị chi tiết từng mô hình", value=False)
    
    st.markdown("---")
    st.caption("© 2026 HeartCare Project. \nLưu ý: Kết quả chỉ mang tính chất tham khảo.")

# --- 4. GIAO DIỆN CHÍNH ---
if models is None:
    st.error("⚠️ Không tìm thấy Model. Vui lòng chạy script huấn luyện trước.")
else:
    st.header("🩺 Nhập Thông Số Lâm Sàng")
    st.markdown("Vui lòng điền đầy đủ thông tin dưới đây để hệ thống phân tích.")
    
    with st.form("health_form"):
        # Nhóm 1: Chỉ số cơ bản
        st.subheader("1. Chỉ số sinh học cơ bản")
        c1, c2, c3, c4 = st.columns(4)
        with c1: age = st.number_input("Tuổi (Years)", min_value=1, max_value=100, value=50, help="Tuổi của bệnh nhân")
        with c2: sex = st.selectbox("Giới tính", options=[0, 1], format_func=lambda x: "Nam" if x==1 else "Nữ")
        with c3: bp = st.number_input("Huyết áp (mmHg)", min_value=50, max_value=250, value=120)
        with c4: chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        
        st.divider()
        
        # Nhóm 2: Triệu chứng & Tiền sử
        st.subheader("2. Triệu chứng & Tiền sử")
        c5, c6, c7 = st.columns(3)
        with c5: cp = st.selectbox("Loại đau ngực", options=[1, 2, 3, 4], format_func=lambda x: {1:"Đau điển hình", 2:"Không điển hình", 3:"Không đau thắt", 4:"Không triệu chứng"}[x])
        with c6: fbs = st.selectbox("Đường huyết đói > 120?", options=[0, 1], format_func=lambda x: "Có" if x==1 else "Không")
        with c7: exang = st.selectbox("Đau thắt khi vận động?", options=[0, 1], format_func=lambda x: "Có" if x==1 else "Không")
        
        st.divider()
        
        # Nhóm 3: Kết quả kiểm tra chuyên sâu
        st.subheader("3. Kết quả kiểm tra tim mạch")
        c8, c9, c10, c11 = st.columns(4)
        with c8: restecg = st.selectbox("Kết quả EKG nghỉ", options=[0, 1, 2], format_func=lambda x: {0:"Bình thường", 1:"Bất thường ST-T", 2:"Phì đại thất trái"}[x])
        with c9: thalach = st.number_input("Nhịp tim tối đa", min_value=50, max_value=250, value=150)
        with c10: oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=0.0, format="%.1f")
        with c11: slope = st.selectbox("Độ dốc đoạn ST", options=[1, 2, 3], format_func=lambda x: {1:"Dốc lên", 2:"Nằm ngang", 3:"Dốc xuống"}[x])
        
        c12, c13 = st.columns(2)
        with c12: ca = st.selectbox("Số mạch máu chính (Fluoroscopy)", options=[0, 1, 2, 3])
        with c13: thal = st.selectbox("Kiểm tra Thallium", options=[3, 6, 7], format_func=lambda x: {3:"Bình thường", 6:"Khuyết tật cố định", 7:"Khuyết tật đảo ngược"}[x])
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 PHÂN TÍCH NGUY CƠ", use_container_width=True, type="primary")

    # --- 5. XỬ LÝ & HIỂN THỊ KẾT QUẢ ---
    
    # Session state để lưu kết quả và điều khiển cuộn
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    if 'scroll_trigger' not in st.session_state:
        st.session_state.scroll_trigger = 0

    if submitted:
        input_df = pd.DataFrame({
            'Age': [age], 'Sex': [sex], 'Chest pain type': [cp], 'BP': [bp],
            'Cholesterol': [chol], 'FBS over 120': [fbs], 'EKG results': [restecg],
            'Max HR': [thalach], 'Exercise angina': [exang], 'ST depression': [oldpeak],
            'Slope of ST': [slope], 'Number of vessels fluro': [ca], 'Thallium': [thal]
        })
        
        processed_df = create_features(input_df)
        try:
            processed_df = processed_df[metadata['feature_names']]
        except KeyError:
            st.error("Lỗi dữ liệu đầu vào. Vui lòng liên hệ quản trị viên.")
            st.stop()

        p_xgb = models[0].predict_proba(processed_df)[0][1]
        p_lgb = models[1].predict_proba(processed_df)[0][1]
        p_cb = models[2].predict_proba(processed_df)[0][1]
        avg_prob = (p_xgb + p_lgb + p_cb) / 3

        # Lưu kết quả
        st.session_state.last_result = {
            'avg': avg_prob, 'xgb': p_xgb, 'lgb': p_lgb, 'cb': p_cb
        }
        # Tăng biến trigger để buộc JS chạy lại mỗi lần nhấn
        st.session_state.scroll_trigger += 1

    # Hiển thị kết quả nếu đã có
    if st.session_state.last_result:
        res = st.session_state.last_result
        avg_prob = res['avg']
        
        # --- ĐẶT NEO (ANCHOR) ---
        st.markdown("<div id='result-anchor'></div>", unsafe_allow_html=True)
        
        # --- SCRIPT CUỘN TRANG BẮT BUỘC ---
        if st.session_state.scroll_trigger > 0:
            js_code = f"""
            <script>
                // Biến trigger: {st.session_state.scroll_trigger}
                const scrollToResult = () => {{
                    const element = window.parent.document.getElementById('result-anchor');
                    if (element) {{
                        element.scrollIntoView({{behavior: 'smooth', block: 'start'}});
                    }}
                }};
                
                scrollToResult();
                setTimeout(scrollToResult, 300);
                setTimeout(scrollToResult, 800);
            </script>
            """
            st.components.v1.html(js_code, height=0)

        st.markdown("---")
        
        # SỬ DỤNG ST.CONTAINER(BORDER=TRUE) ĐỂ TẠO KHUNG
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            with st.container(border=True):
                st.metric("Tỉ lệ rủi ro", f"{avg_prob:.2%}")
                
                if avg_prob >= 0.5:
                    st.markdown(f"<h3 class='risk-high'>⚠️ NGUY CƠ CAO</h3>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 class='risk-low'>✅ BÌNH THƯỜNG</h3>", unsafe_allow_html=True)
                
                st.progress(min(avg_prob, 1.0), text="Thang đo mức độ:")

        with res_col2:
            with st.container(border=True):
                st.subheader("📋 Phân tích & Khuyến nghị y khoa")
                st.write("") # Tạo khoảng trống nhỏ
                if avg_prob >= 0.7:
                    st.error("**Khẩn cấp:** Nguy cơ tim mạch rất cao. Yêu cầu hẹn lịch khám chuyên khoa tim mạch ngay lập tức. Bác sĩ có thể chỉ định chụp mạch vành hoặc các kiểm tra chuyên sâu khác.")
                elif avg_prob >= 0.5:
                    st.warning("**Cảnh báo:** Các chỉ số đang có dấu hiệu bất thường. Khuyến nghị thay đổi lối sống (giảm muối, giảm chất béo động vật) và lên lịch thăm khám bác sĩ trong tuần tới.")
                elif avg_prob >= 0.3:
                    st.info("**Cần theo dõi:** Nguy cơ ở mức thấp nhưng vẫn cần lưu ý một số chỉ số. Khuyến nghị duy trì kiểm tra sức khỏe định kỳ mỗi 6 tháng.")
                else:
                    st.success("**Tín hiệu tốt:** Trạng thái tim mạch hiện tại đang ổn định. Hãy tiếp tục duy trì chế độ dinh dưỡng lành mạnh và tập thể dục đều đặn.")

        if show_details:
            with st.expander("🔍 Chi tiết độ tin cậy từ các Mô hình (Ensemble AI)"):
                # st.markdown("Hệ thống sử dụng **Ensemble Learning** kết hợp 3 mô hình dự đoán tiên tiến nhất hiện nay để đưa ra kết quả cuối cùng:")
                d1, d2, d3 = st.columns(3)
                d1.metric("XGBoost", f"{res['xgb']:.2%}")
                d2.metric("LightGBM", f"{res['lgb']:.2%}")
                d3.metric("CatBoost", f"{res['cb']:.2%}")