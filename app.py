import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="توقّع أدائك الدراسي", page_icon="🎓", layout="centered")

st.markdown("""
<style>
.result-box { padding: 25px; border-radius: 15px; text-align: center; margin-top: 20px; }
.excellent  { background: #e8f5e9; border: 2px solid #4caf50; }
.good       { background: #e3f2fd; border: 2px solid #2196f3; }
.average    { background: #fff8e1; border: 2px solid #ff9800; }
.poor       { background: #fce4ec; border: 2px solid #e91e63; }
.big-emoji  { font-size: 3rem; }
.result-title { font-size: 1.8rem; font-weight: bold; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 1000
    data = {
        "study_hours":     np.random.randint(0, 10, n),
        "attendance":      np.random.randint(50, 100, n),
        "sleep_hours":     np.random.randint(4, 10, n),
        "prev_grade":      np.random.randint(40, 100, n),
        "extracurricular": np.random.randint(0, 2, n),
        "parent_support":  np.random.randint(1, 6, n),
        "screen_time":     np.random.randint(1, 8, n),
    }
    df = pd.DataFrame(data)
    score = (
        df["study_hours"]     * 4.0
        + df["attendance"]    * 0.3
        + df["sleep_hours"]   * 1.5
        + df["prev_grade"]    * 0.4
        + df["extracurricular"] * 3.0
        + df["parent_support"]  * 2.0
        - df["screen_time"]     * 2.0
        + np.random.normal(0, 5, n)
    )
    def to_label(s):
        if s >= 70:   return "ممتاز"
        elif s >= 55: return "جيد"
        elif s >= 40: return "مقبول"
        else:         return "يحتاج تحسين"
    df["performance"] = score.apply(to_label)
    features = ["study_hours","attendance","sleep_hours",
                "prev_grade","extracurricular","parent_support","screen_time"]
    X, y = df[features], df["performance"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, acc, features

model, accuracy, features = train_model()

st.title("🎓 توقّع أدائك الدراسي")
st.markdown("أجب على الأسئلة التالية وسيتوقع الذكاء الاصطناعي مستواك الدراسي.")
st.info(f"🎯 دقة النموذج: **{accuracy * 100:.1f}%**")
st.divider()

col1, col2 = st.columns(2)
with col1:
    study_hours    = st.slider("📚 ساعات المذاكرة يومياً", 0, 10, 3)
    attendance     = st.slider("🏫 نسبة الحضور %", 50, 100, 80)
    sleep_hours    = st.slider("😴 ساعات النوم يومياً", 4, 10, 7)
    screen_time    = st.slider("📱 وقت الشاشة يومياً", 1, 8, 3)
with col2:
    prev_grade      = st.slider("📝 درجتك في الفصل الماضي", 40, 100, 70)
    parent_support  = st.slider("👨‍👩‍👦 دعم الأهل للمذاكرة", 1, 5, 3)
    extracurricular = st.radio("⚽ هل تمارس أنشطة خارجية؟",
                               ["نعم ✅", "لا ❌"]) == "نعم ✅"

st.divider()

if st.button("🔮 توقّع أدائي الآن!", use_container_width=True, type="primary"):
    X_input = np.array([[study_hours, attendance, sleep_hours,
                         prev_grade, int(extracurricular),
                         parent_support, screen_time]])
    prediction = model.predict(X_input)[0]
    confidence = max(model.predict_proba(X_input)[0]) * 100

    config = {
        "ممتاز":        {"emoji":"🏆","color":"excellent",
                         "msg":"أداء رائع! استمر على هذا المستوى.",
                         "tips":["حافظ على روتين المذاكرة","شارك زملاءك بأساليبك","ضع أهدافاً أعلى لنفسك"]},
        "جيد":          {"emoji":"😊","color":"good",
                         "msg":"أداء جيد! قليل من الجهد سيرفعك للقمة.",
                         "tips":["زِد ساعة مذاكرة يومياً","راجع المواد الصعبة أولاً","نظّم وقتك بجدول أسبوعي"]},
        "مقبول":        {"emoji":"💪","color":"average",
                         "msg":"يمكنك تحسين مستواك! لديك القدرة على ذلك.",
                         "tips":["قلّل وقت الشاشة","نَم 8 ساعات يومياً","اطلب المساعدة من معلمك"]},
        "يحتاج تحسين": {"emoji":"🌱","color":"poor",
                         "msg":"لا بأس! كل بطل بدأ من الصفر. ابدأ الآن.",
                         "tips":["ابدأ بـ 30 دقيقة مذاكرة يومياً","أبعد الهاتف أثناء المذاكرة","تحدّث مع أهلك لمساعدتك"]},
    }
    c = config.get(prediction, config["مقبول"])

    st.markdown(f"""
    <div class="result-box {c['color']}">
        <div class="big-emoji">{c['emoji']}</div>
        <div class="result-title">{prediction}</div>
        <p>{c['msg']}</p>
        <p style="color:#888;font-size:0.85rem;">نسبة الثقة: {confidence:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 💡 نصائح لك:")
    for tip in c["tips"]:
        st.success(f"✔ {tip}")

    st.divider()
    st.markdown("### 🔍 ما الذي أثّر في نتيجتك؟")
    labels_ar = {
        "study_hours":"ساعات المذاكرة","attendance":"نسبة الحضور",
        "sleep_hours":"ساعات النوم","prev_grade":"الدرجة السابقة",
        "extracurricular":"الأنشطة الخارجية","parent_support":"دعم الأهل",
        "screen_time":"وقت الشاشة",
    }
    imp_df = pd.DataFrame({
        "العامل":  [labels_ar[f] for f in features],
        "الأهمية": model.feature_importances_,
    }).sort_values("الأهمية", ascending=True)
    st.bar_chart(imp_df.set_index("العامل"))
