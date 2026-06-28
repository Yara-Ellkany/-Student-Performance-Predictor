import streamlit as st
import numpy as np

class SimpleModel:
    """نموذج بسيط يتعلم من البيانات بدون أي مكتبة"""

    def __init__(self):
        self.weights = None   
        self.bias    = 0.0
        self.trained = False

    def _normalize(self, X):
        """تطبيع البيانات بين 0 و 1"""
        return (X - self.x_min) / (self.x_max - self.x_min + 1e-8)

    def train(self, X, y, epochs=200, lr=0.01):
        """
        تدريب النموذج باستخدام Gradient Descent
        X = المدخلات، y = النتائج (0 أو 1)
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        self.x_min = X.min(axis=0)
        self.x_max = X.max(axis=0)
        X = self._normalize(X)

        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias    = 0.0
        self.loss_history = []

        for epoch in range(epochs):
            pred = self._sigmoid(X @ self.weights + self.bias)

            error = pred - y
            loss  = -np.mean(y * np.log(pred + 1e-8) + (1-y) * np.log(1-pred + 1e-8))
            self.loss_history.append(loss)

            # تحديث الأوزان (هنا يتعلم النموذج!)
            self.weights -= lr * (X.T @ error) / len(y)
            self.bias    -= lr * error.mean()

        self.trained = True

    def predict(self, x):
        x = np.array(x, dtype=float)
        x = (x - self.x_min) / (self.x_max - self.x_min + 1e-8)
        prob = self._sigmoid(x @ self.weights + self.bias)
        return prob

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


if "model"    not in st.session_state: st.session_state.model    = SimpleModel()
if "data"     not in st.session_state: st.session_state.data     = []
if "trained"  not in st.session_state: st.session_state.trained  = False

st.title("   Student Performance Predictor")

st.header(" الخطوة ١ — أدخل بيانات الطلاب")
st.caption("كلما أدخلت بيانات أكثر، كلما تعلّم النموذج أفضل!")

with st.form("add_data"):
    c1, c2, c3, c4 = st.columns(4)
    study  = c1.number_input(" ساعات المذاكرة", 0, 10, 3)
    attend = c2.number_input(" أيام الحضور/5", 0, 5, 4)
    sleep  = c3.number_input(" ساعات النوم", 4, 10, 7)
    result = c4.selectbox(" النتيجة الحقيقية", ["نجح ", "رسب "])
    add    = st.form_submit_button(" أضف هذا الطالب", use_container_width=True)

if add:
    label = 1 if "نجح" in result else 0
    st.session_state.data.append([study, attend, sleep, label])
    st.success(f" تم إضافة الطالب! إجمالي البيانات: {len(st.session_state.data)}")

if st.session_state.data:
    st.markdown(f"**البيانات المدخلة حتى الآن: {len(st.session_state.data)} طالب**")
    rows = []
    for i, d in enumerate(st.session_state.data):
        rows.append({
            "#": i+1,
            "ساعات المذاكرة": d[0],
            "أيام الحضور": d[1],
            "ساعات النوم": d[2],
            "النتيجة": "نجح " if d[3]==1 else "رسب "
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    if st.button(" مسح كل البيانات"):
        st.session_state.data    = []
        st.session_state.trained = False
        st.session_state.model   = SimpleModel()
        st.rerun()

st.divider()
st.header(" الخطوة ٢ — درّب النموذج")

needs_more = len(st.session_state.data) < 5
if needs_more:
    st.warning(f" أضيفي على الأقل 5 طلاب للتدريب. لديكِ {len(st.session_state.data)} حتى الآن.")
else:
    if st.button(" ابدأ التدريب!", use_container_width=True, type="primary"):
        data = st.session_state.data
        X    = [[d[0], d[1], d[2]] for d in data]
        y    = [d[3] for d in data]

        with st.spinner(" النموذج يتعلم..."):
            st.session_state.model.train(X, y, epochs=300, lr=0.05)
            st.session_state.trained = True

        st.success(" انتهى التدريب! النموذج تعلّم من بياناتك.")
        st.balloons()

        st.markdown("** منحنى التعلم — كيف انخفض الخطأ مع الوقت:**")
        loss = st.session_state.model.loss_history
        st.line_chart({"الخطأ": loss})
        st.caption("كلما انخفض الخطأ، كلما تعلّم النموذج أكثر ")

        st.markdown("** ماذا تعلّم النموذج؟ (الأوزان)**")
        w = st.session_state.model.weights
        names = ["ساعات المذاكرة", "أيام الحضور", "ساعات النوم"]
        for name, weight in zip(names, w):
            direction = "مهم جداً " if weight > 0.3 else ("مهم " if weight > 0 else "أقل تأثيراً ")
            st.write(f"- **{name}**: {weight:.3f} ← {direction}")

st.divider()
st.header(" الخطوة ٣ — جرّب النموذج!")

if not st.session_state.trained:
    st.info(" درّبي النموذج أولاً في الخطوة ٢")
else:
    st.markdown("الآن أدخلي بيانات طالب جديد والنموذج سيتنبأ بنتيجته:")

    t1, t2, t3 = st.columns(3)
    t_study  = t1.slider(" ساعات المذاكرة", 0, 10, 3)
    t_attend = t2.slider(" أيام الحضور", 0, 5, 3)
    t_sleep  = t3.slider(" ساعات النوم", 4, 10, 7)

    prob = st.session_state.model.predict([t_study, t_attend, t_sleep])

    st.markdown("### نتيجة التنبؤ:")
    st.progress(float(prob))
    st.caption(f"نسبة النجاح المتوقعة: **{prob*100:.1f}%**")

    if prob >= 0.6:
        st.success(f"##  النموذج يتوقع: سينجح! ({prob*100:.0f}%)")
    elif prob >= 0.4:
        st.warning(f"##  النموذج غير متأكد ({prob*100:.0f}%)")
    else:
        st.error(f"##  النموذج يتوقع: قد يرسب ({prob*100:.0f}%)")

    st.info("💡 جرّبي تغيير القيم وشاهدي كيف تتغير النتيجة!")
