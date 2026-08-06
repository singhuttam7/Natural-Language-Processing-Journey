import streamlit as st
import joblib
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Sentiment Analysis AI",
    page_icon="💬",
    layout="wide"
)

# -------------------- LOAD MODEL --------------------
model = joblib.load("model.pkl")
print(model)

vectorizer = joblib.load("tfidf.pkl")
print(vectorizer)
reverse_mapping = joblib.load("reverse_mapping.pkl")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>

.main{
    background-color:#f6f8fc;
}

.title{
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:white;
    padding:20px;
    border-radius:15px;
    background:linear-gradient(90deg,#4A00E0,#8E2DE2);
}

.subtitle{
    text-align:center;
    font-size:18px;
    color:#555;
    margin-bottom:20px;
}

.result{
    padding:20px;
    border-radius:12px;
    background:#ffffff;
    box-shadow:0px 5px 15px rgba(0,0,0,0.1);
    text-align:center;
    font-size:28px;
    font-weight:bold;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown(
    "<div class='title'>💬 Sentiment Analysis AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Detect human emotions from text using Machine Learning</div>",
    unsafe_allow_html=True
)

# -------------------- INPUT --------------------
text = st.text_area(
    "✍ Enter your text",
    height=180,
    placeholder="Example: I am feeling very happy today..."
)

# -------------------- PREDICT --------------------
if st.button("🚀 Analyze Emotion", use_container_width=True):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        vector = vectorizer.transform([text])

        prediction = model.predict(vector)[0]

        emotion = reverse_mapping[prediction]

        probabilities = model.predict_proba(vector)
        confidence = np.max(probabilities) * 100

        emojis = {
            "joy":"😄",
            "sadness":"😢",
            "anger":"😡",
            "fear":"😨",
            "love":"❤️",
            "surprise":"😲"
        }

        emoji = emojis.get(emotion,"🙂")

        st.markdown(
            f"""
            <div class='result'>
                {emoji}<br><br>
                Detected Emotion<br><br>
                <span style='color:#8E2DE2'>{emotion.upper()}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(confidence/100)

        st.success(f"Confidence : {confidence:.2f}%")

# -------------------- SAMPLE TEXT --------------------
st.divider()

st.subheader("💡 Try these examples")

col1,col2,col3 = st.columns(3)

with col1:
    st.info("I am feeling amazing today.")

with col2:
    st.info("Nobody understands me anymore.")

with col3:
    st.info("I hate everything happening around me.")

# -------------------- FOOTER --------------------
st.markdown(
    """
    <hr>
    <div style="text-align:center; color:gray; font-size:14px;">
        Developed with ❤️ by <b>Uttam Singh</b><br>
        © 2026 Uttam Singh. All Rights Reserved.
    </div>
    """,
    unsafe_allow_html=True
)