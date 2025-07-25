import os
# ✅ Set safe cache directories (for Hugging Face Spaces Docker)
os.environ['TRANSFORMERS_CACHE'] = '/tmp'
os.environ['HF_HOME'] = '/tmp'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/tmp'

import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Songs data
songs = {
    "happy": {
        "Hindi": [
            ("Gallan Goodiyan – Dil Dhadakne Do", "https://youtu.be/jCEdTq3j-0U?si=Odusdwl7pf2DqqfW"),
            ("Tera Hone Laga Hoon – Ajab Prem Ki Ghazab Kahani", "https://youtu.be/useznoRhrWU?si=EqFuwB5rPX4KmN-b")
        ],
        "Telugu": [
            ("Oh Baby", "https://youtu.be/6PrHsxxeaV0?si=Rwn7ePkYnFVSjuai"),
            ("Hoyna Hoyna", "https://youtu.be/91EzD9VgwGk?si=J8HI7ZE34BabbSng")
        ],
        "English": [
            ("Happy – Pharrell Williams", "https://youtu.be/ZbZSe6N_BXs"),
            ("Best Day of My Life – American Authors", "https://youtu.be/Y66j_BUCBMY")
        ]
    },
    "sad": {
        "Hindi": [
            ("Channa Mereya – Ae Dil Hai Mushkil", "https://youtu.be/284Ov7ysmfA?si=zEyxC9ATo604qCuX"),
            ("Tujhe Kitna Chahne Lage – Kabir Singh", "https://youtu.be/AgX2II9si7w?si=eKgClYX-nNkRnkyU")
        ],
        "Telugu": [
            ("Athey Nanne", "https://youtu.be/KcbmbnRX-nU?si=23aLy6NsB21j0rsn"),
            ("Telisiney Na Nuvvey", "https://youtu.be/KcbmbnRX-nU?si=23aLy6NsB21j0rsn")
        ],
        "English": [
            ("Let Her Go – Passenger", "https://youtu.be/RBumgq5yVrA"),
            ("Someone Like You – Adele", "https://youtu.be/hLQl3WQQoQ0")
        ]
    }
}

# Multilingual UI
UI_TEXTS = {
    "English": {
        "title": "🎧 Swecha Mood Song Recommender",
        "select_language": "🌍 Select language:",
        "describe_mood": "💬 Describe your current mood:",
        "closest_mood": "🎯 Closest mood match:",
        "recommended_songs": "🎵 Recommended songs for"
    },
    "Hindi": {
        "title": "🎧 स्वेचा मूड गीत सुझावक",
        "select_language": "🌍 भाषा चुनें:",
        "describe_mood": "💬 अपना वर्तमान मूड बताएं:",
        "closest_mood": "🎯 निकटतम मूड मेल:",
        "recommended_songs": "🎵 के लिए अनुशंसित गीत"
    },
    "Telugu": {
        "title": "🎧 స్వేచా మనోభావ పాట సిఫార్సులు",
        "select_language": "🌍 భాషను ఎంచుకోండి:",
        "describe_mood": "💬 మీ ప్రస్తుత మనోభావాన్ని వర్ణించండి:",
        "closest_mood": "🎯 సమీపమైన మనోభావం:",
        "recommended_songs": "🎵 కోసం సిఫార్సు చేసిన పాటలు"
    }
}

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

st.set_page_config(page_title="🎧 Swecha Mood Song Recommender", layout="centered", initial_sidebar_state="collapsed")

# UI language selector
ui_lang = st.selectbox("🌍 " + "Select UI language / भाषा चुनें / భాషను ఎంచుకోండి:", ["English", "Hindi", "Telugu"])
texts = UI_TEXTS[ui_lang]

st.markdown(f"<h1 style='text-align:center'>{texts['title']}</h1>", unsafe_allow_html=True)

# Song language selector
song_lang = st.selectbox(texts["select_language"], ["English", "Hindi", "Telugu"])

# Mood input
user_mood = st.text_input(f"{texts['describe_mood']} ({song_lang})")

if user_mood:
    moods_list = list(songs.keys())
    user_emb = model.encode(user_mood, convert_to_tensor=True)
    mood_embs = model.encode(moods_list, convert_to_tensor=True)

    cos_scores = util.cos_sim(user_emb, mood_embs)[0]
    best_idx = cos_scores.argmax()
    best_mood = moods_list[best_idx]
    confidence = cos_scores[best_idx].item()

    st.success(f"{texts['closest_mood']} **{best_mood}** (Confidence: {confidence:.2f})")
    st.markdown(f"### {texts['recommended_songs']} **{best_mood}** {texts['recommended_songs'].split()[-1]} **{song_lang}**:")

    for title, url in songs[best_mood][song_lang]:
        st.markdown(f"- [{title}]({url})")
