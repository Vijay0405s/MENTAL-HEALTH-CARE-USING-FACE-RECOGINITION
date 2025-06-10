import streamlit as st
import streamlit_authenticator as stauth
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import os
import pandas as pd
from datetime import datetime, date
import cv2
from emotion_model import predict_emotion, detect_faces

# ==== LOGIN SETUP ====
names = ['Alice Johnson', 'Bob Smith']
usernames = ['alice', 'bob']
passwords = ['123', 'abc']
hashed_passwords = stauth.Hasher(passwords).generate()

credentials = {
    'usernames': {
        usernames[i]: {'name': names[i], 'password': hashed_passwords[i]}
        for i in range(len(usernames))
    }
}

authenticator = stauth.Authenticate(credentials, 'mental_health', 'abcdef', cookie_expiry_days=1)
name, auth_status, username = authenticator.login('Login', 'main')

# ==== MAIN UI ====
if auth_status:
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.write(f"Welcome, {name} 👋")
    st.title("🧠 Mental Health Emotion Detector")

    # === Webcam Processor ===
    class EmotionProcessor(VideoProcessorBase):
        def __init__(self):
            self.last_emotion = "neutral"

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray)

            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]
                emotion = predict_emotion(face_img)
                self.last_emotion = emotion
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return img

    ctx = webrtc_streamer(
        key="emotion",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # === Save Emotion History ===
    def save_emotion(emotion, username):
        if not os.path.exists("emotion_history"):
            os.makedirs("emotion_history")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = pd.DataFrame([{"timestamp": timestamp, "emotion": emotion}])
        file = f"emotion_history/{username}.csv"
        df.to_csv(file, mode="a", index=False, header=not os.path.exists(file))

    # === Emotion Resources ===
    tips = {
        "happy": "Stay positive! 😊",
        "sad": "Try some deep breathing. 💙",
        "angry": "Pause and reflect. 🧘",
        "fear": "Ground yourself in the present.",
        "neutral": "Maybe take a short walk.",
        "disgust": "Try focusing on something pleasant.",
        "surprise": "Write down what surprised you!"
    }

    videos = {
        "happy": "https://www.youtube.com/watch?v=ZbZSe6N_BXs",
        "sad": "https://www.youtube.com/watch?v=inpok4MKVLM",
        "angry": "https://www.youtube.com/watch?v=SEfs5TJZ6Nk",
        "neutral": "https://www.youtube.com/watch?v=5qap5aO4i9A",
        "fear": "https://www.youtube.com/watch?v=O-6f5wQXSu8",
        "surprise": "https://www.youtube.com/watch?v=wfDTp2GogaQ",
        "disgust": "https://www.youtube.com/watch?v=mgmVOuLgFB0"
    }

    music = {
        "happy": "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC",
        "sad": "https://open.spotify.com/playlist/37i9dQZF1DWVrtsSlLKzro",
        "angry": "https://open.spotify.com/playlist/37i9dQZF1DWU0ScTcjJBdj",
        "neutral": "https://open.spotify.com/playlist/37i9dQZF1DWYBO1MoTDhZI",
        "fear": "https://open.spotify.com/playlist/37i9dQZF1DX3Ogo9pFvBkY",
        "surprise": "https://open.spotify.com/playlist/37i9dQZF1DX1s9knjP51Oa",
        "disgust": "https://open.spotify.com/playlist/37i9dQZF1DX6VdMW310YC7"
    }

    # === PFA Chatbot ===
    def pfa_chatbot_response(user_input, emotion):
        acknowledgements = {
            "happy": "It's great to hear you're feeling happy! 😊",
            "sad": "I'm really sorry you're feeling sad. 💙",
            "angry": "It's okay to feel angry sometimes. 😤",
            "fear": "That sounds really scary. You're not alone. 😨",
            "disgust": "Feeling disgusted is valid. Let's talk about it. 😖",
            "neutral": "You're feeling neutral — that’s perfectly okay. 😐",
            "surprise": "Surprises can shake us up. 😲"
        }

        open_questions = {
            "happy": "What’s been bringing you joy lately?",
            "sad": "Would you like to share what’s been making you feel down?",
            "angry": "What do you think triggered your anger today?",
            "fear": "Is there a specific worry on your mind?",
            "disgust": "Do you want to talk about what's causing that feeling?",
            "neutral": "Is there anything on your mind you’d like to reflect on?",
            "surprise": "What surprised you today?"
        }

        coping_strategies = {
            "happy": "Keep up those positive habits! Maybe journal this moment.",
            "sad": "Try a calming breathing exercise or reach out to a friend.",
            "angry": "Take a pause — maybe go for a short walk or try box breathing.",
            "fear": "Ground yourself: 5 things you see, 4 you can touch, 3 you hear...",
            "disgust": "Try listening to music or doing something you enjoy.",
            "neutral": "A short walk or creative activity might help spark your energy.",
            "surprise": "Jot your thoughts down to process the surprise more clearly."
        }

        ack = acknowledgements.get(emotion, "I'm here with you.")
        ask = open_questions.get(emotion, "Would you like to talk more about it?")
        tip = coping_strategies.get(emotion, "Take a deep breath. You’ve got this.")
        return ack, ask, tip

    # === DISPLAY EMOTION ===
    if ctx and ctx.video_processor:
        emotion = ctx.video_processor.last_emotion
        save_emotion(emotion, username)

        st.markdown(f"### 🎭 Detected Emotion: `{emotion.capitalize()}`")
        st.info(tips.get(emotion, "Be kind to yourself today."))

        st.markdown("### 🎵 Music")
        st.markdown(f"[Listen on Spotify]({music.get(emotion, '#')})")

        st.markdown("### 🎥 Video")
        st.video(videos.get(emotion, ""))

        # === PFA CHATBOT UI ===
        st.markdown("### 💬 Psychological First Aid (PFA) Chatbot")
        user_input = st.text_input("How are you feeling in your own words?")

        if user_input:
            ack, ask, tip = pfa_chatbot_response(user_input, emotion)
            st.markdown(f"**🧠 Step 1 – Acknowledge:** {ack}")
            st.markdown(f"**❓ Step 2 – Ask:** {ask}")
            st.markdown(f"**💡 Step 3 – Suggest:** {tip}")

        # === HISTORY ===
        st.markdown("### 📈 Emotion History")
        file = f"emotion_history/{username}.csv"
        if os.path.exists(file):
            df = pd.read_csv(file)
            st.dataframe(df.tail(10))
        else:
            st.info("No history yet.")

    # === HABIT TRACKER ===
    st.markdown("### ✅ Daily Habit Tracker")
    habits = ["Drink 2L Water 💧", "Walk 15 Minutes 🚶", "Meditate 🧘", "Limit Social Media 📵", "Sleep 7+ Hours 😴"]
    today = date.today().strftime("%Y-%m-%d")
    habit_file = f"habit_data/{username}.csv"

    if not os.path.exists("habit_data"):
        os.makedirs("habit_data")

    if os.path.exists(habit_file):
        habit_df = pd.read_csv(habit_file)
        if today in habit_df['date'].values:
            today_habits = habit_df[habit_df['date'] == today].iloc[0].to_dict()
        else:
            today_habits = {"date": today, **{h: False for h in habits}}
            habit_df = pd.concat([habit_df, pd.DataFrame([today_habits])], ignore_index=True)
    else:
        today_habits = {"date": today, **{h: False for h in habits}}
        habit_df = pd.DataFrame([today_habits])

    updated = False
    for habit in habits:
        checked = st.checkbox(habit, value=bool(today_habits[habit]))
        if checked != bool(today_habits[habit]):
            habit_df.loc[habit_df['date'] == today, habit] = checked
            updated = True

    if updated:
        habit_df.to_csv(habit_file, index=False)
        st.success("✅ Habit progress saved!")

    st.markdown("### 📊 Your Past 7 Days")
    st.dataframe(habit_df.tail(7).set_index("date"))

elif auth_status is False:
    st.error("Incorrect username or password.")
elif auth_status is None:
    st.warning("Please enter your login details.")
