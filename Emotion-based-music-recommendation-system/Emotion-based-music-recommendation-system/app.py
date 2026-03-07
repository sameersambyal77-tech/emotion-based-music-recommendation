# Importing modules
import numpy as np
import streamlit as st
st.set_page_config(page_title="Moodify | Emotion Music", page_icon="🎵", layout="centered")
if 'favorites' not in st.session_state:
    st.session_state['favorites'] = []
if 'mood_history' not in st.session_state:
    st.session_state['mood_history'] = []
import cv2
import pandas as pd

from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import base64
from pathlib import Path

from tensorflow.keras.layers import MaxPooling2D
import base64
from pathlib import Path
st.set_page_config(page_title="Moodify | Emotion Music", page_icon="🎵", layout="centered")
import time
from urllib.parse import quote_plus
import os
import secrets
import hashlib
import requests
import json

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "muse_v3.csv")

df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']

df = df[['name','emotional','pleasant','link','artist']]
def to_open_web_link(url, name, artist):
    try:
        if isinstance(url, str):
            if 'api.spotify.com/v1/tracks/' in url:
                tid = url.rstrip('/').split('/')[-1].split('?')[0]
                return f'https://open.spotify.com/track/{tid}'
            if 'open.spotify.com/track/' in url:
                return url
            if 'lastfm' in url or 'last.fm' in url:
                return url
        query = quote_plus(f"{name} {artist}")
        return f'https://open.spotify.com/search/{query}'
    except Exception:
        query = quote_plus(f"{name} {artist}")
        return f'https://open.spotify.com/search/{query}'

df['web_link'] = df.apply(lambda r: to_open_web_link(r['link'], r['name'], r['artist']), axis=1)

def to_youtube_search(name, artist):
    query = quote_plus(f"{name} {artist}")
    return f'https://www.youtube.com/results?search_query={query}'
print(df)

df = df.sort_values(by=["emotional", "pleasant"])
df.reset_index()
print(df)

df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]

def fun(list):

    data = pd.DataFrame()

    if len(list) == 1:
        v = list[0]
        t = 30
        if v == 'Neutral':
            data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
        elif v == 'Angry':
             data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
        elif v == 'fear':
            data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
        elif v == 'happy':
            data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
        else:
            data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)

    elif len(list) == 2:
        times = [30,20]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':    
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':              
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':             
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:              
               data = pd.concat([df_sad.sample(n=t)])

    elif len(list) == 3:
        times = [55,20,15]
        for i in range(len(list)): 
            v = list[i]          
            t = times[i]

            if v == 'Neutral':              
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':               
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':             
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':               
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:      
                data = pd.concat([df_sad.sample(n=t)])


    elif len(list) == 4:
        times = [30,29,18,9]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral': 
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':              
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':              
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':               
                data =pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:              
               data = pd.concat([df_sad.sample(n=t)])
    else:
        times = [10,7,6,5,2]
        for i in range(len(list)):           
            v = list[i]         
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':           
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':           
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':          
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:
                data = pd.concat([df_sad.sample(n=t)])

    print("data of list func... :",data)
    return data

def pre(l):

    emotion_counts = Counter(l)
    result = []
    for emotion, count in emotion_counts.items():
        result.extend([emotion] * count)
    print("Processed Emotions:", result)

    # result = [item for items, c in Counter(l).most_common()
    #           for item in [items] * c]

    ul = []
    for x in result:
        if x not in ul:
            ul.append(x)
            print(result)
    print("Return the list of unique emotions in the order of occurrence frequency :",ul)
    return ul
    




model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))


model.load_weights(str(BASE_DIR / 'model.h5'))

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


cv2.ocl.setUseOpenCL(False)

print("Loading Haarcascade Classifier...")
face = cv2.CascadeClassifier(str(BASE_DIR / 'haarcascade_frontalface_default.xml'))
if face.empty():
    print("Haarcascade Classifier failed to load.")
else:
    print("Haarcascade Classifier loaded successfully.")

custom_css = '''
<style>
:root {
  --bg1: #0f0f17;
  --bg2: #1b1b2a;
  --card: rgba(255,255,255,0.06);
  --card-border: rgba(255,255,255,0.12);
  --text: #e6e6f0;
  --muted: #a7b0c0;
  --brand: #7c5cff;
  --brand-2: #22c1c3;
}
html, body, .stApp {
  background: linear-gradient(135deg, var(--bg1) 0%, var(--bg2) 100%) fixed !important;
}
.app-header {
  text-align: center;
  margin: 8px 0 18px 0;
}
.app-title {
  font-weight: 800;
  font-size: 36px;
  letter-spacing: 0.4px;
  color: var(--text);
}
.app-subtitle {
  color: var(--muted);
  font-size: 14px;
  margin-top: -6px;
}
.glass {
  background: var(--card);
  border: 1px solid var(--card-border);
  border-radius: 16px;
  padding: 18px 16px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}
.song-card {
  display: flex; gap: 14px; align-items: center;
  background: var(--card);
  border: 1px solid var(--card-border);
  padding: 14px 16px; border-radius: 14px;
}
.song-num {
  width: 34px; height: 34px; border-radius: 50%;
  display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg, var(--brand) 0%, var(--brand-2) 100%);
  color: #fff; font-weight: 700;
}
.song-main { flex: 1; }
.song-title { color: var(--text); font-weight: 700; margin: 0; }
.song-artist { color: var(--muted); margin: 2px 0 0 0; font-size: 13px; }
.song-actions a { color: #9fd3ff; margin-left: 10px; font-size: 13px; }

/* Buttons */
.stButton>button {
  background: linear-gradient(135deg, var(--brand) 0%, var(--brand-2) 100%);
  color: #fff; border: 0; border-radius: 10px; padding: 8px 14px; font-weight: 700;
}
.stButton>button:hover { filter: brightness(1.05); }

/* Progress bar */
.stProgress > div > div > div > div { background: linear-gradient(135deg, var(--brand) 0%, var(--brand-2) 100%); }

/* Slider color */
input[type=range]::-webkit-slider-thumb { background: var(--brand); }
</style>
'''
st.markdown(custom_css, unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
  <div class="app-title">Emotion based music recommendation</div>
  <div class="app-subtitle">Click on a recommended song to open it</div>
  
</div>
""", unsafe_allow_html=True)

col1,col2,col3 = st.columns([1,2,1])

list = []
with col1:
    pass
with col2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    analysis_seconds = st.slider('Analysis time (seconds)', min_value=2, max_value=10, value=5)
    if st.button('Scan emotion'):

        count = 0
        list.clear()

        # Initialize camera just-in-time with DirectShow backend (reduces latency on Windows)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # Prefer 640x480 @ 30fps for stability; adjust as supported
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            st.error("Unable to access the camera. Please check permissions or device.")
            st.stop()

        # Short warmup to allow auto-exposure/white-balance to settle
        warmup_start = time.time()
        while time.time() - warmup_start < 0.5:
            cap.read()
            time.sleep(0.01)

        # Countdown so user can prepare
        status_placeholder = st.empty()
        for sec in range(3, 0, -1):
            status_placeholder.info(f"Starting in {sec}s ...")
            time.sleep(1)
        status_placeholder.empty()

        frame_placeholder = st.empty()
        progress_placeholder = st.progress(0)
        start_time = time.time()
        duration = float(analysis_seconds)
        prediction_stride = 3  # run model prediction every N frames to keep UI smooth
        last_label = None
        last_confidence = 0.0
        while True:

            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Improve detection robustness
            gray = cv2.equalizeHist(gray)
            h, w = gray.shape[:2]
            min_face = (max(40, int(w * 0.12)), max(40, int(h * 0.12)))
            faces = face.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                flags=cv2.CASCADE_SCALE_IMAGE,
                minSize=min_face,
            )
            count = count + 1

            # If multiple faces, pick the largest
            target_face = None
            if len(faces) > 0:
                target_face = max(faces, key=lambda b: b[2] * b[3])
                (x, y, w, h) = target_face
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                # Throttle predictions
                if count % prediction_stride == 0:
                    roi_gray = gray[y:y + h, x:x + w]
                    face48 = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                    face48 = face48.astype('float32') / 255.0
                    cropped_img = np.expand_dims(np.expand_dims(face48, -1), 0)
                    prediction = model.predict(cropped_img, verbose=0)[0]
                    max_index = int(np.argmax(prediction))
                    confidence = float(np.max(prediction))
                    # Only accept confident predictions
                    if confidence >= 0.5:
                        last_label = emotion_dict[max_index]
                        last_confidence = confidence
                        list.append(last_label)
                if last_label is not None:
                    label_text = last_label if last_confidence >= 0.5 else f"{last_label}*"
                    cv2.putText(frame, label_text, (x + 20, y - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Always update preview even when no faces are detected
            frame_placeholder.image(
                cv2.resize(frame, (1000, 700), interpolation=cv2.INTER_CUBIC),
                channels="BGR",
            )

            elapsed = time.time() - start_time
            if duration > 0:
                progress = min(1.0, elapsed / duration)
                progress_placeholder.progress(int(progress * 100))
            if elapsed >= duration:
                break
            time.sleep(0.01)
        cap.release()
        progress_placeholder.empty()

        list = pre(list)
        if len(list) == 0:
            st.warning("No face/emotion detected. Try better lighting, move closer, or increase analysis time.")
        else:
            st.success("Emotions successfully detected") 
    st.markdown('</div>', unsafe_allow_html=True)
        

with col3:
    pass

# Normalize detected emotions and merge with manual overrides
normalize_map = {
    'Happy': 'happy',
    'Fearful': 'fear',
    'Sad': 'sad',
    'Neutral': 'Neutral',
    'Angry': 'Angry',
}
normalized = []
try:
    for e in list:
        m = normalize_map.get(e)
        if m:
            normalized.append(m)
except Exception:
    pass

manual_norm = []
try:
    for e in manual_moods:
        m = normalize_map.get(e)
        if m and m not in manual_norm:
            manual_norm.append(m)
except Exception:
    pass

merged_emotions = manual_norm if len(manual_norm) > 0 else normalized
new_df = fun(merged_emotions)
st.write("")

# 1. Mood History Logging
if len(merged_emotions) > 0 and not new_df.empty:
    top_row = new_df.iloc[0]
    st.session_state['mood_history'].append({
        'timestamp': int(time.time()),
        'moods': ', '.join(merged_emotions),
        'top_song': top_row['name'],
        'top_artist': top_row['artist']
    })

# 3. Mood/Energy Stats (valence = 'pleasant', arousal = 'arousal_tags')
try:
    pleasant = new_df['pleasant'].astype(float).mean()
    if 'arousal_tags' in new_df.columns:
        energy = new_df['arousal_tags'].astype(float).mean()
    else:
        energy = None
except Exception:
    pleasant = None
    energy = None
if pleasant is not None:
    msg = f"**Batch Mood/Energy Stats:** Pleasantness = {pleasant:.2f}"
    if energy is not None:
        msg += f", Energy = {energy:.2f}"
    st.markdown(msg)

# 2. Favorites management
fav_ids = {(f['name'], f['artist']) for f in st.session_state['favorites']}

def add_to_favorites(name: str, artist: str, web_link: str, youtube: str):
    if 'favorites' not in st.session_state:
        st.session_state['favorites'] = []
    # Prevent duplicates by (name, artist)
    for f in st.session_state['favorites']:
        if f.get('name') == name and f.get('artist') == artist:
            return
    st.session_state['favorites'].append({
        'name': name,
        'artist': artist,
        'web_link': web_link,
        'youtube': youtube,
    })

# Render recommendations
st.markdown("<h5 style='text-align: center; color: #a7b0c0;'><b>Recommended songs with artist names</b></h5>", unsafe_allow_html=True)
st.write("-"*117)
if not new_df.empty:
    for l, a, n, i in zip(new_df["web_link"], new_df['artist'], new_df['name'], range(len(new_df))):
        song_id = (n, a)
        final_link = l
        yt = to_youtube_search(n, a)
        cols = st.columns([0.95,0.05])
        with cols[0]:
            st.markdown(
                f"""
                <div class=\"song-card\">\n              <div class=\"song-num\">{i+1}</div>\n              <div class=\"song-main\">\n                <p class=\"song-title\"><a href=\"{final_link}\" target=\"_blank\">{n}</a></p>\n                <p class=\"song-artist\">{a}</p>\n              </div>\n              <div class=\"song-actions\">\n                <a href=\"{final_link}\" target=\"_blank\">Spotify</a>\n                <a href=\"{yt}\" target=\"_blank\">YouTube</a>\n              </div>\n            </div>\n                """,
                unsafe_allow_html=True,
            )
        with cols[1]:
            key = f"favbtn_{hash((n, a))}_{i}"
            if song_id in fav_ids:
                st.button("★", key=key, disabled=True, help="Already in Favorites")
            else:
                st.button(
                    "☆",
                    key=key,
                    help="Add to Favorites",
                    on_click=add_to_favorites,
                    args=(n, a, l, yt),
                )
else:
    st.info("No songs to recommend based on this mood!")

# Mood Journal (history & chart)
with st.expander("Mood Journal"):
    hist_df = pd.DataFrame(st.session_state['mood_history'])
    if not hist_df.empty:
        hist_df['datetime'] = pd.to_datetime(hist_df['timestamp'], unit='s')
        st.dataframe(hist_df[['datetime','moods','top_song','top_artist']], hide_index=True)
        # Show simple bar chart of mood counts
        mood_counts = hist_df['moods'].str.split(', ').explode().value_counts()
        st.bar_chart(mood_counts)
    else:
        st.info("No scans logged yet.")

# Favorites section
with st.expander("My Favorites"):
    fav_df = pd.DataFrame(st.session_state['favorites'])
    if not fav_df.empty:
        for idx, row in fav_df.iterrows():
            st.markdown(f"- <b>{row['name']}</b> by <i>{row['artist']}</i> [Spotify]({row['web_link']}) | [YouTube]({row['youtube']})", unsafe_allow_html=True)
        # CSV export
        fav_csv = fav_df[['name','artist','web_link','youtube']].to_csv(index=False).encode('utf-8')
        st.download_button('Export Favorites as CSV', fav_csv, file_name='moodify_favorites.csv', mime='text/csv')
    else:
        st.info("No favorites yet! Click ☆ to save songs as favorites.")

st.write("---------------------------------------------------------------------------------------------------------------------")

try:
    rendered_df = new_df.head(num_songs)

    try:
        st.session_state['history'].append({
            'timestamp': int(time.time()),
            'emotions': merged_emotions,
            'count': int(len(rendered_df)),
        })
    except Exception:
        pass

    for l,a,n,i in zip(rendered_df["web_link"],rendered_df['artist'],rendered_df['name'],range(len(rendered_df))):

        final_spotify_link = l
        yt = to_youtube_search(n, a)
        preview_url = None
        # Get preview
        try:
            preview_url = get_spotify_audio_preview(n, a, st.session_state.get('spotify_access_token'))
        except Exception:
            preview_url = None
        st.markdown(
            f"""
            <div class="song-card">
              <div class="song-num">{i+1}</div>
              <div class="song-main">
                <p class="song-title"><a href="{final_spotify_link}">{n}</a></p>
                <p class="song-artist">{a}</p>
              </div>
              <div class="song-actions">
                <a href="{final_spotify_link}">Spotify</a>
                <a href="{yt}">YouTube</a>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Optional audio preview (30s)
        if preview_url:
            st.markdown(
                f'<audio src="{preview_url}" controls style="width:250px; margin-top: 6px;"></audio>',
                unsafe_allow_html=True,
            )

    try:
        csv_bytes = rendered_df[['name','artist','web_link']].to_csv(index=False).encode('utf-8')
        st.download_button('Download CSV', data=csv_bytes, file_name='moodify_recommendations.csv', mime='text/csv')
    except Exception:
        pass


    with st.expander('History'):
        try:
            for h in reversed(st.session_state['history'][-10:]):
                ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(h['timestamp']))
                st.write(f"{ts} — {h['count']} tracks — emotions: {', '.join(h['emotions']) if h['emotions'] else 'N/A'}")
        except Exception:
            pass
except Exception:
    pass