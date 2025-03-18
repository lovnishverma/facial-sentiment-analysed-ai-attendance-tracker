import streamlit as st
import cv2
import face_recognition
import os
import sqlite3
import pandas as pd
import numpy as np  
from datetime import datetime
from PIL import Image
from keras.models import load_model

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="Face and Emotion Recognition Attendance System",
    page_icon=":camera:",
    layout="wide"
)

# --- Initialize session_state attributes ---
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

# --- Password Protection (Optional) ---
password = st.text_input("Enter password", type="password")

# Stop if the password is incorrect
if password != "nielit":
    st.stop()

# --- Database Setup ---
conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()

# Create Table with Emotion Column
cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        roll_no TEXT,
        date TEXT,
        time TEXT,
        status TEXT,
        emotion TEXT
    )
''')
conn.commit()

# --- Load Known Faces ---
def load_known_faces():
    images = []
    classnames = []
    directory = "Photos"

    for cls in os.listdir(directory):
        if os.path.splitext(cls)[1] in [".jpg", ".jpeg", ".png"]:
            img_path = os.path.join(directory, cls)
            curImg = cv2.imread(img_path)
            images.append(curImg)
            classnames.append(os.path.splitext(cls)[0])

    return images, classnames

# --- Encode Known Faces ---
def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

Images, classnames = load_known_faces()
encodeListKnown = find_encodings(Images)

# --- Load Emotion Detection Model ---
@st.cache_resource
def load_emotion_model():
    return load_model('CNN_Model_acc_75.h5')

emotion_model = load_emotion_model()
img_shape = 48
emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Preprocess Frame for Emotion Detection ---
def process_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    emotions_detected = []

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        face_roi = cv2.resize(roi_color, (img_shape, img_shape))
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = face_roi / float(img_shape)
        predictions = emotion_model.predict(face_roi)
        emotion = emotion_labels[np.argmax(predictions[0])]
        emotions_detected.append(emotion)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return frame, emotions_detected

# --- Camera Functionality ---
camera_active = st.session_state.get("camera_active", False)

if st.sidebar.button("Start Camera"):
    st.session_state.camera_active = True

if st.sidebar.button("Stop Camera"):
    st.session_state.camera_active = False

# --- Add New Face ---
def add_new_face():
    new_name = st.text_input("Enter your name:")
    roll_no = st.text_input("Enter your roll number:")

    if st.session_state.camera_active:
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None and new_name and roll_no:
            st.session_state.camera_active = False  # Stop camera after photo is taken

            # Check if the user already exists in the database
            cursor.execute("SELECT * FROM attendance WHERE name = ? AND roll_no = ?", (new_name, roll_no))
            existing_user = cursor.fetchone()

            if existing_user:
                st.warning(f"{new_name} ({roll_no}) is already registered.")
            else:
                # Save the photo and update face encodings
                image = np.array(Image.open(img_file_buffer))
                img_path = os.path.join("Photos", f"{new_name}_{roll_no}.jpg")
                cv2.imwrite(img_path, image)

                global Images, classnames, encodeListKnown
                Images, classnames = load_known_faces()
                encodeListKnown = find_encodings(Images)

                st.success(f"New face added for {new_name} ({roll_no}).")
    else:
        st.info("Camera is not active. Start the camera to take a picture.")

# --- Recognize Face and Emotion ---
def recognize_face():
    if st.session_state.camera_active:
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None:
            st.session_state.camera_active = False  # Stop camera after photo is taken
            with st.spinner("Processing..."):
                image = np.array(Image.open(img_file_buffer))
                imgS = cv2.resize(image, (0, 0), None, 0.25, 0.25)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

                facesCurFrame = face_recognition.face_locations(imgS)
                encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

                detected_emotions = []
                if len(encodesCurFrame) > 0:
                    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                        matchIndex = np.argmin(faceDis)

                        if matches[matchIndex]:
                            name = classnames[matchIndex].split("_")[0]
                            roll_no = classnames[matchIndex].split("_")[1]

                            y1, x2, y2, x1 = faceLoc
                            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                            frame, detected_emotions = process_frame(image)
                            date = datetime.now().strftime('%Y-%m-%d')
                            time = datetime.now().strftime('%H:%M:%S')
                            emotion = detected_emotions[0] if detected_emotions else "Unknown"

                            cursor.execute("INSERT INTO attendance (name, roll_no, date, time, status, emotion) VALUES (?, ?, ?, ?, 'Present', ?)", 
                                           (name, roll_no, date, time, emotion))
                            conn.commit()
                            st.success(f"Attendance marked for {name} with emotion: {emotion}.")
                        else:
                            st.warning("Face not recognized.")
                else:
                    st.warning("No face detected.")
                st.image(image, caption="Detected Face and Emotion", use_container_width=True)

    else:
        st.info("Camera is not active. Start the camera to take a picture.")

# --- View Attendance Records ---
def view_attendance_records():
    st.subheader("Attendance Records")
    cursor.execute("SELECT * FROM attendance ORDER BY date DESC, time DESC")
    records = cursor.fetchall()

    if records:
        df = pd.DataFrame(records, columns=["ID", "Name", "Roll No", "Date", "Time", "Status", "Emotion"])
        st.table(df)
    else:
        st.info("No attendance records available.")

# --- Main Logic ---
if __name__ == "__main__":
    st.title("FACIAL SENTIMENT ANALYSIZED ATTENDANCE TRACKER")
    # Larger title
    st.markdown("<h2 style='text-align: center;'>Can Recognise Face and Detect:</h2>", unsafe_allow_html=True)
    # Smaller subtitle
    st.markdown("<h3 style='text-align: center;'>Emotions: angry, fear, happy, neutral, sad, surprise </h3>", unsafe_allow_html=True)

    app_mode = st.sidebar.selectbox("Select Mode", ["Recognize Face & Emotion", "Add New Face", "View Records"])

    if app_mode == "Recognize Face & Emotion":
        recognize_face()
    elif app_mode == "Add New Face":
        add_new_face()
    elif app_mode == "View Records":
        view_attendance_records()

    conn.close()
