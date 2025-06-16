# Face and Emotion Recognition-Based Attendance System

This project is an AI-powered attendance system that leverages face recognition and emotion detection to automate attendance marking. Utilizing Streamlit, OpenCV, Face Recognition, and Deep Learning (CNN), the system captures real-time video input from a webcam, identifies registered faces, and detects emotions.

## Features

- **Real-time Face Recognition**: Captures and recognizes faces in real-time.
- **Emotion Detection**: Detects six emotions: angry, fear, happy, neutral, sad, and surprise.
- **Data Storage**: Stores attendance data including name, roll number, date, time, and detected emotions in an SQLite database.
- **User Registration**: Allows users to register new faces.
- **View Attendance Records**: Provides a structured view of attendance records.

## Technology Stack

- **Streamlit**: For building the web interface.
- **OpenCV**: For capturing video and processing images.
- **Face Recognition**: For recognizing faces.
- **Deep Learning (CNN)**: For detecting emotions.
- **SQLite**: For storing attendance data.

## Project Structure

The project consists of several components:

1. **Streamlit Configuration**: Sets up the page title, icon, and layout.
2. **Database Setup**: Initializes an SQLite database to store attendance records.
3. **Load Known Faces**: Loads images of known faces and encodes them.
4. **Load Emotion Detection Model**: Loads a pre-trained CNN model for emotion detection.
5. **Camera Functionality**: Starts and stops the camera for capturing images.
6. **Add New Face**: Registers new faces by capturing images and saving them.
7. **Recognize Face and Emotion**: Recognizes faces and detects emotions in captured images.
8. **View Attendance Records**: Retrieves and displays attendance records from the database.

## Code Snippets

### Streamlit Configuration

```python
st.set_page_config(
    page_title="Face and Emotion Recognition Attendance System",
    page_icon=":camera:",
    layout="wide"
)
```

### Database Setup

```python
conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()

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
```

### Load Known Faces

```python
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
```

### Load Emotion Detection Model

```python
@st.cache_resource
def load_emotion_model():
    return load_model('CNN_Model_acc_75.h5')

emotion_model = load_emotion_model()
```

### Recognize Face and Emotion

```python
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
```

## Screenshots

![Screenshot 1](https://github.com/user-attachments/assets/468be307-ce1e-4cd8-8bcc-3200381a593d)  

![Screenshot 2](https://github.com/user-attachments/assets/de6ca45b-c945-4059-ba36-02deededba34)  

![Screenshot 3](https://github.com/user-attachments/assets/5ca0c141-ba6a-48ad-963f-812365edcbb8)  

![Screenshot 4](https://github.com/user-attachments/assets/6921a584-7009-4c78-b7ba-c54316835056)  

## Deployment

To deploy this system, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/lovnishverma/facial-sentiment-analysed-ai-attendance-tracker.git
   ```

2. Navigate to the project directory:
   ```bash
   cd facial-sentiment-analysed-ai-attendance-tracker
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Conclusion

This project provides an innovative way to automate attendance while incorporating emotion tracking, making it suitable for educational institutions, offices, and other organizations.

For more information, visit the [GitHub Repository](https://github.com/lovnishverma/facial-sentiment-analysed-ai-attendance-tracker).

## Live Demo

Face and Emotion Recognition-Based Attendance System

- **Live View** [Password is (nielit)]:

[faceemotionielit.streamlit.app](https://faceemotionielit.streamlit.app/)
