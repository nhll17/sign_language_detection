from flask import Flask, render_template, Response, jsonify
import cv2
import pickle
import numpy as np
import mediapipe as mp

app = Flask(__name__)

# Load trained model
with open("model/sign_model.pkl", "rb") as f:
    model = pickle.load(f)

current_prediction = "..."

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

def generate_frames():
    global current_prediction
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract features (you can customize this part)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                if len(landmarks) == 63:  # 21 landmarks * 3 coords
                    prediction = model.predict([landmarks])
                    current_prediction = prediction[0]

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', prediction=current_prediction)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def prediction():
    return jsonify(prediction=current_prediction)

if __name__ == '__main__':
    app.run(debug=True)
