from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_faces = mp.solutions.face_mesh
draw = mp.solutions.drawing_utils
face_draww=draw.DrawingSpec(thickness=1,circle_radius=1)

hands = mp_hands.Hands()
faces = mp_faces.FaceMesh()
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        result2 = faces.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        if result2.multi_face_landmarks:
            for face_landmarks in result2.multi_face_landmarks:
                draw.draw_landmarks(frame, face_landmarks, mp_faces.FACEMESH_CONTOURS,face_draww,face_draww)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
