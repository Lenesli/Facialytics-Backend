from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import cv2
from werkzeug.utils import secure_filename
import os
from emotion_app.emotion_detection import detect_emotion  # Import the detect_emotion function

app = Flask(__name__)

# Configuration for file uploads
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Perform emotion detection
        emotion = detect_emotion(file_path)  # Replace with your actual function
        return jsonify({'emotion': emotion})
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/video_feed')
def video_feed():
    return Response(stream_with_context(generate_frames()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    capture = cv2.VideoCapture(0)

    while True:
        success, frame = capture.read()
        if not success:
            break

        emotion_result = detect_emotion(frame)
        face_emotion = emotion_result["emotion"]
        face_text = f"{face_emotion}: {emotion_result['probability']:.2f}%"
        cv2.putText(frame, face_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    capture.release()

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
