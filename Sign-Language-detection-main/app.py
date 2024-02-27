from flask import Flask, render_template, Response
import cv2
from test import HandGestureRecognition

app = Flask(__name__)
hgr = HandGestureRecognition()

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        success, frame = hgr.get_frame()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

