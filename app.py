from flask import Flask, Response, render_template
from yolov10_pipeline import Pipeline

app = Flask(__name__)
pipeline = Pipeline('models/best.pth')  # Path to your YOLOv10 model

@app.route('/')
def index():
    return render_template('index.html')  # Render the main HTML template

@app.route('/video_feed')
def video_feed():
    return Response(pipeline.black_jack_game_online(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode/<int:mode>')
def set_mode(mode):
    pipeline.set_mode(mode)
    return 'Mode set'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Run the app