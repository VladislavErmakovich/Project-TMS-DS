from flask import Flask, render_template, Response, request, send_from_directory
from yolov10_pipeline import Pipeline

app = Flask(__name__)

pipeline = Pipeline('models/best.pt')

@app.route('/dealer_video_feed')
def dealer_video_feed():
    """Маршрут для потока с камеры дилера"""
    return Response(pipeline.black_jack_game_online_dealer(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/player_video_feed')
def player_video_feed():
    """Маршрут для потока с камеры игрока"""
    return Response(pipeline.black_jack_game_online_player(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/image.jpg')
def image():
    """Путь для отдачи изображения"""
    return send_from_directory('static', 'image.jpg')

@app.route('/set_mode/<int:mode>')
def set_mode(mode):
    pipeline.set_mode(mode)
    return '', 204  

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)