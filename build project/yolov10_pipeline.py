import cv2
from ultralytics import YOLO
import threading

def counting_cards(class_names: list) -> int:
    class_pred = [card[:-1] for card in class_names]

    num_list = ['2', '3', '4', '5', '6', '7', '8', '9', '10']
    ten_list = ['J', 'K', 'Q']

    rs = 0  

    for card in class_pred:
        if card in num_list:
            rs += int(card)
        elif card in ten_list:
            rs += 10
        elif card == 'A':
            rs += 11 if rs + 11 <= 21 else 1

    return rs

def fun_status(res_1: int, res_2: int) -> tuple:
    if (res_1 == 21 and res_2 != 21) or (res_1 < 21 < res_2) or (res_1 > res_2 and res_1 < 21):
        return 'Win', 'Lose'
    elif (res_2 == 21 and res_1 != 21) or (res_2 < 21 < res_1) or (res_2 > res_1 and res_2 < 21):
        return 'Lose', 'Win'
    elif res_1 == res_2:
        return 'Push', 'Push'
    return 'Lose', 'Lose'

class Pipeline:
    
    def __init__(self, path_model: str):
        self.model = YOLO(path_model)
        self.lock = threading.Lock()
        self.current_mode = 2
        self.status_1 = ''
        self.status_2 = ''
        self.res_1 = 0
        self.res_2 = 0

    def train_model(self, save_path, data_path, ep=50, batch_size=8, optimizer_name='Adam', freeze_layers=8):
        self.model.train(
            data=data_path,    
            epochs=ep,            
            batch=batch_size,            
            project='runs/train', 
            name=save_path, 
            optimizer=optimizer_name,
            freeze=freeze_layers
        )
        
    
    def black_jack_game_online_dealer(self):
        cap_1 = cv2.VideoCapture('/dev/video0')  

        if not cap_1.isOpened():
            print("Error: Could not open dealer camera.")
            exit()

        font = cv2.FONT_HERSHEY_SIMPLEX

        while True:
            ret1, frame1 = cap_1.read()
            if not ret1:
                print("Error: Could not read from dealer camera.")
                break
            results_1 = self.model(frame1)
            class_names_1 = {self.model.names[int(box.cls[0])] for box in results_1[0].boxes}
            annotated_frame_1 = results_1[0].plot()
            self.res_1 = counting_cards(class_names_1)

            self.status_1, self.status_2 = fun_status(self.res_1, self.res_2)  

            with self.lock:
                mode = self.current_mode

            
            if mode == 1:
                cv2.putText(annotated_frame_1, f'{", ".join(class_names_1)} -> {self.res_1}', (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            elif mode == 2:
                cv2.putText(annotated_frame_1, f'{self.res_1} -> {self.status_1}', (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            elif mode == 3:
                cv2.putText(annotated_frame_1, f'{self.res_1}', (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            elif mode == 4:
                cv2.putText(annotated_frame_1, f'{self.res_1}', (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            elif mode == 5:
                cv2.putText(annotated_frame_1, f'{self.res_1}', (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            
            ret1, buffer1 = cv2.imencode('.jpg', annotated_frame_1)
            frame1 = buffer1.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')

    
    def black_jack_game_online_player(self):
        cap_2 = cv2.VideoCapture('/dev/video2')  

        if not cap_2.isOpened():
            print("Error: Could not open player camera.")
            exit()

        font = cv2.FONT_HERSHEY_SIMPLEX

        while True:
            ret2, frame2 = cap_2.read()
            if not ret2:
                print("Error: Could not read from player camera.")
                break
            results_2 = self.model(frame2)
            class_names_2 = {self.model.names[int(box.cls[0])] for box in results_2[0].boxes}
            annotated_frame_2 = results_2[0].plot()
            self.res_2 = counting_cards(class_names_2)

            self.status_1, self.status_2 = fun_status(self.res_1, self.res_2)  

            with self.lock:
                mode = self.current_mode

            
            if mode == 1:
                cv2.putText(annotated_frame_2, f'{", ".join(class_names_2)} -> {self.res_2}', (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            elif mode == 2:
                cv2.putText(annotated_frame_2, f'{self.res_2} -> {self.status_2}', (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            elif mode == 3:
                cv2.putText(annotated_frame_2, f'{self.res_2} - Move', (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            elif mode == 4:
                cv2.putText(annotated_frame_2, f'{self.res_2} - Pass', (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            elif mode == 5:
                cv2.putText(annotated_frame_2, f'{self.res_2} - Surrender', (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            
            ret2, buffer2 = cv2.imencode('.jpg', annotated_frame_2)
            frame2 = buffer2.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')

    def set_mode(self, mode):
        with self.lock:
            self.current_mode = mode
