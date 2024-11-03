import cv2
from ultralytics import YOLO
import threading

# Функция для подсчета очков с карт
def counting_cards(class_names: list) -> int:
    class_pred = [card[:-1] for card in class_names]

    num_list = ['2', '3', '4', '5', '6', '7', '8', '9', '10']
    ten_list = ['J', 'K', 'Q']

    rs = 0  

    for i in range(len(class_pred)):
        if class_pred[i] in num_list:
            rs+=int(class_pred[i])
        elif class_pred[i] in ten_list:
            rs+=10
        elif class_pred[i]=='A':
            if rs<21:
                rs+=11
            else:
                rs+=1

    return rs

 def fun_status(res_1: int, res_2: int) -> tuple:
     st_1, st_2 = '', ''
    
     if (res_1 > res_2 and res_1 < 21) or (res_1 < res_2 and res_2 > 21) or (res_1 == 21 and res_2 != 21):
         st_1, st_2 = 'Win', 'Lose'
     elif (res_2 > res_1 and res_2 < 21) or (res_2 < res_1 and res_1 > 21) or (res_2 == 21 and res_1 != 21):
         st_1, st_2 = 'Lose', 'Win'
     elif res_1 > 21 and res_2 > 21:
         if res_1 < res_2:
             st_1, st_2 = 'Win', 'Lose'
         else:
             st_1, st_2 = 'Lose', 'Win'
     elif res_1 == res_2:
         st_1, st_2 = 'Push', 'Push'
     return st_1, st_2


class Pipeline:
    # Инициализация
    def __init__(self, path_model: str):
        self.model = YOLO(path_model)

    # Обучение модели с параметрами
    def train_model(self, save_path, data_path, ep=50, batch_size=8,optomizer_name = 'Adam', size_fr = 8):
        self.model.train(
            data=data_path,    
            epochs=ep,            
            batch=batch_size,            
            project='runs/train', 
            name=save_path, 
            optomizer = optomizer_name,
            freeze = size_fr
        )
        
    # Метод, который показывает как детектируются карты
    def detection_cards(self):
        cap_1 = cv2.VideoCapture(0)

        if not cap_1.isOpened():
            print("Не удалось открыть камерy.")
            exit()

        while True:
            ret1, frame1 = cap_1.read()
            if not ret1 :
                print("Ошибка при чтении камеры.")
                break

   
            results_1 = self.model(frame1)
            annotated_frame_1 = results_1[0].plot() 
            font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX     

            cv2.imshow('Детекция карт_1', annotated_frame_1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap_1.release()
        cv2.destroyAllWindows()

    def black_jack_game_offline(self):
        cap_1 = cv2.VideoCapture(0)
        cap_2 = cv2.VideoCapture(1)

        if not cap_1.isOpened() or not cap_2.isOpened():
            print("Не удалось открыть одну из камер.")
            exit()

        # Шрифт для вывода текста
        font = cv2.FONT_HERSHEY_SIMPLEX

        current_mode = 2
        while True:
            ret1, frame1 = cap_1.read()
            ret2, frame2 = cap_2.read()
            if not ret1 or not ret2:
                print("Ошибка при чтении с одной из камер.")
                break

            # Применяем модель к кадрам
            results_1 = self.model(frame1)
            results_2 = self.model(frame2)

            # Для камеры 1
            class_names_1 = [self.model.names[int(box.cls[0])] for box in results_1[0].boxes]  # Имена классов для каждого объекта
            annotated_frame_1 = results_1[0].plot()  # Получаем изображение с аннотациями
            class_names_1 = list(set(class_names_1))
            res_1  = counting_cards( class_names_1)

            # Для камеры 2
            class_names_2 = [self.model.names[int(box.cls[0])] for box in results_2[0].boxes]  # Имена классов для каждого объекта
            annotated_frame_2 = results_2[0].plot()  # Получаем изображение с аннотациями
            class_names_2 = list(set(class_names_2))
            res_2  = counting_cards( class_names_2)

            # Вычисляем статус игрока и дилера(победа, поражение, ничья)
            status_1, status_2 = fun_status(res_1, res_2)
            
            # Режимы работы камер 
            elif current_mode==1:
                cv2.putText(annotated_frame_1, f'Dealer: {", ".join(class_names_1)} -> {res_1}', (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(annotated_frame_2, f'Player: {", ".join(class_names_2)} -> {res_2}', (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow('Детекция карт_1 (Dealer)', annotated_frame_1)
                cv2.imshow('Детекция карт_2 (Player)', annotated_frame_2)
                
            elif current_mode==2:
                cv2.putText(frame1, f'Dealer: {res_1}->{status_1}', (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame2, f'Player: {res_2}->{status_2}', (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow('Детекция карт_1 (Dealer)', frame1)
                cv2.imshow('Детекция карт_2 (Player)', frame2)
        
            key = cv2.waitKey(1) & 0xFF
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('a'):
                current_mode = 1
            elif cv2.waitKey(1) & 0xFF == ord('p'):
                current_mode = 2

        cap_1.release()
        cap_2.release()
        cv2.destroyAllWindows()

    def black_jack_game_online(self):
        self.lock = threading.Lock()
        self.current_mode = 2
        cap_1 = cv2.VideoCapture(0)
        cap_2 = cv2.VideoCapture(1)

        if not cap_1.isOpened() or not cap_2.isOpened():
            print("Не удалось открыть одну из камер.")
            exit()

        font = cv2.FONT_HERSHEY_SIMPLEX

        self.current_mode = 2
        while True:
            ret1, frame1 = cap_1.read()
            ret2, frame2 = cap_2.read()
            if not ret1 or not ret2:
                print("Ошибка при чтении с одной из камер.")
                break

            # Применяем модель к кадрам
            results_1 = self.model(frame1)
            results_2 = self.model(frame2)

            # Для камеры 1
            class_names_1 = [self.model.names[int(box.cls[0])] for box in results_1[0].boxes]  # Имена классов для каждого объекта
            annotated_frame_1 = results_1[0].plot()  # Получаем изображение с аннотациями
            class_names_1 = list(set(class_names_1))
            res_1  = counting_cards( class_names_1)

            # Для камеры 2
            class_names_2 = [self.model.names[int(box.cls[0])] for box in results_2[0].boxes]  # Имена классов для каждого объекта
            annotated_frame_2 = results_2[0].plot()  # Получаем изображение с аннотациями
            class_names_2 = list(set(class_names_2))
            res_2  = counting_cards( class_names_2)

            # Вычисляем статус игрока и дилера(победа, поражение, ничья)
            status_1, status_2 = fun_status(res_1, res_2)

            # Режимы работы камер
            if self.current_mode == 0:
                cv2.putText(frame1, f'Dealer: {", ".join(class_names_1)} -> {res_1}', (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame2, f'Player: {", ".join(class_names_2)} -> {res_2}', (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow('Детекция карт_1 (Dealer)', frame1)
                cv2.imshow('Детекция карт_2 (Player)', frame2)
                
            elif self.current_mode == 1:
                cv2.putText(annotated_frame_1, f'Dealer: {", ".join(class_names_1)} -> {res_1}', (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(annotated_frame_2, f'Player: {", ".join(class_names_2)} -> {res_2}', (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow('Детекция карт_1 (Dealer)', annotated_frame_1)
                cv2.imshow('Детекция карт_2 (Player)', annotated_frame_2)
                
            elif self.current_mode == 2:
                cv2.putText(frame1, f'Dealer: {res_1}->{status_1}', (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame2, f'Player: {res_2}->{status_2}', (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow('Детекция карт_1 (Dealer)', frame1)
                cv2.imshow('Детекция карт_2 (Player)', frame2)

            ret1, buffer1 = cv2.imencode('.jpg', frame1)
            ret2, buffer2 = cv2.imencode('.jpg', frame2)
            frame1 = buffer1.tobytes()
            frame2 = buffer2.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n'
                   b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')

    def set_mode(self, mode):
        with self.lock:
            self.current_mode = mode 


