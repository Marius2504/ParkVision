import os
import cv2
import numpy as np
from collections import deque
import torch
from shapely.geometry import Polygon, box


def calculate_overlap_area(bbox, spot):
    bbox_poly = box(*bbox)  # Create a polygon from the bounding box
    spot_poly = Polygon(spot)  # Create a polygon from the parking spot

    if not bbox_poly.intersects(spot_poly):
        return 0.0

    intersection_area = bbox_poly.intersection(spot_poly).area
    bbox_area = bbox_poly.area
    return intersection_area / bbox_area


def box_in_parking_spot(box, spot, threshold=0.3):
    overlap_ratio = calculate_overlap_area(box, spot)
    return overlap_ratio >= threshold


def getObjects(model, img):
    results = model(img)
    detections = results.xyxy[0].numpy()
    return detections


# Coordonate locuri de parcare
spots = [
    [(1541, 1038), (1583, 826), (1773, 924), (1760, 1038)],
    [(1376, 919), (1546, 1023), (1590, 831), (1456, 736)],
    [(1236, 809), (1388, 914), (1451, 746), (1339, 667)],
    [(1141, 729), (1261, 818), (1334, 675), (1251, 616)],
    [(1063, 662), (1161, 740), (1266, 628), (1166, 556)],
    [(1074, 685), (974, 619), (1110, 495), (1203, 558)],
    [(930, 560), (1027, 478), (1097, 529), (998, 616)],
    [(865, 515), (923, 564), (1011, 486), (948, 442)],
    [(812, 477), (863, 514), (954, 442), (900, 407)],
    [(771, 444), (820, 479), (902, 407), (863, 376)]
]
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


def task1(path, number):
    img_path = os.path.join(path, number + '.jpg')
    query_file = img_path.replace('.jpg', '_query.txt')
    lines = []

    # Verificăm dacă fișierul _query există
    if os.path.exists(query_file):
        with open(query_file, 'r') as f:
            content = f.readlines()
            n = int(content[0])
            for i in range(n):
                lines.append(int(content[i + 1]))

    file_path = 'evaluation/submission_files/407_Dumitrescu_Marius/Task1/' + number + '_predicted.txt'
    # Verifică dacă fișierul există
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write('')

    file = open(file_path, 'w')
    file.write(f"{n}\n")
    # Detectăm obiectele în imagine
    img = cv2.imread(img_path)
    detections = getObjects(model, img)

    for index, spot in enumerate(spots):
        if index + 1 in lines:
            isCar = False
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if cls == 2 or cls == 7:
                    if box_in_parking_spot((x1, y1, x2, y2), spot):
                        file.write(f"{index + 1} 1\n")
                        isCar = True
                        break
            if isCar == False:
                file.write(f"{index + 1} 0\n")


# Task 2
def task2(video_path, number):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    p = os.path.join(video_path, number + ".mp4")
    cap = cv2.VideoCapture(p)
    file_path = 'evaluation/submission_files/407_Dumitrescu_Marius/Task2/' + number + '_predicted.txt'

    # Verifică dacă fișierul există
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write('')

    file = open(file_path, 'w')
    frame_queue = deque(maxlen=30)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.append(frame)

    for idx, frame in enumerate(frame_queue):
        detections = getObjects(model, frame)
        occupied_spots = set()
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if cls == 2:
                bbox = (x1, y1, x2, y2)
                for i, spot in enumerate(spots):
                    if box_in_parking_spot(bbox, spot):
                        occupied_spots.add(i)
    for i, spot in enumerate(spots):
        if i not in occupied_spots:
            file.write("0\n")
        else:
            file.write("1\n")
    cap.release()



def task3(video_path, number):
    text = os.path.join(video_path, number + ".txt")
    video_path = os.path.join(video_path, number + ".mp4")
    file_path = 'evaluation/submission_files/407_Dumitrescu_Marius/Task3/' + number + '_predicted.txt'
    # Verifică dacă fișierul există
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write('')
    file = open(file_path, 'w')

    file_read = open(text)
    # read file
    lines = file_read.readlines()
    file.write(lines[0])
    if not lines[0].endswith("\n"):
        file.write("\n")
    file.write(lines[1])
    if not lines[1].endswith("\n"):
        file.write("\n")
    max_id = int(lines[0].split()[0])
    id, x1, y1, x2, y2 = lines[1].split()

    id = int(id)
    initial_box = (int(x1), int(y1), int(x2), int(y2))
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    cap = cv2.VideoCapture(video_path)
    score = 0.5
    x1_avg = []
    y1_avg = []
    x2_avg = []
    y2_avg = []
    number_frames = 0
    while cap.isOpened() and id < max_id:
        ret, frame = cap.read()
        if not ret:
            break

        detections = getObjects(model, frame)

        max_score = score
        next_box = (0, 0, 0, 0)
        detected = False
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if cls == 2:  # Assuming cls == 2 represents cars
                bbox = (x1, y1, x2, y2)
                bbox_poly = box(*bbox)
                init_poly = box(*initial_box)
                score = bbox_poly.intersection(init_poly).area / bbox_poly.area
                if score > max_score:
                    max_score = score
                    next_box = bbox
                    detected = True

        if detected:
            number_frames = number_frames + 1
            x1_avg.append(float(next_box[0]) - float(initial_box[0]))
            y1_avg.append(float(next_box[1]) - float(initial_box[1]))
            x2_avg.append(float(next_box[2]) - float(initial_box[2]))
            y2_avg.append(float(next_box[3]) - float(initial_box[3]))
            if len(x1_avg) > 30: x1_avg = x1_avg[1:]
            if len(y1_avg) > 30: y1_avg = y1_avg[1:]
            if len(x2_avg) > 30: x2_avg = x2_avg[1:]
            if len(y2_avg) > 30: y2_avg = y2_avg[1:]
            score = 0.5
        else:
            x1_med = sum(x1_avg) / 30
            x2_med = sum(x2_avg) / 30
            y1_med = sum(y1_avg) / 30
            y2_med = sum(y2_avg) / 30
            next_box = (
                initial_box[0] + x1_med, initial_box[1] + y1_med, initial_box[2] + x2_med, initial_box[3] + y2_med)
            score = 0.2

        id = id + 1
        file.write(f"{id} {int(next_box[0])} {int(next_box[1])} {int(next_box[2])} {int(next_box[3])}\n")

        initial_box = next_box
        # print(initial_box)
        # Desenează dreptunghiul pe frame
        pt1 = (int(next_box[0]), int(next_box[1]))  # colțul stânga-sus
        pt2 = (int(next_box[2]), int(next_box[3]))  # colțul dreapta-jos
        cv2.rectangle(frame, pt1, pt2, (0, 255, 255), 2)

        cv2.imshow('Parking Spot Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



# Task 4
roi_points = np.array([(528, 332), (710, 509), (793, 485), (603, 322)], dtype=np.int32)


class Car:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.isUpdated = False
        self.lastCoord = [(x1, y1, x2, y2)]

    def update(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.lastCoord.append((x1, y1, x2, y2))
        if len(self.lastCoord) > 30:
            self.lastCoord = self.lastCoord[1:]


def task4(video_path, number):
    video_path = os.path.join(video_path, number + ".mp4")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    cap = cv2.VideoCapture(video_path)
    cars = []

    file_path = 'evaluation/submission_files/407_Dumitrescu_Marius/Task4/' + number + '_predicted.txt'
    # Verifică dacă fișierul există
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write('')
    file = open(file_path, 'w')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.polylines(frame, [roi_points], isClosed=True, color=(255, 0, 0), thickness=2)
        detections = getObjects(model, frame)

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if cls == 2 or cls == 7:  # Assuming cls == 2 represents cars
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                if cv2.pointPolygonTest(roi_points, (center_x, center_y), False) == 1.0:
                    newCar = True
                    for car in cars:
                        bbox = (x1, y1, x2, y2)
                        bbox_poly = box(*bbox)

                        carbox = (car.x1, car.y1, car.x2, car.y2)
                        carbox_poly = box(*carbox)
                        score = bbox_poly.intersection(carbox_poly).area / bbox_poly.area
                        if score > 0.25:
                            car.isUpdated = True
                            car.update(x1, y1, x2, y2)
                            newCar = False
                    if newCar:
                        ccar = Car(x1, y1, x2, y2)
                        ccar.isUpdated = True
                        cars.append(ccar)
        for car in cars:
            if not car.isUpdated:
                coord = car.lastCoord[-1]
                x1 = 0
                y1 = 0
                x2 = 0
                y2 = 0
                for index, c in enumerate(car.lastCoord[1:]):
                    x1 = x1 + car.lastCoord[index][0] - car.lastCoord[index - 1][0]
                    y1 = y1 + car.lastCoord[index][1] - car.lastCoord[index - 1][1]
                    x2 = x2 + car.lastCoord[index][2] - car.lastCoord[index - 1][2]
                    y2 = y2 + car.lastCoord[index][3] - car.lastCoord[index - 1][3]
                x1 = x1 / 30 + coord[0]
                y1 = y1 / 30 + coord[1]
                x2 = x2 / 30 + coord[2]
                y2 = y2 / 30 + coord[3]
                car.update(x1, y1, x2, y2)
            car.isUpdated = False
    file.write(f"{len(cars)}")
    cap.release()



# execute task 1
def ext1(files_path):
    maxi = 4
    for i in range(1, 16):
        number = ''
        if i > 10:
            maxi = 5
        for j in range(1, maxi):
            if i < 10 and number != '0':
                number = number + '0'
            task1(files_path, number + f'{i}_{j}')


# execute task 2
def ext2(files_path):
    for i in range(1, 16):
        number = ''
        if i < 10 and number != '0':
            number = number + '0'
        task2(files_path, number + f'{i}')


def ext3(files_path):
    for i in range(5, 16):
        number = ''
        if i < 10 and number != '0':
            number = number + '0'
        task3(files_path, number + f'{i}')


def ext4(files_path):
    for i in range(1, 16):
        number = ''
        if i < 10 and number != '0':
            number = number + '0'
        task4(files_path, number + f'{i}')


#ext1('train/Task1')
#ext2('train/Task2')
#ext3('train/Task3')
#ext4('train/Task4')
