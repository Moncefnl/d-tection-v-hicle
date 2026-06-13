import cv2
import numpy as np
import time
from tkinter import Tk, Label, Entry, Button

line_y = 80
scale_factor = 1.2

class SpeedDetection:
    def __init__(self, model_cfg, model_weights, class_names, video_src, max_speed, min_speed, alert_distance):
        self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        with open(class_names, 'r') as f:
            self.classes = f.read().strip().split('\n')

        self.cap = cv2.VideoCapture(video_src)
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.alert_distance = alert_distance
        self.start_time = None
        self.frame_count = 0
        self.car_counter = 0
        self.car_speeds = {}
        self.prev_center = None
        self.distance_recorded = False

    def get_outputs(self, image):
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (320, 320), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return outputs

    def detect_objects(self, outputs, height, width):
        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and self.classes[class_id] == "car":
                    box = detection[0:4] * np.array([width, height, width, height])
                    centerX, centerY, w, h = box.astype("int")
                    x = int(centerX - w / 2)
                    y = int(centerY - h / 2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            return [(boxes[i], confidences[i], class_ids[i]) for i in indices.flatten()]
        return []

    def calculate_speed(self, time_elapsed):
        try:
            speed = (9.144 / 1000) / (time_elapsed / 3600)
            return speed
        except ZeroDivisionError:
            return 0

    def calculate_distance(self, center1, center2):
        if center1 is None or center2 is None:
            return None
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance * 0.1  # Assuming 1 pixel = 0.1 meter

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None

        frame_resized = cv2.resize(frame, (640, 360))
        height, width = frame_resized.shape[:2]

        outputs = self.get_outputs(frame_resized)
        detections = self.detect_objects(outputs, height, width)

        for i, (box, confidence, class_id) in enumerate(detections):
            x, y, w, h = box
            center = (int((x + x + w) / 2), int((y + y + h) / 2))
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(frame_resized, center, 5, (0, 255, 0), -1)

            car_label = f"Car number {self.car_counter + i + 1}"
            cv2.putText(frame_resized, car_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if line_y - 5 <= center[1] <= line_y + 5 and self.start_time is None:
                self.start_time = time.time()
                self.car_counter += 1
            elif line_y + 30 <= center[1] <= line_y + 40 and self.start_time is not None:
                time_elapsed = time.time() - self.start_time
                speed = self.calculate_speed(time_elapsed)
                self.car_speeds[self.car_counter] = speed

                speed_message = f"Vehicle Speed number {self.car_counter}: {speed:.2f} km/h"
                if speed > self.max_speed:
                    print(f"Warning: Car {self.car_counter} exceeded max speed!")
                elif speed < self.min_speed:
                    print(f"Warning: Car {self.car_counter} below min speed!")

                current_datetime = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"{speed_message} | Date & Time: {current_datetime}")

                cv2.putText(frame_resized, speed_message, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame_resized, f"Date & Time: {current_datetime}", (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                self.start_time = None

            if center[1] > line_y and not self.distance_recorded:
                if self.prev_center is not None:
                    distance = self.calculate_distance(self.prev_center, center)
                    if distance is not None:
                        print(f"Distance between vehicle {self.car_counter - 1} and {self.car_counter}: {distance:.2f} meters")
                        if distance < self.alert_distance:
                            print(f"Alert: Distance between vehicle {self.car_counter - 1} and {self.car_counter} is less than {self.alert_distance} meters!")
                    self.distance_recorded = True
                self.prev_center = center
            elif center[1] < line_y:
                self.distance_recorded = False

        return True, frame_resized

    def run(self):
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video', 800, 600)

        root = Tk()
        root.withdraw()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()

        window_width = 800
        window_height = 600
        x = int((screen_width - window_width) / 2)
        y = int((screen_height - window_height) / 2)
        cv2.moveWindow('Video', x, y)

        while True:
            ret, frame = self.process_frame()
            if not ret:
                break

            if frame is not None:
                cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

def get_speed_limits():
    root = Tk()
    root.title("Welcome to Speed Limits Application")
    root.geometry("600x400")

    def set_speed_limits():
        global max_speed, min_speed, alert_distance
        max_speed = int(max_entry.get())
        min_speed = int(min_entry.get())
        alert_distance = float(alert_entry.get())
        root.destroy()

    max_label = Label(root, text="Enter the maximum speed (km/h):", font=("Franklin Gothic Demi Cond", 14))
    max_label.pack(pady=10)

    max_entry = Entry(root, font=("Franklin Gothic Demi Cond", 18), width=40)
    max_entry.pack(pady=5)

    min_label = Label(root, text="Enter the minimum speed (km/h):", font=("Franklin Gothic Demi Cond", 14))
    min_label.pack(pady=10)

    min_entry = Entry(root, font=("Franklin Gothic Demi Cond", 18), width=40)
    min_entry.pack(pady=5)

    alert_label = Label(root, text="Enter the alert distance (meters):", font=("Franklin Gothic Demi Cond", 14))
    alert_label.pack(pady=10)

    alert_entry = Entry(root, font=("Franklin Gothic Demi Cond", 18), width=40)
    alert_entry.pack(pady=5)

    button = Button(root, text="Set Limits", command=set_speed_limits, font=("Franklin Gothic Demi Cond", 14), width=55)
    button.pack(pady=14)
    root.mainloop()

# Get speed limits from user
get_speed_limits()

# Initialize and run the speed detection
model_cfg = r'C:\\Users\\OMG\\Desktop\\Speed_detection\\yolov3-tiny.cfg'
model_weights = r'C:\\Users\\OMG\\Desktop\\Speed_detection\\yolov3-tiny.weights'
class_names = r'C:\\Users\\OMG\\Desktop\\Speed_detection\\coco.names'
video_src = r'C:\\Users\\OMG\\Desktop\\Speed_detection\\Video_detection.mp4'

detector = SpeedDetection(model_cfg, model_weights, class_names, video_src, max_speed, min_speed, alert_distance)
detector.run()

