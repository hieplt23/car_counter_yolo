from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# for camera
# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 640)

# for input video
cap = cv2.VideoCapture("./videos/traffic.mp4")

# mask image
mask = cv2.imread("./images/mask.png")

# tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# line
line = [300, 350, 830, 350]

# total count
totalCount = []

model = YOLO('../yolo_weights/yolov8n.pt')
class_names = model.names

while(True):
    success, frame = cap.read()

    # resize
    frame = cv2.resize(frame, (1152, 648))
    mask = cv2.resize(mask, (1152, 648))

    imageRegion = cv2.bitwise_and(frame, mask)

    results = model(imageRegion, stream=True)

    detections = np.empty((0, 5))
    for result in results:
        boxes = result.boxes
        for box in boxes:

            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # confidence
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0]*100))/100
            # cvzone.cornerRect(frame, (x1, y1, w, h), colorR=(160, 160, 160), l=10, t=2, rt=2)

            # class name
            cls = int(box.cls[0])
            current_cls = class_names[cls]

            # cvzone.putTextRect(frame, f"{class_names[cls]} {conf}", (max(0, x1), max(10, y1-5)),
            #                    thickness=2, scale=1.5, colorR=(0, 102, 102), offset=3)
            currentArray = np.array([x1, y1, x2, y2, conf])
            detections = np.vstack((detections, currentArray))

    cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3)

    resultsTracker = tracker.update(detections)
    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(frame, (x1, y1, w, h), colorR=(160, 160, 160), l=10, t=2, rt=2)
        cvzone.putTextRect(frame, f"ID: {int(Id)}", (max(0, x1), max(10, y1 - 5)),
                           thickness=3, scale=1.5, colorR=(0, 102, 102), offset=3)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 4, (0, 55, 255), cv2.FILLED)

        if line[0] < cx < line[2] and line[1] - 15 < cy < line[1] + 15:
            if totalCount.count(Id) == 0:
                totalCount.append(Id)
                cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 3)

    cvzone.putTextRect(frame, f"Count: {len(totalCount)}", (960, 50),
                       thickness=3, scale=2, colorR=(0, 102, 102), offset=3)
    cv2.imshow('Output', frame)
    # cv2.imshow('Output2', imageRegion)

    # press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

