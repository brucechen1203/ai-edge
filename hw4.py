import cv2
import numpy as np

# === 1. Load video and model ===
cap = cv2.VideoCapture("original_video/full.mp4")

net = cv2.dnn.readNetFromCaffe(
    "MobileNet-SSD/deploy.prototxt",
    "MobileNet-SSD/mobilenet_iter_73000.caffemodel"
)

# ROI 範圍
roi1_startX, roi1_startY, roi1_endX, roi1_endY = 180, 100, 730, 400
roi2_startX, roi2_startY, roi2_endX, roi2_endY = 170, 100, 350, 400

# 門狀態點位
door_markerX, door_markerY = 260, 130
door_marker_radius = 5
door_marker_color_threhold = 10

output_video = cv2.VideoWriter(
    'output2_video.mp4',
    cv2.VideoWriter_fourcc(*'MP4V'),
    30,
    (int(cap.get(3)), int(cap.get(4))),
    True
)

# === 2. Confirm model loaded ===
if net.empty():
    print("No data")
else:
    print("Success model load")

max_people_roi2_during_open = 0

# === 3. Start frame-by-frame detection ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === 檢查門開關 ===
    door_roi = frame[door_markerY - door_marker_radius : door_markerY + door_marker_radius,
                     door_markerX - door_marker_radius : door_markerX + door_marker_radius]
    door_mean_color = np.mean(door_roi, axis=(0, 1))
    elevator_open = door_mean_color[1] - door_mean_color[0] > door_marker_color_threhold
    door_status = "Open" if elevator_open else "Close"
    cv2.putText(frame, f"Door status: {door_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # === ROI1: 持續偵測畫面 ===
    roi1 = frame[roi1_startY:roi1_endY, roi1_startX:roi1_endX]
    blob1 = cv2.dnn.blobFromImage(roi1, 0.007843, (300, 300), 127.5)
    net.setInput(blob1)
    detections1 = net.forward()

    roi1_people_count = 0
    for i in range(detections1.shape[2]):
        confidence = detections1[0, 0, i, 2]
        class_id = int(detections1[0, 0, i, 1])
        if confidence > 0.7 and class_id == 15:
            x1 = int(detections1[0, 0, i, 3] * roi1.shape[1]) + roi1_startX
            y1 = int(detections1[0, 0, i, 4] * roi1.shape[0]) + roi1_startY
            x2 = int(detections1[0, 0, i, 5] * roi1.shape[1]) + roi1_startX
            y2 = int(detections1[0, 0, i, 6] * roi1.shape[0]) + roi1_startY
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "human", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            roi1_people_count += 1

    # 畫 ROI1
    cv2.rectangle(frame, (roi1_startX, roi1_startY), (roi1_endX, roi1_endY), (255, 0, 0), 2)
    cv2.putText(frame, f"external people: {roi1_people_count}", (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # === ROI2: 只在電梯門開啟時偵測 ===
    roi2_count = 0
    if elevator_open:
        roi2 = frame[roi2_startY:roi2_endY, roi2_startX:roi2_endX]
        blob2 = cv2.dnn.blobFromImage(roi2, 0.007843, (300, 300), 127.5)
        net.setInput(blob2)
        detections2 = net.forward()

        for j in range(detections2.shape[2]):
            confidence = detections2[0, 0, j, 2]
            class_id = int(detections2[0, 0, j, 1])
            if confidence > 0.6 :
                x1 = int(detections2[0, 0, j, 3] * roi2.shape[1]) + roi2_startX
                y1 = int(detections2[0, 0, j, 4] * roi2.shape[0]) + roi2_startY
                x2 = int(detections2[0, 0, j, 5] * roi2.shape[1]) + roi2_startX
                y2 = int(detections2[0, 0, j, 6] * roi2.shape[0]) + roi2_startY
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(frame, "人", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 50), 2)
                roi2_count += 1

        # 更新最大值
        if roi2_count > max_people_roi2_during_open:
            max_people_roi2_during_open = roi2_count

        # 畫 ROI2
        #cv2.rectangle(frame, (roi2_startX, roi2_startY), (roi2_endX, roi2_endY), (0, 0, 255), 2)
        #cv2.putText(frame, f"ROI2 people: {roi2_count}", (roi2_startX, roi2_startY - 10),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 顯示最大 ROI2 人數（只在門開啟時計算）
    cv2.putText(frame, f"people in elevator: {max_people_roi2_during_open}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # 寫入影片
    output_video.write(frame)

# === 5. Release resources ===
cap.release()
output_video.release()
print("Success video output")