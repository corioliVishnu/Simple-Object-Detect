import cv2
import numpy as np

# Load YOLO model
yolo_weights = 'yolov3.weights'
yolo_config = 'yolov3.cfg'
yolo_classes = 'coco.names'

net = cv2.dnn.readNet(yolo_weights, yolo_config)

# Load COCO class names
with open(yolo_classes, 'r') as f:
    classes = f.read().strip().split('\n')

# Open a video capture object (0 represents the default camera)
cap = cv2.VideoCapture(0) ----------> # You can replace this number with video

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Perform object detection
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Process YOLO predictions and draw bounding boxes
    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0:  # Adjust confidence threshold as needed
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) to filter out redundant bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices:
        box = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color1 = (0, 255, 0) # Green color for bounding boxes
        color2 = (0, 0, 0)  # Black color for text
        cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color1, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 2)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
