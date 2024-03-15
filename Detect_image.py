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

# Read an image
image = cv2.imread('2.jpg')

# Get image dimensions
height, width, _ = image.shape

# Create a blob from the image
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# Set the input to the YOLO network
net.setInput(blob)

# Get output layer names
out_layer_names = net.getUnconnectedOutLayersNames()

# Run forward pass and get output
outs = net.forward(out_layer_names)

# Parse YOLO predictions
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


# Apply Non-Maximum Suppression (NMS)
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0, 0.4)

# Draw bounding boxes on the image for the selected indices after NMS
for i in indices:
    box = boxes[i]
    confidence = confidences[i]
    class_id = class_ids[i]

# Draw bounding box and label
    x, y, w, h = box
    label = f"{classes[class_id]}: {confidence:.2f}"
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Resize image to fit the screen while preserving the aspect ratio
screen_height, screen_width = 600, 800  # Set your screen resolution
image_aspect_ratio = image.shape[1] / image.shape[0]
screen_aspect_ratio = screen_width / screen_height

if image_aspect_ratio > screen_aspect_ratio:
    # Fit to width
    new_width = screen_width
    new_height = int(new_width / image_aspect_ratio)
else:
    # Fit to height
    new_height = screen_height
    new_width = int(new_height * image_aspect_ratio)

#resized_image = cv2.resize(image, (new_width, new_height))

# Display the image with bounding boxes and display the resized image in full screen
#cv2.namedWindow('Object Detection', cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty('Object Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
