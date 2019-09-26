import cv2
import numpy as np
import time
import argparse
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False, default="/videos/car.mp4",
    help="path to input video")
ap.add_argument("-o", "--output", required=False, default="",
    help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
    help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

labels = open(os.path.sep.join([args["yolo"], "coco.names"])).read().strip().split('\n')

weights = os.path.sep.join([args["yolo"], "yolov3-tiny.weights"])
config = os.path.sep.join([args["yolo"], "yolov3-tiny.cfg"])

net = cv2.dnn.readNet(config, weights)

# Load video

vc = cv2.VideoCapture('videos/car.mp4')

# Get layer names

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

frame_count = 0

images = []

while(vc.isOpened()):    
    
    ret, frame = vc.read()
    if ret == False:
        break
    start_time = time.time()
    
    # Get frame size
    (h, w) = frame.shape[:2]

    # Preprocess the image 
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB = True, crop = False)
    
    # Make prediction on image
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    IDs = []

    confidence_threshold = args['confidence']
    
    # For each output save object prediction and box if confidence > threshold
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            ID = np.argmax(scores)
            confidence = scores[ID]

            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                
                # Get top-left corner coords of a frame
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                # Append prediction
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                IDs.append(ID)

    # Set non max suppresion threshold
    nms_threshold = args['threshold']
    
    # Run NMS to get only needed boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    
    np.random.seed(33)
    colors = np.random.randint(0, 255, size=(len(labels), 3),
        dtype="uint8")

    # Draw boxes and label names
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in colors[IDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = str(labels[IDs[i]])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
    
    end_time = time.time()
    frame_count += 1
    if frame_count % 20 == 0:
        print("YOLO runs at:", int(1 / (end_time - start_time)), " fps")
    
    # Resize and show the image
    frame = cv2.resize(frame, (1280, 720))
    images.append(frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    else:
        continue

out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 24, (1280, 720))

for i in range(len(images)):
    out.write(images[i])
out.release()

vc.release()
cv2.destroyAllWindows()

print("end")