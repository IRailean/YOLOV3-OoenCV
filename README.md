# YOLOV3-OpenCV
Object detection with Yolov3 using OpenCV

## Parameters
To run this script use these parameters:  
**-y --yolo - specifies YOLO model directory  
-i --input - path to input video (/videos/car.mp4 by default)  
-o --output - path to output video (by default video will be create in the same folder where script is located)  
-c --confidence - confidence threshold  
-t --threshold - NMS threshold**  

## Run script
To see how it works run this command. 
```
python YOLOOpenCV.py --yolo yolo-model
```

Here is the example. (Tiny yolov3 model is used)  
You can download other yolo models from [here](https://pjreddie.com/darknet/yolo/)

<img src ="https://im.ezgif.com/tmp/ezgif-1-8d170aba5d95.gif" width="600" height="400"/>
