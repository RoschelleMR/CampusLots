import cv2
import numpy as np
from ultralytics import YOLO
import torch

# USE GPU
device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0)

#Web Camera
cap = cv2.VideoCapture('video2_low quality.mp4')

#Import YOLO
yolo_version = 'yolov8n.pt'
model = YOLO(yolo_version)

# WHILE LOOP: Continuously have the window open until exit key is pressed
# or all frames have been displayed
while True:
    
    ret,frame = cap.read()
    ret2,frame2 = cap.read()
    ret3, frame3 = cap.read()
    
    results = model.track(frame, persist=False) #PERSIST: When True, the model will set ids to previously discovered detections in subsequent frames
    results2 = model.track(frame2, persist=False)
    results3 = model.track(frame3, persist=False)
    
    # Visualize the results of the model on the frame in a bounding box
    annotated_frame = results[0].plot()
    annotated_frame2 = results2[0].plot()
    annotated_frame3 = results3[0].plot()
    
    # display lines on video capture
    
    # right line
    cv2.line(annotated_frame, (800,500), (1240,500), (222, 49, 99), 4, 2)
    
    # left line
    cv2.line(annotated_frame, (750,500), (250,500), (20, 49, 200), 4, 2)
    
    cv2.putText(annotated_frame, 'COUNT: ', (50, 50) , cv2.FONT_HERSHEY_SIMPLEX , 1,  
                 (0, 50, 255) , 4, cv2.LINE_AA, False) 
    
    # Display the annotated frame
    cv2.imshow("YOLOv8 Tracking", annotated_frame)
    cv2.imshow("YOLOv8 2nd", annotated_frame2)
    cv2.imshow("YOLOv8 3nd", annotated_frame3)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): #change frame every 0.1s, exit by pressing q
        break
  
cap.release() 
cv2.destroyAllWindows()