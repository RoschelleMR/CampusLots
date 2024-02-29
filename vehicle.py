import cv2
import numpy as np
from ultralytics import YOLO

#Web Camera
cap = cv2.VideoCapture('video2.mp4')

#Import YOLO
yolo_version = 'yolov8n.pt'
model = YOLO(yolo_version)

# WHILE LOOP: Continuously have the window open until exit key is pressed
# or all frames have been displayed
while True:
    
    ret,frame = cap.read()
    results = model.track(frame, persist=True) #PERSIST: When True, the model will set ids to previously discovered detections in subsequent frames
    
    # Visualize the results of the model on the frame in a bounding box
    annotated_frame = results[0].plot()
    
    # display lines on video capture
    
    # right line
    cv2.line(annotated_frame, (800,500), (1240,500), (222, 49, 99), 4, 2)
    
    # left line
    cv2.line(annotated_frame, (750,500), (250,500), (20, 49, 200), 4, 2)
    
    cv2.putText(annotated_frame, 'COUNT: ', (50, 50) , cv2.FONT_HERSHEY_SIMPLEX , 1,  
                 (0, 50, 255) , 4, cv2.LINE_AA, False) 
    
    # Display the annotated frame
    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'): #change frame every 0.1s, exit by pressing q
        break
  
cap.release() 
cv2.destroyAllWindows()