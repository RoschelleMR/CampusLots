import cv2
import numpy as np

#Web Camera

cap = cv2.VideoCapture('video2.mp4')


# Initialize Substractor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

# WHILE LOOP: Continuously have the window open until exit key is pressed
# or all frames have been displayed
while True:
    
    ret,frame = cap.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    
    # applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterShape = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    # display line on video capture
    cv2.line(frame, (0,500), (1280,500), (222, 49, 99), 4, 2)
    
    
    # for (i,c) in enumerate(counterShape):
    #     (x,y,w,h) = cv2.boundingRect()
    
    
    # cv2.imshow('Detector', dilatada) # displays frames
    
    cv2.imshow('Original', frame) # displays frames
    
    if cv2.waitKey(100) & 0xFF == ord('q'): #change frame every 0.1s, exit by pressing q
        break
  
cap.release() 
cv2.destroyAllWindows()