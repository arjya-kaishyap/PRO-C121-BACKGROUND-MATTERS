from curses import keyname
from curses.ascii import ESC
import cv2
import time
from cv2 import KeyPoint
import numpy as np

#to save the output in a file output.avi
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter("output.avi",fourcc,20.0,(640, 480))


#starting the webcam

cap = cv2.VideoCapture(0)

#allowing the webcam to start by making the code sleep for 2 seconds
time.sleep(2)
bg = 0

#capturing background for 60 frames
for i in range(60):
    ret,bg = cap.read()

bg = np.flip(bg,axis = 1)

#reading the capture frame until the camera is open 
while(cap.isOpened()):
    ret,img = cap.read()
    if not ret:
        break
    
    #flipping the image for consistency
    img = np.flip(img,axis = 1)
    frame = cv2.resize(frame,(640,480))
    image = cv2.resize(img, (640,480))

    #converting the color from BGR to HSV
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #generating mask to detect red color
    l_black = np.array([30,30,0])
    u_black = np.array([104, 153, 70])
    mask = cv2.inRange(hsv,l_black,u_black)

    #open and expand the image where there is mask_1(color)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask = cv2.morphologyEx(mask,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))

    #selecting only the part that does not have mask
    res = cv2.bitwise_and(bg, bg, mask = mask)
    
    #generating final output by merging res_1 and res_2
    final_output = cv2.addWeighted(res, 1, 1, 0)
    output_file.write(final_output)

    #displaying the output to the user
    cv2.imshow("Magic",final_output)
    cv2.waitKey(1)
if KeyPoint :
    keyname(ESC)
    cap.release()
    cv2.destroyAllWindows()
