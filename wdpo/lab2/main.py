import cv2
import numpy as np


def empty_callback(value):
    pass

# create a black image, a window
img = cv2.imread("lab2/AdditiveColor.png",cv2.IMREAD_GRAYSCALE)
img_scale=cv2.resize(img,dsize=(300,300))
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('progowanie', 'image', 0, 255, empty_callback)

while True:
    # sleep for 10 ms waiting for user to press some key, return -1 on timeout
    key_code = cv2.waitKey(10)
    if key_code == 27:
        # escape key pressed
        break
    cv2.imshow('image', img_scale) 


    # get current positions of four trackbars
    val = cv2.getTrackbarPos('progowanie', 'image')
    _, img_scale =cv2.threshold(img_scale,val,255,cv2.THRESH_BINARY)

    cv2.imshow("res",img_scale)
     

# closes all windows (usually optional as the script ends anyway)
cv2.destroyAllWindows()

#zadanie domowe na kostke 