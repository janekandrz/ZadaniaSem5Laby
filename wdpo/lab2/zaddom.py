import cv2 
import numpy as np 


img = cv2.imread("lab2/235-235-max.jpg",cv2.IMREAD_GRAYSCALE)

while True:
    key_code = cv2.waitKey(10)
    if key_code == 27:
        # escape key pressed
        break

    img_scale=cv2.resize(img,dsize=(500,500))
    cv2.imshow('img',img_scale)

    img_inter=cv2.resize(img,dsize=(500,500),interpolation=cv2.INTER_LINEAR)
    cv2.imshow('img_interpol',img_inter)

    img_gauss=cv2.GaussianBlur(img_scale,(15,15),6)
    cv2.imshow("img_guass",img_scale)