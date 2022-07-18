import numpy as np 
import cv2 as cv
import time
while (1):
    depth_img = cv.imread ("depth_out.png")
    gray_out = cv.cvtColor(depth_img, cv.COLOR_BGR2GRAY)
    ret,thresh = cv.threshold(gray_out,238,255,cv.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    fin_thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    contours, hierarchy = cv.findContours(fin_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(gray_out, contours, -1, (0,255,0), 3)
    count=0
    for i in contours:
        M = cv.moments(i)
        if M['m00'] != 0:
            dist = cv.pointPolygonTest((contours[count]), (320,240), True)
            print ("Object:", count, " Distance: ", dist)
            count = count + 1
            if (dist > 15):
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                if (cx >= 360):
                    print ("Right")
                elif (cx <=280):
                    print ("Left")
                else:
                    print ("Center")
            else:
                print ("Center")
    cv.imwrite('image.png', gray_out)
    time.sleep(0.88)