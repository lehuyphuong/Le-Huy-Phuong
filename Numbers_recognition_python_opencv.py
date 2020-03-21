import numpy as np
import cv2
import math
#import imutils
from math import atan2

cropVals = 100,100,350,450

cap = cv2.VideoCapture(0)

def nothing(x):
    pass

cv2.namedWindow("HSV value")
cv2.createTrackbar("H_min","HSV value",0,255,nothing)
cv2.createTrackbar("H_max","HSV value",0,255,nothing)
cv2.createTrackbar("S_min","HSV value",0,255,nothing)
cv2.createTrackbar("S_max","HSV value",0,255,nothing)
cv2.createTrackbar("V_min","HSV value",0,255,nothing)
cv2.createTrackbar("V_max","HSV value",0,255,nothing)

def getContours(imgCon, imgMatch):

    #reduce noise before analysis
    kernel = np.ones((5,5),np.uint8)
    imgCon = cv2.erode(imgCon,kernel,iterations = 2)

    contours, hierarchy = cv2.findContours(imgCon,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    imgCon = cv2.cvtColor(imgCon, cv2.COLOR_GRAY2BGR)

    myCounter = 0

    print(hierarchy)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if(area > 1200):
            cv2.drawContours(imgCon,cnt,-1, (255,255,255),2)
            cv2.drawContours(imgMatch,cnt,-1,(255,255,255),2)

            Group1 = np.array([[[-1,-1,-1,-1]]])
            Group2 = np.array([[[-1,-1,1,-1],[-1,-1,-1,0]]])
            Group3 = np.array([[[-1,-1,1,-1],[2,-1,-1,0],[-1,1,-1,0]]])

            #calculate the perimeter
            peri = cv2.arcLength(cnt, True)

            #approximate the shape of object
            approx = cv2.approxPolyDP(cnt,0.015*peri, True) # this line is very important because it considers the percentage of accuracy of tracking coordinates, 
                                                            #the smaller number will get the better result.

            #find centroid of the object
            M = cv2.moments(cnt)
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            cv2.circle(imgCon,(cx,cy),5,(0,0,255),-1) #red point
            CenterPoint = (cx,cy)
            #cv2.line(imgCon,(cx,cy),(cx,cy),(0,255,0),2) #green line


            # put a frame for the object
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(imgCon,(x,y),(x+w,y+h),(125,230,0),2) # color light blue
            cv2.line(imgCon,(x,int((y+h)/2)),(x+w,int((y+h)/2)),(0,0,125),2) # color red
            
            #this line will help you to detect number 3
            halfline_horizonal = np.array([x,int((y+h)/2)])
            #halfline_vertical = np.array([int((x+w)/2),y])

            #access the coordinates
            for g in range(0,len(approx)):
                for y in range(0,len(approx[0][0])-1):
                    cv2.circle(imgCon,(approx[g][0][y],approx[g][0][y+1]),5,(255,0,239),-2) #color pink
                    #cv2.circle(imgMatch,(approx[2][0][y],approx[2][0][y+1]),5,(0,0,255),-2)
                    
                    myCounter +=1

                    if (myCounter == 4):
                        Num ="One"
                        color = (0,225,225) # yellow
                    
                    elif (myCounter == 8):
                        Num = "Zero"
                        color = (0,225,0) # green

                    elif (myCounter == 6):
                        Num = "Seven"
                        color = (120,0,225) #bright pink

                    elif (myCounter ==10): 
                        Num = "Four"
                        color =(0,137,225) #orange

                    elif(myCounter == 12):
                        if np.all(hierarchy == Group1):
                            #Convex hull & convexity defects of the hull
                            hull = cv2.convexHull(cnt,returnPoints = False)
                            defects = cv2.convexityDefects(cnt, hull)

                            for i in range (defects.shape[0]):
                                s,e,f,d = defects[i][0]
                                start = tuple(cnt[s][0])
                                end = tuple(cnt[e][0])
                                far = tuple(cnt[f][0])
                                cv2.circle(imgCon,far,5,(0,255,255),-1) #yellow

                                if (far[0] > int((x+w)/2)) and (y < far[1] < (y+h)):
                                    Num = "Three"
                                else:
                                    if(approx[2][0][y]> int((x+w)/2)) and (y < approx[2][0][y+1] < int((y+h)/2)):
                                        Num = "Two"
                                    else:
                                        Num = "Five"
                            
                        if np.all(hierarchy == Group2):
                            
                            #convex hull & convexity defects of the hull again
                            hull = cv2.convexHull(cnt,returnPoints = False)
                            defects = cv2.convexityDefects(cnt,hull)

                            for i in range(defects.shape[0]):
                                s,e,f,d = defects[i][0]
                                start = tuple(cnt[s][0])
                                end = tuple(cnt[e][0])
                                far = tuple(cnt[f][0])
                                cv2.circle(imgCon,far,5,(0,255,255),-1)
                                
                                if(np.all(far > halfline_horizonal)):
                                    Num = "Six"
                                else:
                                    Num = "Nine"

                        if np.all(hierarchy == Group3):
                            Num = "Eight"

                        color = (255,0,85) # another blue

                    else:
                        Num = "Unknown"
                        color =(125,125,125)

            cv2.putText(imgMatch,Num,(20,20),cv2.FONT_HERSHEY_COMPLEX,1,color,2)

    return imgCon, imgMatch

while True:
    _, frame = cap.read()
    imgResult = frame.copy()
    #imgResult = cv2.GaussianBlur(imgResult,(5,5),1)

    # convert BGR to HSV color space
    hsv = cv2.cvtColor(imgResult, cv2.COLOR_BGR2HSV)

    # get values of hsv from trackbar
    H_min = cv2.getTrackbarPos("H_min","HSV value")
    H_max = cv2.getTrackbarPos("H_max","HSV value")
    S_min = cv2.getTrackbarPos("S_min","HSV value")
    S_max = cv2.getTrackbarPos("S_max","HSV value")
    V_min = cv2.getTrackbarPos("V_min","HSV value")
    V_max = cv2.getTrackbarPos("V_max","HSV value")

    # set up the parameters: lower and upper
    lower = np.array([H_min,S_min,V_min])
    upper = np.array([H_max,S_max,V_max])
    mask = cv2.inRange(hsv,lower, upper)

    # color filter
    imgColorFilter = cv2.bitwise_and(hsv, hsv, mask= mask)
    ret, imgMask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)

    imgCropped = imgMask[cropVals[1]:cropVals[2] + cropVals[1],cropVals[0]:cropVals[0]+cropVals[3]] #resize the image
    imgResult = imgResult[cropVals[1]:cropVals[2] + cropVals[1], cropVals[0]:cropVals[0]+cropVals[3]] #resize the image

    # reduce noises from the image
    imgOpen = cv2.morphologyEx(imgCropped, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    imgClosed = cv2.morphologyEx(imgOpen, cv2.MORPH_CLOSE,np.ones((10,10),np.uint8))
    imgFilter = cv2.bilateralFilter(imgClosed,5,75,75)

    # get contour
    imgContour, imgResult = getContours(imgFilter,imgResult)

    # display on the screen
    cv2.imshow("imgContour",imgContour)
    cv2.imshow("camera", imgResult)
    if (cv2.waitKey(1) == 27):
        break

cap.release()
