import cv2
import numpy as np
import HandTracking as hm
import time
import autopy

##########################
wCam, hCam = 640, 480
frameR = 100 # frame Reduction for smoohening
smoothening = 7
##########################

pTime=0
plocX,plocY = 0,0
clocX,clocY = 0,0

cap = cv2.VideoCapture(0)

detector = hm.handDetector(maxHands=1)
wScr , hScr =autopy.screen.size()
print(wScr,hScr)
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # get the tip of index and middle finger
    if len(lmList)!=0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

    # Check fingers up
    fingers = detector.fingersUp()

    cv2.rectangle(img, (frameR, frameR), (wCam - frameR,hCam - frameR), (255, 0, 255), 2)
    # Only Index FINGER : MOVING
    if fingers[1]==1 and fingers[2]==0:

        #Converting Coordinates to SCreen Size
        x3 = np.interp(x1, (frameR,wCam - frameR), (0,wScr))
        y3 = np.interp(y1, (frameR,hCam - frameR), (0,hScr))

        #SMoothening
        clocX = plocX + (x3 - plocX) /smoothening
        clocY = plocY + (y3 - plocY) /smoothening

        autopy.mouse.move(wScr-clocX,clocY)
        cv2.circle(img,(x1,y1), 15, (255,0,233), cv2.FILLED)
        plocX ,plocY = clocX ,clocY
    # CLICKING PERIOD
    if fingers[1] == 1 and fingers[2] == 1:
        length,img, Lineinfo = detector.findDistance(8,12, img)
        print(length)
        # Find distance between fingers
        if length < 30:
            cv2.circle(img,(Lineinfo[4], Lineinfo[5]), 15, (0,255,0), cv2.FILLED)
            autopy.mouse.click()


    # Frame Rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN,3, (255,0,0), 3)

    # DIsplay
    cv2.imshow("Image", img)
    cv2.waitKey(1)