import cv2
import time
import numpy as np
import HandTracking as hm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)
detector = hm.handDetector(detectionCon=0.7)




from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]



while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList)!=0:
        x1,y1= lmList[4][1],lmList[4][2]
        x2,y2= lmList[8][1],lmList[8][2]
        cx,cy =(x1+x2) //2, (y1+y2) //2

        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        #cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1,y1) , (x2,y2), (255, 0 ,255), 3)

        length = math.hypot(x2-x1,y2-y1)

        vol = np.interp(length,[20,180], [minVol,maxVol])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)
        if length<30:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Image", img)
    cv2.waitKey(1)