import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
import cv2
import time
import numpy as np
import HandDetectionModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.HandDetector()

# Correct way to get audio interface
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, 0, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get volume range
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
volbar = 400
vol = 0
volper =0
while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lmList = detector.find_position(img, draw=False)
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        
        cv2.circle(img, (x1,y1), 10, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 10, (255,0,255), cv2.FILLED)
        cv2.circle(img, (cx,cy), 10, (255,0,255), cv2.FILLED)
        cv2.line(img, (x1,y1), (x2,y2), (255,0,255), 3)
        
        length = math.hypot(x2-x1, y2-y1)
        
        # Convert hand range to volume range
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volbar = np.interp(length, [50, 300], [400, 150])
        volper = np.interp(length, [50, 300], [0, 100])

        volume.SetMasterVolumeLevel(vol, None)
        print(int(length),vol)
        if length < 50:
            cv2.circle(img, (cx, cy), 10, (0,255,0), cv2.FILLED)

    cv2.rectangle(img, (50,150), (85,400), (255,0,0), 3)
    cv2.rectangle(img, (50,int(volbar)), (85,400), (255,0,0), cv2.FILLED)
    cv2.putText(img, f'{int(volper)}%',(40,450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)