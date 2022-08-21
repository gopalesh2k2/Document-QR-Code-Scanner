import cv2
import numpy as np
from pyzbar.pyzbar import decode
import streamlit as st


def scanCode(webCam, img = None):

    if webCam == True:
        cap = cv2.VideoCapture(0)
        cap.set(3, 720)
        cap.set(4, 720)
    while True:
        if webCam:
            # img = np.zeros((720,720,3), np.uint8)
            success, img = cap.read()
            print(img)
            print(success)
        for  barcode in decode(img):
            myData = barcode.data.decode('utf-8')
            pts = np.array([barcode.polygon], np.int32)
            pts.reshape((-1,1,2))
            cv2.polylines(img, [pts], True, (255,0,255), 5)
            start = barcode.rect
            cv2.putText(img, myData,(start[0], start[1]) , cv2.FONT_HERSHEY_SIMPLEX, 0.9 , (255,0,25), 4)
        if webCam:
            st.image(img)
            cv2.waitKey(1)
        else:
            break;
    cv2.destroyAllWindows()
    cv2.VideoCapture(1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

