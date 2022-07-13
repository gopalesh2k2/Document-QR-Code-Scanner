import cv2
import numpy as np
import streamlit as st

## TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2))
    # print("old")
    # print(myPoints)
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)

    add = myPoints.sum(1)
    # print(add)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


def biggestContour(contours):
    # print('hey')
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.018 * peri, True) # approximates to a finitie polygon shape depending upon epsilon
            # print(len(approx))
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    print(max_area)
    return biggest,max_area
def drawRectangle(img,biggest,thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 0, 255), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 0, 255), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 0, 255), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 0, 255), thickness)

    return img

def nothing(x):
    pass

def initializeTrackbars(intialTracbarVals=0):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 200,255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 200, 255, nothing)


def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    src = Threshold1,Threshold2
    return src

def autoCanny(image):
    # Finds optimal thresholds based on median image pixel intensity
    blurred_img = cv2.blur(image, ksize=(5,5))
    med_val = np.median(image)
    lower = int(max(0, 0.66 * med_val))
    upper = int(min(255, 1.33 * med_val))
    edges = cv2.Canny(image=image, threshold1=lower, threshold2=upper)
    return edges

def scanImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    heightImg = int(0.5*img.shape[1])
    widthImg = int(0.3*img.shape[0])
    img = cv2.resize(img,(widthImg, heightImg))
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # grayscale image
    imgblur = cv2.GaussianBlur(imggray,(7,7),1)
    # thres = valTrackbars()
    imgthreshold = autoCanny(imgblur)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgthreshold, kernel, iterations=2)
    imgthreshold = cv2.erode(imgDial, kernel, iterations=2)

    # find contours
    imgContours = img.copy() # copy image for display purpose
    imgBigContour = img.copy() # copy image for display purpose
    contours, hierarchy = cv2.findContours(imgthreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0,255,0), 4)

    # find the biggest contour
    biggest, maxarea= biggestContour(contours) # find the biggest contour
    if biggest.size != 0:
        biggest = reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0,0,255),20)
        imgBigContour = drawRectangle(imgBigContour,biggest,2)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0,0], [widthImg, 0], [0, heightImg], [widthImg,heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        # remove 20 pixels from all sides
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0]-20, 20:imgWarpColored.shape[1]-20]
        imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))

        # apply adaptive imgthreshold
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 5)
        # imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre,3)
    else:
        st.write('Unable to process this image please upload a new image')
        imgAdaptiveThre = imgBigContour.copy()
        cv2.putText(imgAdaptiveThre, "ERROR PROCESSING", (int(widthImg/8), int(heightImg/2)), cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,0),5)

    return cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2RGB), cv2.cvtColor(imgBigContour, cv2.COLOR_BGR2RGB)


def download(image):
    cv2.imwrite('output.jpg',image)