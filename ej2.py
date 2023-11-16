import numpy as np
import cv2
import matplotlib.pyplot as plt

def detect_pattent(img):
    # Load the image
    img = cv2.imread(img)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 100, 254, cv2.THRESH_BINARY_INV)[1]
    B = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    Aop = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, B)
    # countors = cv2.findContours(Aop, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    connected_components = cv2.connectedComponentsWithStats(Aop, 4, cv2.CV_32S)
    inverted = cv2.bitwise_not(connected_components[1])
    plt.imshow(inverted, cmap='gray')
    
    license_ratio = 0.5
    min_w = 0
    max_w = 10
    min_h = 0
    max_h = 20
    candidates = []
    for cnt in connected_components[1]:
        x,y,w,h = cv2.boundingRect(cnt)
        ratio = float(w)/h
        if (np.isclose(ratio, license_ratio, atol=0.7)):
            if max_w > w >= min_w and max_h > h >= min_h:
                candidates.append(cnt)

    plt.imshow(candidates[0], cmap='gray')

    return thresh

img = 'Patentes/img01.png'
img = detect_pattent(img)