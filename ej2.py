import numpy as np
import cv2
import matplotlib.pyplot as plt

def detect_pattent(img):
    # Load the image
    img = cv2.imread(img)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 80, 253, cv2.THRESH_BINARY_INV)[1]

    canvas = cv2.Canny(thresh, 100, 170)

    B = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    Aop = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, B)

    plt.figure(figsize=(10,5))
    plt.subplot(2,2,1)
    plt.imshow(img)
    plt.subplot(2,2,2)
    plt.imshow(gray, cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(thresh, cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(Aop, cmap='gray')
    plt.show()
    # cv2.imshow('Patente', thresh)
    # cv2.imshow('Patente', canvas)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

detect_pattent('Patentes/img01.png')