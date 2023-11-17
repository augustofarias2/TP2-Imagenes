"""
Implementar un algoritmo de procesamiento que segmente los caracteres de 
la  patente  detectada  en  el  punto  anterior.  Informar  las  distintas  etapas  de 
procesamiento y mostrar los resultados de cada etapa. 
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Defininimos función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

# enderezar ?

# limpiar de ruido?

#segmentar
img = cv2.imread('Patentes\ejemplo_limpio_derecho.png',cv2.IMREAD_GRAYSCALE)
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
imshow(img=img)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity= 8, ltype=cv2.CV_32S)
contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

imshow(img=labels)

# metodo ratio ?