import cv2
import numpy as np
import matplotlib.pyplot as plt

# Leer la imagen en color
img= cv2.imread("monedas.jpg")
img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img, cmap = 'gray'), plt.show(block = False)

#Imagen filtrada gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(img_gray, cmap="gray"), plt.show(block = False)

kernel_size = 7  # Tamaño del kernel de mediana
img_filtered = cv2.medianBlur(img_gray, kernel_size)

# Muestra las imágenes originales y las filtradas
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img_color)
plt.title('Imagen Original')
plt.subplot(122)
plt.imshow(img_filtered)
plt.title('Imagen filtrada - 0.1')
plt.show()



#Canny
img_canny_CV2 = cv2.Canny(img_filtered, 20, 100)#, apertureSize=3, L2gradient=True)
plt.imshow(img_canny_CV2, cmap="gray"), plt.show(block = False)

#Dilato
# kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
Fd = cv2.dilate(img_canny_CV2, kernel, iterations=1)
plt.imshow(Fd, cmap = "gray"), plt.show(block=False)

#Obtener estructura? No funciona
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
fop = cv2.morphologyEx(Fd, cv2.MORPH_OPEN, se)
fop_cls = cv2.morphologyEx(fop, cv2.MORPH_CLOSE, se)
plt.imshow(fop_cls, cmap="gray")
plt.show(block= False)

#Opcion 2
se = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
f_mg = cv2.morphologyEx(Fd, cv2.MORPH_GRADIENT, se)
plt.imshow(f_mg, cmap= "gray"), plt.show(block = False)


# _, img_binary = cv2.threshold(img_filtered1_gray, 42, 255, cv2.THRESH_BINARY)
# plt.imshow(img_binary, cmap="gray"), plt.show(block = False)

#Encontrar componentes conectadas
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fop_cls, 4, cv2.CV_32S)

im_color = cv2.applyColorMap(np.uint8(255/num_labels*labels), cv2.COLORMAP_INFERNO)


# escala de grises
# filtro pasa bajos
# canny
# dilato
# relleno
# aplico morfologia para mejorar
# luego clasifico en factor de forma
#Los dados los pego en otra, elimino el fondo
#Relleno los circulos, uso formula para eliminar los que no sean circulos
# pego la mascara de color sobre la imagen original para que quede mas lindo
