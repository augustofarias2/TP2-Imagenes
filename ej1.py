import cv2
import numpy as np
import matplotlib.pyplot as plt

# Leer la imagen en color
img = cv2.imread("monedas.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img), plt.show(block = False)

def median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

# Aplica el filtro de mediana a cada canal R, G y B por separado
kernel_size = 5  # Tamaño del kernel de mediana

img_filtered_b = median_filter(img, kernel_size)

# Muestra las imágenes originales y las filtradas
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img)
plt.title('Imagen Original')

plt.subplot(122)
plt.imshow(img_filtered_b)
plt.title('Imagen filtrada - 0.1')
plt.show()

img_filtered1_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(img_filtered1_gray, cmap="gray"), plt.show(block = False)

# _, img_binary = cv2.threshold(img_filtered1_gray, 42, 255, cv2.THRESH_BINARY)
# plt.imshow(img_binary, cmap="gray"), plt.show(block = False)

#Encontrar componentes conectadas
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_filtered1_gray, 4, cv2.CV_32S)

im_color = cv2.applyColorMap(np.uint8(255/num_labels*labels), cv2.COLORMAP_INFERNO)
