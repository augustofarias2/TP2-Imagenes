import cv2
import numpy as np
import matplotlib.pyplot as plt

# Leer la imagen en color
img= cv2.imread("monedas.jpg")
img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_color), plt.show(block = False)

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
plt.imshow(img_filtered, cmap = "gray")
plt.title('Imagen filtrada - 0.1')
plt.show()

#Canny
img_canny_CV2 = cv2.Canny(img_filtered, 30, 100)
plt.imshow(img_canny_CV2, cmap="gray"), plt.show(block = False)

#Dilato
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
img_dilated = cv2.dilate(img_canny_CV2, kernel, iterations=1)
plt.imshow(img_dilated, cmap = "gray"), plt.show(block=False)


#Obtener estructura
img_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18)))
img = cv2.morphologyEx(img_dilated, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
plt.imshow(img_close, cmap="gray")
plt.show(block= False)

plt.figure(figsize=(16, 9))
plt.subplot(221)
plt.imshow(cv2.morphologyEx(img_close, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))), cmap= "gray")
plt.subplot(222)
plt.imshow(cv2.morphologyEx(img_close, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))), cmap = "gray")
plt.subplot(223)
plt.imshow(cv2.morphologyEx(img_close, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))), cmap = "gray")
plt.subplot(224)
plt.imshow(cv2.morphologyEx(img_close, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))), cmap = "gray")
plt.show()

# ---------------------------------------------------------------------------------------
# --- Reconstrucción Morgológica --------------------------------------------------------
# ---------------------------------------------------------------------------------------
def imreconstruct(marker, mask, kernel=None):
    if kernel==None:
        kernel = np.ones((3,3), np.uint8)
    while True:
        expanded = cv2.dilate(marker, kernel)                               # Dilatacion
        expanded_intersection = cv2.bitwise_and(src1=expanded, src2=mask)   # Interseccion
        if (marker == expanded_intersection).all():                         # Finalizacion?
            break                                                           #
        marker = expanded_intersection        
    return expanded_intersection

def imfillhole_v2(img):
    img_flood_fill = img.copy().astype("uint8")             # Genero la imagen de salida
    h, w = img.shape[:2]                                    # Genero una máscara necesaria para cv2.floodFill()
    mask = np.zeros((h+2, w+2), np.uint8)                   # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#floodfill
    cv2.floodFill(img_flood_fill, mask, (0,0), 255)         # Relleno o inundo la imagen.
    img_flood_fill_inv = cv2.bitwise_not(img_flood_fill)    # Tomo el complemento de la imagen inundada --> Obtenog SOLO los huecos rellenos.
    img_fh = img | img_flood_fill_inv                       # La salida es un OR entre la imagen original y los huecos rellenos.
    return img_fh 

img_fh_v2 = imfillhole_v2(img)
plt.imshow(img_fh_v2, cmap = "gray"), plt.show(block=False)


#Componentes conectadas para ver areas
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_fh_v2, 4, cv2.CV_32S)


# --- Defino parametros para la clasificación -------------------------------------------
RHO_TH = 0.8    # Factor de forma (rho)
AREA_TH = 5000   # Umbral de area
aux = np.zeros_like(labels)
labeled_image = cv2.merge([aux, aux, aux])

# --- Clasificación ---------------------------------------------------------------------
# Clasifico en base al factor de forma
for i in range(1, num_labels):

    # --- Remuevo celulas con area chica --------------------------------------
    if (stats[i, cv2.CC_STAT_AREA] < AREA_TH):
        continue

    # --- Selecciono el objeto actual -----------------------------------------
    obj = (labels == i).astype(np.uint8)

    # --- Calculo Rho ---------------------------------------------------------
    ext_contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(ext_contours[0])
    perimeter = cv2.arcLength(ext_contours[0], True)
    rho = 4 * np.pi * area/(perimeter**2)
    flag_circular = rho > RHO_TH

    # --- Calculo cantidad de huecos ------------------------------------------
    all_contours, _ = cv2.findContours(obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    holes = len(all_contours) - 1

    # --- Clasifico -----------------------------------------------------------
    if rho > RHO_TH:
        labeled_image[obj == 1, 0] = 255
    else:
        labeled_image[obj == 1, 1] = 255    # Circular con mas de 1 hueco
    
plt.figure(); plt.imshow(labeled_image); plt.show(block=False)




























umbral_moneda = 1000
# Encontrar contornos en la imagen después de la morfología
contours, _ = cv2.findContours(img_fh_v2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Crear una copia de la imagen original para dibujar contornos y etiquetas de color
img_labeled = img_color.copy()

# Iterar sobre los contornos y dibujar etiquetas de color
for contour in contours:
    area = cv2.contourArea(contour)
    if area > umbral_moneda:  # Puedes ajustar este umbral según tus necesidades
        cv2.drawContours(img_labeled, [contour], 0, (0, 0, 255), 2)  # Dibujar contorno en rojo
    # else:
    #     cv2.drawContours(img_labeled, [contour], 0, (255, 0, 0), 2)  # Dibujar contorno en azul

# Mostrar la imagen con contornos y etiquetas de color
plt.imshow(img_labeled)
plt.title('Objetos detectados')
plt.show()


#Opcion 2
# se = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# f_mg = cv2.morphologyEx(Fd, cv2.MORPH_GRADIENT, se)
# plt.imshow(f_mg, cmap= "gray"), plt.show(block = False)


# _, img_binary = cv2.threshold(img_filtered1_gray, 42, 255, cv2.THRESH_BINARY)
# plt.imshow(img_binary, cmap="gray"), plt.show(block = False)

#Encontrar componentes conectadas
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fop_cls, 4, cv2.CV_32S)

im_color = cv2.applyColorMap(np.uint8(255/num_labels*labels), cv2.COLORMAP_INFERNO)

plt.imshow(im_color); plt.show(block=False)
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
