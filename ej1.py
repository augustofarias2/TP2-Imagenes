import cv2
import numpy as np
import matplotlib.pyplot as plt

# Leer la imagen en color
img= cv2.imread("monedas.jpg")
img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_color), plt.show(block = False)

#_____________________________________________________________EJERCICIO 1________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________________
#Imagen filtrada gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(img_gray, cmap="gray"), plt.show(block = False)

#Filtro pasabajo
img_filtered = cv2.medianBlur(img_gray, 6)
plt.imshow(img_filtered, cmap = "gray"), plt.show(block=False)

#Canny
img_canny_CV2 = cv2.Canny(img_filtered, 35, 110)
plt.imshow(img_canny_CV2, cmap="gray"), plt.show(block = False)

#Dilato
img_dilated = cv2.dilate(img_canny_CV2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12)), iterations=1)
img_dilated = cv2.dilate(img_dilated, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
plt.imshow(img_dilated, cmap = "gray"), plt.show(block=False)


#Obtener estructura
img_open = cv2.morphologyEx(img_dilated, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16)))
plt.imshow(img_close, cmap="gray"), plt.show(block= False)


# ---------------------------------------------------------------------------------------
# --- Reconstrucción Morgológica --------------------------------------------------------
# ---------------------------------------------------------------------------------------
def imreconstruct(marker, mask, kernel=None):
    if kernel==None:
        kernel = np.ones((3,3), np.uint8)
    while True:
        expanded = cv2.dilate(marker, kernel)                               # Dilatacion
        expanded_intersection = cv2.bitwise_and(src1=expanded, src2=mask)   # Interseccion
        if (marker == expanded_intersection).all():                        
            break                                                           
        marker = expanded_intersection        
    return expanded_intersection

def imfillhole_v2(img):
    img_flood_fill = img.copy().astype("uint8")             # Genero la imagen de salida
    h, w = img.shape[:2]                                    # Genero una máscara necesaria para cv2.floodFill()
    mask = np.zeros((h+2, w+2), np.uint8)                   
    cv2.floodFill(img_flood_fill, mask, (0,0), 255)         # Relleno o inundo la imagen.
    img_flood_fill_inv = cv2.bitwise_not(img_flood_fill)    # Tomo el complemento de la imagen inundada --> Obtenog SOLO los huecos rellenos.
    img_fh = img | img_flood_fill_inv                       # La salida es un OR entre la imagen original y los huecos rellenos.
    return img_fh 

img_fh = imfillhole_v2(img_close)
plt.imshow(img_fh, cmap = "gray"), plt.show(block=False)


#Componentes conectadas para ver areas
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_fh, 4, cv2.CV_32S)


# --- Defino parametros para la clasificación -------------------------------------------
RHO_TH = 0.8	# Factor de forma (rho), si es circulo el valor es mayor a 0.8
AREA_TH = 5000   # Umbral de area para descartar los labels que no sean figuras
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
        labeled_image[obj == 1, 2] = 255
    else:
        labeled_image[obj == 1, 1] = 255
    
plt.figure(); plt.imshow(labeled_image); plt.show(block=False)

#_____________________________________________________________EJERCICIO 2________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________________
# Convertir a profundidad de 8 bits por canal
monedas_8u = labeled_image[:, :, 2].astype(np.uint8)

# Convertir a escala de grises
monedas_gray = cv2.cvtColor(monedas_8u, cv2.COLOR_BGR2GRAY)

# Aplicar umbral (opcional, dependiendo de tus necesidades)
_, monedas_binary = cv2.threshold(monedas_gray, 15, 255, cv2.THRESH_BINARY)
plt.imshow(monedas_binary, cmap= "gray"), plt.show(block=False)

# Supongamos que 'stats' es tu matriz de estadísticas y 'original_image' es tu imagen original
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(monedas_binary, 4, cv2.CV_32S)

# Definir los rangos de áreas para cada tipo de moneda
area_rangos = {
    '10_cent': (0, 85000),
    '1_peso': (85000, 105000),
    '50_cent': (105000, 300000)
}

# Crear una lista para almacenar el tipo de cada moneda
tipos_monedas = []
cant_monedas = {"10 centavos": 0,"50 centavos": 0, "1 peso": 0}
# Clasificar cada moneda según su área
for i in range(1, stats.shape[0]):
    area = stats[i, cv2.CC_STAT_AREA]
    
    # Comparar el área con los rangos definidos
    if area_rangos['10_cent'][0] <= area <= area_rangos['10_cent'][1]:
        tipo = '10 CENT'
        cant_monedas["10 centavos"] +=1

    elif area_rangos['1_peso'][0] <= area <= area_rangos['1_peso'][1]:
        tipo = '1 PESO'
        cant_monedas["1 peso"] +=1
    elif area_rangos['50_cent'][0] <= area <= area_rangos['50_cent'][1]:
        tipo = '50 CENT'
        cant_monedas["50 centavos"] +=1
    else:
        tipo = 'No Clasificado'
    
    tipos_monedas.append([tipo, i])

# Crear una copia de la imagen original para agregar etiquetas
etiquetas_image = img_color.copy()

# Configurar la fuente y otros parámetros del texto
font = cv2.FONT_HERSHEY_SIMPLEX
escala = 2.5
color = (0, 0, 0)  # Color del texto en blanco

# Agregar texto para cada tipo de moneda en la posición central
for tipo, i in tipos_monedas:
    centro = (int(centroids[i, 0]), int(centroids[i, 1]))
    cv2.putText(etiquetas_image, tipo, centro, font, escala, color, 2, cv2.LINE_AA)

# Agregar texto con la cantidad de monedas por tipo
posicion_y = 2300
for tipo, cantidad in cant_monedas.items():
    texto_cantidad = f'{tipo}: {cantidad} monedas'
    cv2.putText(etiquetas_image, texto_cantidad, (30, posicion_y), font, 3, color, 2, cv2.LINE_AA)
    posicion_y += 90  # Ajusta el espacio vertical entre líneas según tus preferencias

#Agregar borde a la moneda
# Encontrar contornos en la imagen después de la morfología
contours, _ = cv2.findContours(monedas_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterar sobre los contornos y dibujar etiquetas de color
for contour in contours:
    area = cv2.contourArea(contour)
    cv2.drawContours(etiquetas_image, [contour], 0, (0, 0, 255), 2)  # Dibujar contorno en rojo
   

# Mostrar la imagen con texto
plt.imshow(etiquetas_image), plt.show(block=False)

#_____________________________________________________________EJERCICIO 3________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________________
# Encontrar las coordenadas de los píxeles verdes en la imagen con dados
coordenadas_verdes = cv2.findNonZero(labeled_image[:, :, 1])

# Crear una máscara en blanco del mismo tamaño que la imagen filtrada en gray scale
img_dados = np.zeros_like(img_filtered, dtype = np.uint8)

# Dibujar regiones segmentadas en la máscara usando las coordenadas verdes
for coord in coordenadas_verdes:
    x, y = coord[0]
    img_dados[y, x] = img_filtered[y, x]

# Mostrar la máscara con las regiones segmentadas
plt.imshow(img_dados, cmap = "gray"), plt.show(block=False)


#Canny
img_canny_dados = cv2.Canny(img_dados, 35, 110)
plt.imshow(img_canny_dados, cmap="gray"), plt.show(block = False)

#Dilato
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
# img_dilated = cv2.dilate(img_canny_CV2, kernel, iterations=1)
img_dilated_dado = cv2.dilate(img_canny_dados, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
plt.imshow(img_dilated_dado, cmap = "gray"), plt.show(block=False)


#Obtener estructura
img_open_dado = cv2.morphologyEx(img_dilated_dado, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
img_close_dado = cv2.morphologyEx(img_open_dado, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
plt.imshow(img_close_dado, cmap="gray")
plt.show(block= False)



# ________________________________________________________________________________________________________________________________________________________________________












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
