# Encontrar contornos en la imagen después de la operación de cierre
contours, _ = cv2.findContours(fop_cls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar los contornos en la imagen original
img_contours = img_color.copy()
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)  # -1: dibujar todos los contornos encontrados

# Clasificar formas basadas en el número de lados
for contour in contours:
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_sides = len(approx)

    
        
    if num_sides == 4:
        # Puedes verificar si es un cuadrado o un rectángulo comparando las longitudes de los lados
        shape = "Cuadrado" if cv2.contourArea(contour) > 100 else "Rectángulo"
    else:
        shape = "Círculo"

   # Muestra la forma clasificada
    print(f"Forma clasificada: {shape}")

    # Dibujar el nombre de la forma en la imagen
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.putText(img_contours, shape, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Mostrar la imagen con los contornos y las etiquetas de las formas
plt.imshow(img_contours), plt.show()



#Se encuentran los contornos en la imagen después de la operación de cierre.
#Se utiliza cv2.approxPolyDP para aproximar los contornos y determinar el número de lados.
#Se clasifican las formas según el número de lados.
#Se dibujan los contornos y se etiquetan las formas encontradas en la imagen original.
#Encontrar componentes conectadas
#num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fop_cls, 4, cv2.CV_32S)
