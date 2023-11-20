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

def detect_patent(filename):
    # Read the image
    image = cv2.imread(filename)

    # Convert the image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for white color in HSV format
    lower_white = np.array([0, 0, 120], dtype=np.uint8)
    upper_white = np.array([180, 75, 255], dtype=np.uint8)
    # Create a mask to identify white pixels within the specified HSV range
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    # Bitwise AND operation to extract white pixels
    whitish_pixels = cv2.bitwise_and(image, image, mask=white_mask)

    # Convert the Value channel to grayscale
    gray = whitish_pixels[:, :, 2]
    
    # Laplacian of Gaussian
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    LoG = cv2.Laplacian(blur, cv2.CV_64F, ksize=3)
    LoG_abs = cv2.convertScaleAbs(LoG)   # Pasamos a 8 bit
    LoG_abs_th = LoG_abs > LoG_abs.max()*0.45
    filtered = LoG_abs_th.astype(np.uint8) * 255
    
    # Morphological Opening & Closing
    B = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, B)
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, B)

    num_labels, labels, stats, centroids  = cv2.connectedComponentsWithStats(filtered, 4, cv2.CV_32S)

    possible_patents = []
    for st in stats:
        x, y, w, h, area = st
        ratio = w / h
        if  (1 < ratio < 3) and (65 < w < 110) and (20 < x < 620) and (20 < y < 460):

            highlighted = cv2.rectangle(image.copy(), (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
            possible_patents.append(image[y-5:y+h+5, x-5:x+w+5])

    #cv2.imshow(filename, highlighted)
    #cv2.waitKey(0)
    return possible_patents


def detect_characters(patent, filename):
        # Convert the image from BGR to HSV
        hsv_image = cv2.cvtColor(patent, cv2.COLOR_RGB2HSV)

        # Define lower and upper bounds for white color in HSV format
        lower_white = np.array([0, 0, 120], dtype=np.uint8)
        upper_white = np.array([180, 75, 255], dtype=np.uint8)
        # Create a mask to identify white pixels within the specified HSV range
        white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
        # Bitwise AND operation to extract white pixels
        whitish_pixels = cv2.bitwise_and(patent, patent, mask=white_mask)

        # Convert the Value channel to grayscale
        gray = whitish_pixels[:, :, 2]
        
        num_labels, labels, stats, centroids  = cv2.connectedComponentsWithStats(gray, 4, cv2.CV_32S)

        characters = []
        for st in stats:
            x, y, w, h, area = st
            ratio = w / h
            if (0.3 < ratio < 1) and (10 < h):
                highlighted = cv2.rectangle(patent.copy(), (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
                characters.append(patent[y:y+h, x:x+w])

        #cv2.imshow(filename, highlighted)
        #cv2.waitKey(0)
        return characters



def show_all_characters():
    for i in range (1,13):
        filename = 'Patentes/img'+("0"+str(i) if i<=9 else str(i))+'.png'
        possible_patents = detect_patent(filename)
        if possible_patents == []:
            print('No se detectó posible patente en ' + filename)
            continue
        six_char = False
        for patent in possible_patents:
            characters = detect_characters(patent, filename)
            if len(characters) == 6:
                six_char = True
                break
        if six_char:
            imshow(patent)

            fig, axs = plt.subplots(1, 6, figsize=(4, 1))
            for j in range(len(characters)):
                axs[j].axis('off')
                axs[j].imshow(characters[j])
            plt.suptitle(filename, fontsize=16)
            plt.tight_layout()
            plt.show()
        else:
            print('No se detectaron 6 caracteres en ' + filename)


show_all_characters()
