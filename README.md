# Trabajo Práctico de Procesamiento de Imágenes I

## Objetivo del Trabajo Práctico
### Integrantes:

- Farias, Augusto
- Lorenzetti, Guido
- Pozzo, Micaela
- Vercesi, Patricio

## Objetivo del Trabajo Práctico

### Problema 1 – Detección y Clasificación de Monedas y Dados
Puedes ejecutar parte por parte del archivo `ej1.py` en la terminal python3 para observasr el procedimiento y resultados del ejercicio.

La imagen `monedas.jpg`, adquirida con un smartphone, consiste en monedas de distintos valores y tamaños, así como dados sobre un fondo de intensidad no uniforme (ver Figura 1).

#### a) Segmentación Automática de Monedas y Dados

El script segmenta dados y monedas distinguiendolos en distintos colores, de manera automática

#### b) Clasificación Automática de Monedas

El código también realiza la clasificación de distintos tipos de monedas y realiza un conteo automático. Se muestra en la imágen original.

#### c) Determinación Automática de Números en Dados

El mismo script determina el número presente en cada dado. Es decir qué cara del dado observamos

### Problema 2 – Detección de Patentes
Puedes ejecutar parte por parte del archivo `ej2.py` en la terminal python3 para observasr el procedimiento y resultados del ejercicio.

La carpeta `Patentes` contiene imágenes de la vista anterior o posterior de diversos vehículos donde se visualizan las correspondientes patentes.

#### a) Detección Automática de Patentes

Este script implementa un algoritmo para detectar automáticamente las patentes y segmentarlas. Informa las distintas etapas de procesamiento y muestra los resultados de cada etapa.

#### b) Segmentación de Caracteres en la Patente

El mismo script implementa un algoritmo para segmentar los caracteres de la patente detectada en el punto anterior. También informa las distintas etapas de procesamiento y muestra los resultados de cada etapa.

## Requisitos
- Python 3.10
- Bibliotecas de python (puedes instalarlas usando pip install -r requirements.txt)