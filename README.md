# Procesamiento de Imágenes de Mamografías para Clasificación con Redes Neuronales Convolucionales 🩺📊

Este repositorio contiene herramientas y scripts para el procesamiento de imágenes mamográficas, orientado a entrenar modelos de redes neuronales convolucionales (CNN). El objetivo principal es construir un modelo capaz de clasificar estas imágenes de manera precisa, contribuyendo al diagnóstico temprano de condiciones relacionadas con el cáncer de mama.

---

## Objetivos del Proyecto 🎯

- **Procesamiento de imágenes**: Preprocesamiento de imágenes mamográficas para garantizar uniformidad en tamaño, resolución y calidad.
- **Entrenamiento de modelos CNN**: Desarrollo y entrenamiento de redes neuronales convolucionales para la clasificación precisa de las imágenes.
- **Evaluación del modelo**: Medición del desempeño mediante métricas como precisión, sensibilidad, especificidad y AUC (Área Bajo la Curva).
- **Optimización del flujo de trabajo**: Automatización de tareas de procesamiento para manejar grandes volúmenes de datos.

---

## Características principales 🔍

- **Preprocesamiento avanzado**:
  - Normalización y escalado de imágenes.
  - Aumento de datos (data augmentation) para mejorar la generalización.
  - Eliminación de artefactos y ruidos en las imágenes.
  
- **Arquitectura de redes neuronales**:
  - Modelos personalizados de CNN para clasificación.
  - Exploración de arquitecturas preentrenadas como VGG16, ResNet50 e InceptionV3.

- **Pipeline eficiente**:
  - Carga y procesamiento en lotes de imágenes.
  - División de datos en conjuntos de entrenamiento, validación y prueba.
  
- **Soporte para métricas detalladas**:
  - Curvas ROC/AUC.
  - Matrices de confusión.


## Ejemplos de los Resultados:

Imagen original:
![Texto alternativo](https://raw.githubusercontent.com/AndreaCTS/imagenes/main/Screenshot%202025-01-08%20120509.png "Título opcional")

Imagen después de aplicar todo el preprocesamiento paa la eliminación de ruido y mejora del contraste:
![Texto alternativo](https://raw.githubusercontent.com/AndreaCTS/imagenes/main/Screenshot%202025-01-08%20120448.png "Título opcional")

Matriz de resultado de la precisión por cada una de las clases y del modelo en general, se recuerda que el accuracy del modelo es del 82%:
![Texto alternativo](https://raw.githubusercontent.com/AndreaCTS/imagenes/main/Screenshot%202025-01-08%20120539.png "Título opcional")


---

## Requisitos del Proyecto 🛠️

1. **Lenguaje principal**: Python (>=3.8)
2. **Librerías principales**:
   - TensorFlow 
   - NumPy
   - OpenCV
   - Scikit-learn
   - Matplotlib / Seaborn

