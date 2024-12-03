# Proyecto-Sistema-CBIR-Recuperacion-de-Imagenes-Basada-en-Contenido
Sistema CBIR para recuperación de imágenes basada en contenido, que utiliza extractores como Color Histogram, LBP, SIFT y CNN. Incluye preprocesamiento, indexación con FAISS, evaluación mediante Recall, F1-Score y Accuracy, y una interfaz en Streamlit para consultas y visualización interactiva de resultados. 

Este proyecto está organizado en una estructura de carpetas y archivos que facilita el desarrollo y la ejecución del sistema CBIR. A continuación, se detalla el contenido de cada archivo y carpeta. Además, a través del siguiente enlace, podrás encontrar las imágenes utilizadas en el proyecto: https://drive.google.com/drive/folders/1_P_OsKDcX9J_3mzeOnKlQIuTlg47lvNC?usp=sharing

### Autores: David Hernando González y Álvaro Hernández Rodríguez
---

## **Estructura del Proyecto**

### **Carpeta principal: `cbir/`**
Esta carpeta contiene todo el proyecto relacionado con el sistema CBIR.

- **`app.py`**
  - Archivo principal que ejecuta la aplicación interactiva en Streamlit.
  - Permite cargar imágenes de consulta, seleccionar extractores de características y visualizar los resultados de recuperación.
  - Incluye funcionalidades para calcular métricas como Recall, F1-Score y Accuracy.

- **`feature_extractors.py`**
  - Contiene la implementación de los métodos de extracción de características:
    - **Color Histogram**
    - **LBP**
    - **SIFT**
    - **CNN**
  - Incluye funciones de preprocesamiento y normalización de imágenes.
  - Gestiona la creación de índices FAISS para la búsqueda eficiente.

- **`requirements.txt`**
  - Archivo que lista las dependencias necesarias para ejecutar el proyecto.
  - Incluye bibliotecas como `streamlit`, `faiss`, `tensorflow`, entre otras.

- **`README.md`**
  - Archivo de documentación del proyecto, que incluye una introducción, instrucciones de instalación y descripción de los componentes del sistema.

---

### **Carpeta `database/`**
Contiene los índices y datos necesarios para la recuperación de imágenes.

- **`cnn_index.index`**
  - Índice FAISS entrenado para características extraídas con CNN.

- **`color_histogram_index.index`**
  - Índice FAISS para características basadas en histogramas de color.

- **`lbp_index.index`**
  - Índice FAISS para características extraídas con LBP.

- **`sift_index.index`**
  - Índice FAISS para características basadas en SIFT.

- **`db.csv`**
  - Archivo CSV que contiene las rutas de todas las imágenes en el dataset.

- **`labels.csv`**
  - Archivo CSV con las rutas y etiquetas asociadas a cada imagen del dataset.

---

### **Carpeta `imagenes/`**
Estructura que almacena las imágenes utilizadas en el proyecto.

- **`train/`**
  - Contiene las imágenes utilizadas para crear los índices.
  - Organizadas en subcarpetas por categoría (por ejemplo, baloncesto, hockey, etc.).

- **`test/`**
  - Imágenes de prueba utilizadas para evaluar el sistema.
  - Permite calcular métricas como Recall, F1-Score y Accuracy.


## **Creación del Entorno**

Para configurar el entorno necesario para ejecutar el proyecto, sigue estos pasos:

1. **Crea un entorno virtual** (opcional pero recomendado):

    ```bash
    python -m venv cbir
    ```
    
    - En **Windows**:
      ```bash
      cbir\Scripts\activate
      ```
    
    - En **macOS/Linux**:
      ```bash
      source cbir/bin/activate  
      ```

2. **Instala las dependencias**:

    ```bash
    pip install -r requirements.txt
    ```

---

## **Ejecución**

Para ejecutar la aplicación, utiliza el siguiente comando:

```bash
streamlit run app.py

---

