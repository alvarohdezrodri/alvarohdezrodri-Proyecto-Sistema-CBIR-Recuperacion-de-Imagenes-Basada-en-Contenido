import os
import numpy as np
from PIL import Image
import cv2
from skimage import feature
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import faiss

# == PREPROCESAMIENTO ==

'''

El preporcesamiento que realizaremos en las imágenes será el siguiente:
1. Normalizar los valores de píxeles de la imagen a [0, 1].
2. Redimensionar la imagen al tamaño especificado.
3. Eliminar el ruido de la imagen usando un filtro gaussiano.

'''
def normalize_image(img):
    return img / 255.0

def resize_image(img, size=(224, 224)):
    return img.resize(size)

def denoise_image(img):
    img_cv = np.array(img)  
    denoised_img = cv2.GaussianBlur(img_cv, (5, 5), 0)  
    return Image.fromarray(denoised_img)

def preprocess_image(img, size=(224, 224)):
    img = resize_image(img, size=size)
    img = denoise_image(img)
    img_np = normalize_image(np.array(img))
    return Image.fromarray((img_np * 255).astype(np.uint8))  

# == EXTRACTORES DE CARACTERÍSTICAS ==

'''

En este caso, vamos a definir cuatro extractores de características diferentes:
1. Histograma de color en el espacio HSV.
2. Características LBP (Local Binary Patterns).
3. Características extraídas de la red VGG16 preentrenada.
4. Características SIFT (Scale-Invariant Feature Transform).

'''

def extract_color_histogram(img, bins=(8, 8, 8)):
    img = preprocess_image(img)  
    img_hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_lbp(img, num_points=24, radius=8):
    img = preprocess_image(img)  
    img_gray = np.array(img.convert('L'))  
    lbp = feature.local_binary_pattern(img_gray, num_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_cnn_features(img):
    img = preprocess_image(img, size=(224, 224))  
    model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

def extract_sift_features(img):
    img = preprocess_image(img)  
    img_gray = np.array(img.convert('L'))  
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(img_gray, None)
    if descriptors is not None:
        return np.mean(descriptors, axis=0) 
    else:
        return np.zeros(128)

# == CREACIÓN Y GESTIÓN DE ÍNDICES FAISS ==

def create_faiss_index(features, save_path):
    '''
    Crearemos un índice FAISS a partir de características y lo guardaremos en un archivo.

    '''
    features_np = np.vstack(features).astype(np.float32)  
    d = features_np.shape[1]  
    index = faiss.IndexFlatL2(d)  # Creamos el índice de distancia L2
    faiss.normalize_L2(features_np)  
    index.add(features_np)  
    faiss.write_index(index, save_path)  

def load_faiss_index(load_path):
    '''
    Cargamos el índice FAISS desde un archivo.

    '''
    return faiss.read_index(load_path)

def search_faiss_index(index, query_vector, k=10):
    '''
    Realizamos una búsqueda en el índice FAISS con un vector de consulta.

    '''
    faiss.normalize_L2(query_vector)  
    distances, indices = index.search(query_vector, k)
    return distances, indices

# == PROCESAMIENTO Y EXTRACCIÓN DE CARACTERÍSTICAS ==

def preprocess_and_extract_features(image_path, extractor_type):

    img = Image.open(image_path).convert('RGB')  
    img = resize_image(img)  
    img = denoise_image(img)  
    
    if extractor_type == 'color_histogram':
        return extract_color_histogram(img)
    elif extractor_type == 'lbp':
        return extract_lbp(img)
    elif extractor_type == 'cnn':
        return extract_cnn_features(img)
    elif extractor_type == 'sift':
        return extract_sift_features(img)
    else:
        raise ValueError(f"Extractor '{extractor_type}' no reconocido.")



def process_images_in_directory(data_dir, extractor_type):

    features = []
    for sport_folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, sport_folder)
        if os.path.isdir(folder_path):
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)
                feature_vector = preprocess_and_extract_features(image_path, extractor_type)
                if feature_vector is not None:
                    features.append(feature_vector)
    return features

def create_indices(data_dir, extractors, index_save_dir):
    '''
    Procesamos todas las imágenes en un directorio y creamos un índice FAISS para cada extractor.
    Llamamos a las funciones anteriores para realizar el procesamiento y la creación de índices.

    '''
    os.makedirs(index_save_dir, exist_ok=True)
    for extractor_type in extractors:
        print(f"Procesando imágenes con extractor: {extractor_type}")
        index_save_path = os.path.join(index_save_dir, f'{extractor_type}_index.index')  # Índice específico

        features = process_images_in_directory(data_dir, extractor_type)
        create_faiss_index(features, index_save_path)

        print(f"Índice FAISS para '{extractor_type}' guardado en: {index_save_path}")

if __name__ == '__main__':
    # Configuraciones iniciales
    data_dir = 'imagenes/train'  # Directorio donde están las imágenes de entrenamiento
    extractors = ['color_histogram', 'lbp', 'cnn', 'sift']  # Tipos de extractores
    index_save_dir = 'database/'  # Directorio donde guardar los índices FAISS

    print("== INICIO DEL PROCESAMIENTO ==")
    create_indices(data_dir, extractors, index_save_dir)
    print("== PROCESAMIENTO COMPLETADO PARA TODOS LOS EXTRACTORES ==")
