import time
import torch
import faiss
import pathlib
from PIL import Image
import numpy as np
import pandas as pd
import os
import time

import streamlit as st
from streamlit_cropper import st_cropper
from feature_extractors import preprocess_and_extract_features, load_faiss_index, resize_image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(layout="wide")

device = torch.device('cpu')

FILES_PATH = str(pathlib.Path().resolve())

IMAGES_PATH = os.path.join(FILES_PATH, 'imagenes/train') # Ruta de las imágenes de entrenamiento
DB_PATH = os.path.join(FILES_PATH, 'database') # Ruta de la base de datos

DB_FILE = 'db.csv'  # Nombre de la base de datos
LABELS_FILE = 'labels.csv'  # Nombre de la base de datos de etiquetas

EXTRACTORS = {
    'Color Histogram': 'color_histogram_index.index',
    'LBP': 'lbp_index.index',
    'CNN': 'cnn_index.index',
    'SIFT': 'sift_index.index',
}

def get_image_list():
    db_file_path = os.path.join(DB_PATH, DB_FILE)
    if not os.path.exists(db_file_path):
        st.error(f"El archivo de la base de datos (CSV) no existe en: {db_file_path}")
        return []
    df = pd.read_csv(db_file_path)
    return list(df.image.values)

def calculate_metrics(retrieved_indices, true_label, labels_df):
    """
    Calcula las métricas de evaluación: Recall, F1-Score y Accuracy.

    """
    retrieved_labels = labels_df.iloc[retrieved_indices].label.values
    true_positives = np.sum(retrieved_labels == true_label)
    false_positives = np.sum(retrieved_labels != true_label)
    false_negatives = np.sum((labels_df.label.values == true_label) & (~np.isin(labels_df.index, retrieved_indices)))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = true_positives / len(retrieved_indices) if len(retrieved_indices) > 0 else 0

    return {
        'Recall': recall,
        'F1-Score': f1_score,
        'Accuracy': accuracy
    }

def retrieve_image(img_query, feature_extractor, n_imgs=11):
    """
    Recupera imágenes similares basándose en las características extraídas y el índice FAISS.
    
    """
    # Selección del extractor y del índice
    if feature_extractor == 'Color Histogram':
        from feature_extractors import extract_color_histogram, resize_image  
        model_feature_extractor = extract_color_histogram
        indexer = faiss.read_index(os.path.join(DB_PATH, 'color_histogram_index.index'))
    elif feature_extractor == 'LBP':
        from feature_extractors import extract_lbp, resize_image  
        model_feature_extractor = extract_lbp
        indexer = faiss.read_index(os.path.join(DB_PATH, 'lbp_index.index'))
    elif feature_extractor == 'CNN':
        from feature_extractors import extract_cnn_features, resize_image  
        model_feature_extractor = extract_cnn_features
        indexer = faiss.read_index(os.path.join(DB_PATH, 'cnn_index.index'))
    elif feature_extractor == 'SIFT':
        from feature_extractors import extract_sift_features, resize_image  
        model_feature_extractor = extract_sift_features
        indexer = faiss.read_index(os.path.join(DB_PATH, 'sift_index.index'))
    else:
        raise ValueError(f"Extractor '{feature_extractor}' no reconocido.")

    
    img_query = resize_image(img_query)  
    embeddings = model_feature_extractor(img_query)


    vector = np.array(embeddings, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vector)

    _, indices = indexer.search(vector, k=n_imgs)

    return indices[0]



def main():

    '''
    Ajustamos la inerfaz para que el usuario pueda seleccionar el extractor de características,
    la categoría de la imagen de prueba y subir una imagen de prueba.
    Posteriormente, mostramos los resultados de la búsqueda de imágenes similares y las métricas de evaluación.
    
    '''
    st.title('CBIR IMAGE SEARCH')
    
    col1, col2 = st.columns(2)

    with col1:

        st.header('QUERY')

        st.subheader('Choose feature extractor')
        option = st.selectbox('.', list(EXTRACTORS.keys()))

        st.subheader('Select Test Image Category')
        labels_df = pd.read_csv('database\labels.csv')
        categories = labels_df['label'].unique()
        selected_category = st.selectbox('Select Category', categories)

        st.subheader('Upload image')
        img_file = st.file_uploader(label='.', type=['png', 'jpg'])

        if img_file:
            img = Image.open(img_file)
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004', aspect_ratio=None)
            
            st.write("Preview")
            _ = cropped_img.thumbnail((200, 200))  
            st.image(cropped_img)

    with col2:
        st.header('RESULT')
        if img_file:
            st.markdown('**Retrieving .......**')
            start = time.time()

            retriev = retrieve_image(cropped_img, option, n_imgs=11)
            image_list = get_image_list()

            end = time.time()
            st.markdown('**Finished in ' + str(round(end - start, 2)) + ' seconds**')

            st.subheader('Evaluation Metrics')
            metrics = calculate_metrics(retriev, selected_category, labels_df)
            st.write(metrics)

            col3, col4 = st.columns(2)

            with col3:
                image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[0]]))
                image.thumbnail((200, 200))  # Redimensionar a 200x200 píxeles
                st.image(image, use_container_width=False, caption="Result 1")

            with col4:
                image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[1]]))
                image.thumbnail((200, 200))  # Redimensionar a 200x200 píxeles
                st.image(image, use_container_width=False, caption="Result 2")

            col5, col6, col7 = st.columns(3)

            with col5:
                for u in range(2, 11, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]]))
                    image.thumbnail((200, 200))  # Redimensionar a 200x200 píxeles
                    st.image(image, use_container_width=False)

            with col6:
                for u in range(3, 11, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]]))
                    image.thumbnail((200, 200))  # Redimensionar a 200x200 píxeles
                    st.image(image, use_container_width=False)

            with col7:
                for u in range(4, 11, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]]))
                    image.thumbnail((200, 200))  # Redimensionar a 200x200 píxeles
                    st.image(image, use_container_width=False)

if __name__ == '__main__':
    main()

