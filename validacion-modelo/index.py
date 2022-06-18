#Cargar el label_map.pbtxt
#Cargar el fine_tuned_model.zip

import zipfile
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import tensorflow as tf
import numpy as np

local_zip = "/content/fine_tuned_model.zip"
zip_ref = zipfile.ZipFile(local_zip, "r")
zip_ref.extractall("/content/fine_tuned_model")
zip_ref.close()

## Revisar que la ruta sea la misma de la carpeta.
PATH_TO_MODEL_DIR = "/content/fine_tuned_model/content/fine_tuned_model"
PATH_TO_SAVE_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

#cargamos el modelo
detect_fn = tf.saved_model.load(PATH_TO_SAVE_MODEL)

# Cargamos el label map para utilizarlo.
label_map_pbtxt_fname = "/content/label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(label_map_pbtxt_fname)

from PIL import Image
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Importamos la imagen
image_path = "/content/5.jpg"

# La convertimos a array
image_np = np.array(Image.open(image_path))

# La convertimos a tensor y la agregamos una dimensión para que pueda leerla nuestro modelo
input_tensor = tf.convert_to_tensor(image_np)
input_tensor = input_tensor[tf.newaxis, ...]

# Realizamos la detección del objeto
detections = detect_fn(input_tensor)


# Analizamos cuántas detecciones se obtuvieron
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0,:num_detections].numpy() for key, value in detections.items()}

detections['num_detections'] = num_detections

detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

# Tomamos una imagen y la copiamos para dibujar los bounding box
image_np_with_detections = image_np.copy()

# Utilizamos la libreria de obejct detection para visualizar le bounding box y la clasificación
viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'],
    detections['detection_classes'],
    detections['detection_scores'],
    category_index,
    max_boxes_to_draw=1,
    min_score_thresh=0.30,
    use_normalized_coordinates = True
)


# Visualizamos resultados
cv2_imshow(image_np_with_detections)
print(detections['detection_scores'])