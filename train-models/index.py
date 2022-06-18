import os
import pandas as pd
import json
import pickle
import zipfile
import shutil

# Instalamos los paquetes necesarios para que funcione desde la Colab
# !pip install avro-python3
# !pip install 
# !pip install tf_slim==1.1.0
# !pip install tf-models-official==2.7.0
# !pip install lvis
# !pip install tensorflow_io==0.23.1
# !pip install keras==2.7.0
# !pip install opencv-python-headless==4.5.2.52

labels = [
    { 'name': 'VehÃ­culo ambulancia', 'id': 1},
    { 'name': 'Logo ambulancia', 'id': 2},
    { 'name': 'Texto ambulancia', 'id': 3},
    { 'name': 'VehÃ­culo bomberos', 'id': 4},
    { 'name': 'Logo bomberos', 'id': 5},
    { 'name': 'Texto bomberos', 'id': 6},
]

with open("label_map.pbtxt", "w") as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

output_path = 'models/research/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
output_path = os.path.join(os.getcwd(), output_path)
print ("La carpeta se almaceno en {}".format(output_path))

## ENTRENAMIENTO

path_training = 'ssd_mobilenet'
os.makedirs(path_training, exist_ok=True)

source_config = "{}/pipeline.config".format(output_path)
target_config = "{}/pipeline.config".format(path_training)

shutil.copyfile(source_config, target_config)

## ----
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

config = config_util.get_configs_from_pipeline_file(target_config)
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

with tf.io.gfile.GFile(target_config, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

# La
label_map_pbtxt_fname = "/label_map.pbtxt"
train_record_fname = "/train.record"
test_record_fname = "/test.record"

pipeline_config.model.ssd.num_classes = 6
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = "{}/checkpoint/ckpt-0".format(output_path)
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path = label_map_pbtxt_fname
pipeline_config.train_input_reader.tf_record_input_reader.input_path[0] = train_record_fname

pipeline_config.eval_input_reader[0].label_map_path = label_map_pbtxt_fname
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[0] = train_record_fname

## 4. configurar el archivo
config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(target_config, "wb") as f:
    f.write(config_text)

## 5 Entrenar el modelo
num_steps = 5000
model_dir = "/ssd_mobilenet"
print (target_config)
## pip install -r ./models/official/requirements.txt

# python .\models\research\object_detection\model_main.py --pipeline_config_path={target_config} --model_dir={model_dir} --num_train_steps={num_steps}
# python .\models\research\object_detection\model_main.py --pipeline_config_path=../../../ssd_mobilenet/pipeline.config --model_dir=ssd_mobilenet --num_train_steps=5000
