## 1. Imports de librerias.
import json
import pickle
import zipfile
import pandas as pd
# pip install tf_slim

## 2. Descomprimimos la bd
local_zip = "dataset_original.zip"
zip_ref = zipfile.ZipFile(local_zip, "r")
zip_ref.extractall("dataset_original")
zip_ref.close()

## 3. Json labels to csv.
type_file = "train"
path = "train.json"
data_file = open(path)
data = json.load(data_file)

csv_list = []

for classification in data:
    width, height = classification['width'], classification['height']
    image = classification['image']
    for item in classification['tags']:
        name = item['name']
        xmin = item['pos']['x']
        ymin = item['pos']['y']
        xmax = item['pos']['x'] + item['pos']['w']
        ymax = item['pos']['y'] + item['pos']['h']

        values = (image, width, height, name, xmin, ymin, xmax, ymax)
        csv_list.append(values)

column_names = ['filename', 'width', 'height', 'classname', 'xmin', 'ymin', 'xmax', 'ymax']
csv_df = pd.DataFrame(csv_list, columns=column_names)

csv_df.to_csv("{}_labels.csv".format(type_file))