## 1. Importación de data set.
import os
import zipfile
import random
import shutil

local_zip = "dataset.zip"
zip_ref = zipfile.ZipFile(local_zip, "r")
zip_ref.extractall("dataset")
zip_ref.close()

## 2. Dividir base de datos del dataset aleatoriamente. (Creación directorios)
path = "dataset/dataset"
content = os.listdir(path)

path_general = "datasetFinal"
path_train = "{}/train".format(path_general)
path_test = "{}/test".format(path_general)

os.makedirs(path_general, exist_ok=True)
os.makedirs(path_train, exist_ok=True)
os.makedirs(path_test, exist_ok=True)

## 2.1 Dividir base de datos del dataset aleatoriamente. (División)
train = 0.5
for nCount in range(int(len(content) * train)):
    random_choice_img = random.choice(content)
    random_choice_img_abs = "{}/{}".format(path, random_choice_img)
    target_img = "{}/{}".format(path_train, random_choice_img)
    shutil.copyfile(random_choice_img_abs, target_img)
    content.remove(random_choice_img)

for img in content:
    random_choice_img_abs = "{}/{}".format(path, img)
    target_img = "{}/{}".format(path_test, img)
    shutil.copyfile(random_choice_img_abs, target_img)

## 3. Descargamos la bd del dataset.
shutil.make_archive("datasetFinal", 'zip', path_general)