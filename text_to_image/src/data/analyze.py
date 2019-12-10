import pandas as pd
import numpy as np
import cv2
from PIL import Image
import requests
import io
import random
import category_encoders as ce


def load_data(ids_location, labels_location, relationship_location):
    ids = pd.read_csv(ids_location)
    labels = pd.read_csv(labels_location)
    relationships = pd.read_csv(relationship_location)
    return ids, labels, relationships


def get_decode_dict(path='./metadata/label_names.csv'):
    decode_dict = {}
    label_names = pd.read_csv(path, header=None, names=['Code', 'Name'])
    for index, row in label_names.iterrows():
        decode_dict[row['Code']] = row['Name']
    return decode_dict


def decode_class_names(class_codes, decode_dict, path='./metadata/label_names.csv'):
    return [decode_dict[class_code] for class_code in class_codes]


def show_image(image_url):
    picture_request = requests.get(image_url)
    if picture_request.status_code == 200:
        return Image.open(io.BytesIO(picture_request.content))


def get_image_labels(image_id, labels_df):
    return labels_df[labels_df["ImageID"] == image_id]['LabelName'].values


def get_image_relationships(df_rel, image_id, cols=['LabelName1',
                                                         'RelationshipLabel',
                                                         'LabelName2']):
    return df_rel[df_rel['ImageID'] == image_id][cols].reset_index(drop=True)


def load_image(url):
    picture_request = requests.get(url)
    if picture_request.status_code == 200:
        image = np.array(bytearray(picture_request.content))
        image = cv2.imdecode(image, -1)
        return image
    return None
