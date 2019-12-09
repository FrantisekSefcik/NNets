import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os
import requests
import io
import random
import category_encoders as ce


def load_data(ids_location, labels_location, relationship_location):
    ids = pd.read_csv(ids_location)
    labels = pd.read_csv(labels_location)
    relationships = pd.read_csv(relationship_location)
    return ids, labels, relationships


def normalize_image(cv2_image, min=-1, max=1, norm_type=cv2.NORM_MINMAX):
    return cv2.normalize(cv2_image, None, alpha=min, beta=max, norm_type=norm_type, dtype=cv2.CV_32F)


def resize_image(cv2_image, new_size=(300, 300), interpolation_mode=cv2.INTER_LINEAR):
    return cv2.resize(cv2_image, new_size, interpolation_mode)


def get_boundaries(imagex, imagey, xmin, xmax, ymin, ymax):
    newxmin = round(xmin * imagex)
    newxmax = round(xmax * imagex)
    newymin = round(ymin * imagey)
    newymax = round(ymax * imagey)

    xdif, ydif = newxmax - newxmin, newymax - newymin
    vertical = True if xdif >= ydif else False
    sizedif = xdif - ydif if vertical else ydif - xdif
    sizedifhalf = sizedif // 2
    par = sizedif % 2
    if vertical and sizedif > 0:
        newymin -= sizedifhalf + par
        newymax += sizedifhalf
        if newymin < 0:
            newymax += newymin
            newymin = 0
            if newymax > imagey:
                newymax = imagey
        elif newymax > imagey:
            newymin -= (newymax - imagey)
            newymax = imagey
            if newymin < 0:
                newymin = 0
    elif sizedif > 0:
        newxmin -= sizedifhalf + par
        newxmax += sizedifhalf
        if newxmin < 0:
            newxmax += newxmin
            newxmin = 0
            if newxmax > imagex:
                newxmax = imagex
        elif newxmax > imagex:
            newxmin -= (newxmax - imagex)
            newxmax = imagex
            if newxmin < 0:
                newxmin = 0

    return newxmin, newxmax, newymin, newymax


def image_preprocessing_func(imgs_dir, relationships, top, labels_coded,
                             number_of_images=-1,
                             resize=True, size=(300, 300), interpolation=cv2.INTER_LINEAR,
                             normalize=True, min=-1, max=1, norm_type=cv2.NORM_MINMAX):
    directory = os.fsencode(imgs_dir)

    labels = []
    boxes = []

    for file in os.listdir(directory)[:number_of_images]:
        filename = os.fsdecode(file)
        if filename.endswith(".npy"):
            image_id = filename.split('.')[0]
            image = np.load(imgs_dir + filename)
            if len(image.shape) < 3:
                continue

            for index, row in relationships[relationships['ImageID'] == image_id].iterrows():

                label = row['DecodedName1']

                imagex = image.shape[1]
                imagey = image.shape[0]

                if label in top:
                    x1min, x1max, y1min, y1max = row['XMin1'], row['XMax1'], row['YMin1'], row['YMax1']

                    labels.append(labels_coded[label])
                    x1, x2, y1, y2 = get_boundaries(imagex, imagey, x1min, x1max, y1min, y1max)
                    box = image[y1: y2, x1: x2]
                    if resize:
                        box = resize_image(box, size, interpolation)
                    if normalize:
                        box = normalize_image(box, min, max, norm_type)
                    boxes.append(box)

                label = row['DecodedName2']

                if label in top:
                    x2min, x2max, y2min, y2max = row['XMin2'], row['XMax2'], row['YMin2'], row['YMax2']
                    labels.append(labels_coded[label])
                    x1, x2, y1, y2 = get_boundaries(imagex, imagey, x2min, x2max, y2min, y2max)
                    box = image[y1: y2, x1: x2]
                    if resize:
                        box = resize_image(box, size, interpolation)
                    if normalize:
                        box = normalize_image(box, min, max, norm_type)
                    boxes.append(box)

    return np.array(boxes), np.array(labels)


def get_decode_dict(path='../../data/metadata/label_names.csv'):
    decode_dict = {}
    label_names = pd.read_csv(path, header=None, names=['Code', 'Name'])
    for index, row in label_names.iterrows():
        decode_dict[row['Code']] = row['Name']
    return decode_dict


def get_top_n_labels(df_relationships, top_n_labels):

    top_labels1 = df_relationships.groupby('DecodedName1').count().sort_values(
        'ImageID', ascending=False).head(top_n_labels)['ImageID'].to_frame()

    top_labels2 = df_relationships.groupby('DecodedName2').count().sort_values(
        'ImageID', ascending=False).head(top_n_labels)['ImageID'].to_frame()

    for i in range(top_n_labels):
        if top_labels2.index[i] in top_labels1.index:
            top_labels1.loc[top_labels2.index[i]] += top_labels2.iloc[i].values[0]
        else:
            top_labels1 = top_labels1.append(top_labels2.iloc[i])

    return top_labels1.head(top_n_labels).index.values


def get_image_generator(relationships_location, imgs_dir,
                        labels=[], top_n_labels=10,
                        number_of_images=-1,
                        resize=True, size=(300, 300), interpolation=cv2.INTER_LINEAR,
                        normalize=True, min=-1, max=1, norm_type=cv2.NORM_MINMAX
                        ):

    df_relationships = pd.read_csv(relationships_location)

    decode_dict = get_decode_dict()

    def decode_label(label_code):
        return decode_dict[label_code]

    df_relationships['DecodedName1'] = df_relationships['LabelName1'].apply(decode_label)
    df_relationships['DecodedName2'] = df_relationships['LabelName2'].apply(decode_label)

    if not labels:
        labels = get_top_n_labels(df_relationships, top_n_labels)
    labels_coded = {}

    for i in range(len(labels)):
        labels_coded[labels[i]] = i

    g = image_preprocessing_func(imgs_dir, df_relationships, labels, labels_coded,
                                 number_of_images, resize, size, interpolation,
                                 normalize, min, max, norm_type)
    return g, labels_coded
