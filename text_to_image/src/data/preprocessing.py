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
        elif newymax > imagey:
            newymin -= (newymax - imagey)
            newymax = imagey
    elif sizedif > 0:
        newxmin -= sizedifhalf + par
        newxmax += sizedifhalf
        if newxmin < 0:
            newxmax -= newxmin
            newxmin = 0
        elif newxmax > imagex:
            newxmin -= (newxmax - imagex)
            newxmax = imagex

    return newxmin, newxmax, newymin, newymax


def image_generator(imgs_dir, relationships, top, labels_coded,
                    batch_size=0,
                    resize=True, size=(300, 300), interpolation=cv2.INTER_LINEAR,
                    normalize=True, min=-1, max=1, norm_type=cv2.NORM_MINMAX):
    directory = os.fsencode(imgs_dir)

    labels = []
    boxes = []

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".npy"):
            image_id = filename.split('.')[0]
            image = np.load(imgs_dir + filename)

            for index, row in relationships[relationships['ImageID'] == image_id].iterrows():

                label = row['LabelName1']

                imagex = image.shape[0]
                imagey = image.shape[1]

                if label in top:
                    x1min, x1max, y1min, y1max = row['XMin1'], row['XMax1'], row['YMin1'], row['YMax1']

                    labels.append(labels_coded[row['LabelName1']])
                    x1, x2, y1, y2 = get_boundaries(imagex, imagey, x1min, x1max, y1min, y1max)
                    box = image[x1: x2, y1: y2]
                    if resize:
                        box = resize_image(box, size, interpolation)
                    if normalize:
                        box = normalize_image(box, min, max, norm_type)
                    boxes.append(box)

                    if 0 < batch_size <= len(labels):
                        yield labels, boxes
                        labels = []
                        boxes = []

                label = row['LabelName2']

                if label in top:
                    x2min, x2max, y2min, y2max = row['XMin2'], row['XMax2'], row['YMin2'], row['YMax2']
                    labels.append(labels_coded[row['LabelName1']])
                    x1, x2, y1, y2 = get_boundaries(imagex, imagey, x2min, x2max, y2min, y2max)
                    box = image[x1: x2, y1: y2]
                    if resize:
                        box = resize_image(box, size, interpolation)
                    if normalize:
                        box = normalize_image(box, min, max, norm_type)
                    boxes.append(box)

                    if 0 < batch_size <= len(labels):
                        yield labels, boxes
                        labels = []
                        boxes = []
        yield labels, boxes


def get_image_generator(relationships_location, imgs_dir,
                        top_n_labels=10,
                        batch_size=0,
                        resize=True, size=(300, 300), interpolation=cv2.INTER_LINEAR,
                        normalize=True, min=-1, max=1, norm_type=cv2.NORM_MINMAX
                        ):
    relationships = pd.read_csv(relationships_location)

    top_labels1 = relationships.groupby('LabelName1').count().sort_values(
        'ImageID', ascending=False).head(top_n_labels)['ImageID'].to_frame()

    top_labels2 = relationships.groupby('LabelName2').count().sort_values(
        'ImageID', ascending=False).head(top_n_labels)['ImageID'].to_frame()

    for i in range(top_n_labels):
        if top_labels2.index[i] in top_labels1.index:
            top_labels1.loc[top_labels2.index[i]] += top_labels2.iloc[i].values[0]
        else:
            top_labels1 = top_labels1.append(top_labels2.iloc[i])

    top = top_labels1.head(top_n_labels).index.values

    labels_coded = {}

    for i in range(len(top)):
        labels_coded[top[i]] = i

    g = image_generator(imgs_dir, relationships, top, labels_coded,
                        batch_size, resize, size, interpolation,
                        normalize, min, max, norm_type)
    return g, labels_coded
