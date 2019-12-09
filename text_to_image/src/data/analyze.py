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


def get_ids_with_labels(df_images, df_labels, labels, file_name='image_ids', file_type='txt'):
    image_ids = set(df_labels[df_labels['LabelName'].isin(labels)]['ImageID'].values)
    new_df = df_images[df_images['ImageID'].isin(list(image_ids))].reset_index()
    return find_image_ids_with_labels(new_df, df_labels, labels, file_name=file_name, file_type=file_type)


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


def map_top_labels(df, top_labels):
    mapping_labels = pd.Series([i for i in range(len(top_labels))], index=top_labels)
    top_map = [{"col": "LabelName1", "mapping": mapping_labels},
               {"col": "LabelName2", "mapping": mapping_labels}
               ]
    e = ce.OrdinalEncoder(mapping=top_map)
    e.fit(df[['LabelName1', 'LabelName2']])
    return e


def get_object_bounding_boxes(df_rel, image_id):
    return df_rel[df_rel['ImageID'] == image_id][['XMin1',
                                                  'XMax1',
                                                  'YMin1',
                                                  'YMax1',
                                                  'XMin2',
                                                  'XMax2',
                                                  'YMin2',
                                                  'YMax2']].reset_index(drop=True)


def load_image(url):
    picture_request = requests.get(url)
    if picture_request.status_code == 200:
        image = np.array(bytearray(picture_request.content))
        image = cv2.imdecode(image, -1)
        return image
    return None


def normalize_image(cv2_image, min=-1, max=1, norm_type=cv2.NORM_MINMAX):
    return cv2.normalize(cv2_image, None, alpha=min, beta=max, norm_type=norm_type, dtype=cv2.CV_32F)


def resize_image(cv2_image, new_size=(300, 300), interpolation_mode=cv2.INTER_LINEAR):
    return cv2.resize(cv2_image, new_size, interpolation_mode)


def create_ordinal_encoder_for_triplets(df):
    l1 = df['LabelName1'].unique()
    l2 = df['LabelName2'].unique()
    labels = np.union1d(l1, l2)
    rel = df['RelationshipLabel'].unique()

    mapping_labels = pd.Series([i for i in range(len(labels))], index=labels)
    mapping_rel = pd.Series([i for i in range(len(rel))], index=rel)

    top_map = [{"col": "LabelName1", "mapping": mapping_labels},
               {"col": "LabelName2", "mapping": mapping_labels},
               {"col": "RelationshipLabel", "mapping": mapping_rel},
               ]

    e = ce.OrdinalEncoder(mapping=top_map)
    e.fit(df[['LabelName1', 'RelationshipLabel', 'LabelName2']])
    return e


def encode_image_labels(image_id, df_rel, encoder, cols=['LabelName1',
                                                         'RelationshipLabel',
                                                         'LabelName2']):
    labels = get_image_relationships(df_rel, image_id, cols)
    return encoder.transform(labels)


def image_array_generator(urls, df_rel, df_ids, batch_size=0,
                          resize=True, size=(300, 300), interpolation=cv2.INTER_LINEAR,
                          normalize=True, min=-1, max=1, norm_type=cv2.NORM_MINMAX):
    """
    Generator prepares batches of images for NN

    :param df_ids: dataframe with image IDs and URLs
    :param df_rel: dataframe with labeled relationships
    :param urls: list of urls of images
    :param batch_size: int
    :param resize: bool - defines whether image scaling should be used
    :param size: tuple (int, int) - new size of the images, if resize is True
    :param interpolation: interpolation method from cv2
    :param normalize: bool - defines whether pixel values should be normalized
    :param min: float - the min normalized value if normalize is true
    :param max: float - the max normalized value if normalize is true
    :param norm_type: normalization type from cv2

    yields a batch of preprocessed images
    """
    batch = []
    encoder = create_ordinal_encoder_for_triplets(df_rel)
    if batch_size == 0:
        batch_size = len(urls)
    random.shuffle(urls)
    for URL in urls:
        print(URL)
        if len(batch) >= batch_size:
            yield batch
            batch = []
        image = load_image(URL)
        if image is not None:
            if resize:
                image = resize_image(image, size, interpolation)
            if normalize:
                image = normalize_image(image, min, max, norm_type)

            image_id = df_ids[df_ids['OriginalURL'] == URL]['ImageID'].values[0]

            coded_labels = encode_image_labels(image_id, df_rel, encoder)
            bounding_boxes = get_object_bounding_boxes(df_rel, image_id)
            for i, lab in coded_labels.iterrows():
                batch.append([image, lab.values, bounding_boxes.iloc[i].values])
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    yield batch


def image_array_generator2(urls, df_rel, df_ids, labels=[], batch_size=0,
                           resize=True, size=(300, 300), interpolation=cv2.INTER_LINEAR,
                           normalize=True, min=-1, max=1, norm_type=cv2.NORM_MINMAX):
    """
    Generator prepares batches of images for NN

    :param labels: array-like - list of labels used
    :param df_ids: dataframe with image IDs and URLs
    :param df_rel: dataframe with labeled relationships
    :param urls: list of urls of images
    :param batch_size: int
    :param resize: bool - defines whether image scaling should be used
    :param size: tuple (int, int) - new size of the images, if resize is True
    :param interpolation: interpolation method from cv2
    :param normalize: bool - defines whether pixel values should be normalized
    :param min: float - the min normalized value if normalize is true
    :param max: float - the max normalized value if normalize is true
    :param norm_type: normalization type from cv2

    yields a batch of preprocessed images
    """
    batch_images = []
    batch_labels1 = []
    batch_labels2 = []
    encoder = map_top_labels(df_rel, labels)
    if batch_size == 0:
        batch_size = len(urls)
    random.shuffle(urls)
    for URL in urls:
        # print(URL)
        if len(batch_images) >= batch_size:
            yield [batch_images, batch_labels1, batch_labels2]
            batch_images = []
            batch_labels1 = []
            batch_labels2 = []
        image = load_image(URL)
        if image is not None:
            if resize:
                image = resize_image(image, size, interpolation)
            if normalize:
                image = normalize_image(image, min, max, norm_type)

            image_id = df_ids[df_ids['OriginalURL'] == URL]['ImageID'].values[0]

            coded_labels = encode_image_labels(image_id, df_rel, encoder, cols=['LabelName1','LabelName2'])

            batch_labels1.append(coded_labels.iloc[0, 0])
            batch_labels2.append(coded_labels.iloc[0, 1])
            batch_images.append(image)

    yield [batch_images, batch_labels1, batch_labels2]