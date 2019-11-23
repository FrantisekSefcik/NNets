import pandas as pd
import numpy as np
import cv2
from PIL import Image
import requests
import io
import random

label_names = pd.read_csv('./metadata/label_names.csv', header=None, names=['Code', 'Name'])


def load_data(ids_location, labels_location, relationship_location):
    ids = pd.read_csv(ids_location)
    labels = pd.read_csv(labels_location)
    relationships = pd.read_csv(relationship_location)
    return ids, labels, relationships


def basic_analysis(ids, labels):
    print("number of images: " + str(ids.shape[0]))
    top_overall = top_labels_analysis(labels)
    print()

    groupby_label = labels.groupby(['ImageID']).count()
    one_label = groupby_label[groupby_label['LabelName'] < 2]

    print("number of images with 1 label: " + str(len(one_label)))
    top_one_label = top_labels_analysis(labels.loc[labels['ImageID'].isin(one_label.index)])
    print()

    return top_overall, top_one_label


def top_labels_analysis(df):
    print("number of label classes: " + str(df['LabelName'].nunique()))
    print("average number of labels per image: " + str(
        df['ImageID'].count() / df['ImageID'].nunique()))
    print("top 10 classes: ")
    top_classes = df.groupby('LabelName').count().sort_values('ImageID', ascending=False).head(10)['ImageID']
    top_classes = top_classes.to_frame()
    top_classes['Name'] = decode_class_names(top_classes.index)
    top_classes.columns = ['Image count', 'Class name']
    top_classes.loc['sum'] = [top_classes['Image count'].sum(), "sum"]
    print(top_classes)
    return top_classes.index.values[:10].tolist()


def find_image_ids_with_labels(df_images, df_labels, labels, file_name='image_ids', file_type='txt'):
    id_list = []
    return_list = []
    for index, image in df_images.iterrows():
        image_labels = set(get_image_labels(image['ImageID'], df_labels))
        if len(image_labels) and len(image_labels - labels) == 0:
            id_list += [image['ImageID']]
        if index % 100 == 0:
            f = open(file_name + '.' + file_type, "a")
            return_list += id_list
            for item in id_list:
                f.write(item + ',\n')
            f.close()
            id_list = []
            print(index, "/", len(df_images))
    return return_list


def get_ids_with_labels(df_images, df_labels, labels, file_name='image_ids', file_type='txt'):
    image_ids = set(df_labels[df_labels['LabelName'].isin(labels)]['ImageID'].values)
    new_df = df_images[df_images['ImageID'].isin(list(image_ids))].reset_index()
    return find_image_ids_with_labels(new_df, df_labels, labels, file_name=file_name, file_type=file_type)


def decode_class_names(class_codes):
    return [label_names.loc[label_names['Code'] == class_code].iat[0, 1] for class_code in class_codes]


def show_image(image_url):
    picture_request = requests.get(image_url)
    if picture_request.status_code == 200:
        return Image.open(io.BytesIO(picture_request.content))


def get_image_labels(image_id, labels_df):
    return labels_df[labels_df["ImageID"] == image_id]['LabelName'].values


def get_image_relationships(df_rel, image_id):
    l1 = df_rel[df_rel['ImageID'] == image_id]['LabelName1']
    lr = df_rel[df_rel['ImageID'] == image_id]['RelationshipLabel']
    l2 = df_rel[df_rel['ImageID'] == image_id]['LabelName2']
    return [a + " " for a in decode_class_names(l1)] + lr + [" " + a for a in decode_class_names(l2)]


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


def image_array_generator(urls, batch_size=0,
                          resize=True, size=(300, 300), interpolation=cv2.INTER_LINEAR,
                          normalize=True, min=-1, max=1, norm_type=cv2.NORM_MINMAX):
    """
    Generator prepares batches of images for NN

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
    i = 0
    batch = []
    if batch_size == 0:
        batch_size = len(urls)
    random.shuffle(urls)
    for URL in urls:
        print(URL)
        if len(batch) >= batch_size:
            yield batch
            batch = []
            i = 0
        image = load_image(URL)
        if image is not None:
            if resize:
                image = resize_image(image, size, interpolation)
            if normalize:
                image = normalize_image(image, min, max, norm_type)
            batch.append(image)
        i += 1
    yield batch
