import pandas as pd
from PIL import Image
import requests
import io
from IPython.core.display import HTML
from IPython.display import clear_output

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
            clear_output(wait=True)
            print(index, "/", len(df_images))
    return return_list


def get_ids_with_labels(df_images, df_labels, labels, file_name='image_ids', file_type='txt'):
    image_ids = set(df_labels[df_labels['LabelName'].isin(labels)]['ImageID'].values)
    new_df = df_images[df_images['ImageID'].isin(list(image_ids))].reset_index()
    return find_image_ids_with_labels(new_df, df_labels, labels, file_name=file_name, file_type=file_type)


def decode_class_names(class_codes):
    return [label_names.loc[label_names['Code'] == class_code].iat[0, 1] for class_code in class_codes]


def show_image(image_url, url_col='OriginalURL', width=600):
    picture_coded = requests.get(image_url).content
    return Image.open(io.BytesIO(picture_coded))


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
        return np.array(Image.open(io.BytesIO(picture_request.content)))


def image_array_generator(urls, batch_size=0):
    i = 0
    batch = []
    if batch_size == 0:
        batch_size = len(urls)
    for URL in imageUrlGenerator():
        if i > batch_size:
            yield batch
            batch = []
            i = 0
        batch.append(loadImage(URL))
        i += 1
