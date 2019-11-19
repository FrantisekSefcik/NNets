import pandas as pd
from IPython.display import Image
from IPython.core.display import HTML

label_names = pd.read_csv('./metadata/label_names.csv', header=None, names=['Code', 'Name'])


def load_data(ids_location, labels_location, relationship_location):
    ids = pd.read_csv(ids_location)
    labels = pd.read_csv(labels_location)
    relationships = pd.read_csv(relationship_location)
    return ids, labels, relationships


def basic_analysis(ids, labels, relationships):

    print("number of images: " + str(ids.shape[0]))
    label_class_analysis(labels)
    print()

    groupby_label = labels.groupby(['ImageID']).count()
    one_label = groupby_label[groupby_label['LabelName'] < 2]
    two_labels = groupby_label[groupby_label['LabelName'] < 3]
    three_labels = groupby_label[groupby_label['LabelName'] < 4]

    print("number of images with 1 label: " + str(len(one_label)))
    label_class_analysis(labels.loc[labels['ImageID'].isin(one_label.index)])
    print()

    print("number of images with less than 3 label: " + str(len(two_labels)))
    label_class_analysis(labels.loc[labels['ImageID'].isin(two_labels.index)])
    print()

    print("number of images with less than 4 label: " + str(len(three_labels)))
    label_class_analysis(labels.loc[labels['ImageID'].isin(three_labels.index)])


def label_class_analysis(df):
    print("number of label classes: " + str(df['LabelName'].nunique()))
    print("average number of labels per image: " + str(
        df['ImageID'].count() / df['ImageID'].nunique()))
    print("top 10 classes: ")
    top_classes = df.groupby('LabelName').count().sort_values('ImageID', ascending=False).head(10)['ImageID']
    top_classes = top_classes.to_frame()
    top_classes['Name'] = [decode_class_name(name) for name in top_classes.index]
    top_classes.columns = ['Image count', 'Class name']
    print(top_classes)


def decode_class_name(class_code):
    return label_names.loc[label_names['Code'] == class_code].iat[0, 1]


def show_image(ids_df, imageID, url_col='OriginalURL', width=600):
    url = ids_df[ids_df['ImageID'] == imageID][url_col].values[0]
    return Image(url=url, width=width)
