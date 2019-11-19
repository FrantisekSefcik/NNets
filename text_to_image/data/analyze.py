import pandas as pd


def basic_analysis(ids_location, labels_location, relationspih_location):

    ids = pd.read_csv(ids_location)
    labels = pd.read_csv(labels_location)
    relationships = pd.read_csv(relationspih_location)

    print("average number of labels per image: " + str(
        labels['ImageID'].count() / labels['ImageID'].nunique()))

