from Config import Config
from pymongo import MongoClient
import traceback
import pandas as pd
from descriptor import DescriptorType, Descriptor
from latentsymantics import LatentSymanticsType, LatentSymantics
import cv2
import numpy as np


def findLabels(label):
    if label == 1:
        return "left", "right"
    elif label == 2:
        return "right", "left"
    elif label == 3:
        return "dorsal", "palmar"
    elif label == 4:
        return "palmar", "dorsal"
    elif label == 5:
        return 1, 0
    elif label == 6:
        return 0, 1
    elif label == 7:
        return "male", "female"
    elif label == 8:
        return "female", "male"


def findsimilarity(source_latent_semantics, destination_latent_semantics):
    squares = 0
    for i in range(len(destination_latent_semantics)):
        squares = squares + (source_latent_semantics[i] - destination_latent_semantics[i]) ** 2
    return squares ** 0.5


def findlabel(feature_model, dimension_reduction, k, label, collection, config_object, image_id):
    config_object = Config()
    try:
        feature_vector1 = []
        similarity_vector_given = {}
        similarity_vector_vs = {}
        descriptor_type = DescriptorType(feature_model).descriptor_type
        semantics_type = LatentSymanticsType(dimension_reduction).symantics_type
        label_given, label_vs = findLabels(label)

        img_path = config_object.read_path() + image_id
        image = cv2.imread(img_path)
        feature_vector1.append(Descriptor(image, feature_model).feature_descriptor)
        source_latent_semantics = LatentSymantics(np.array(feature_vector1), k, dimension_reduction).latent_symantics

        for y in collection.find(
                {"descriptor_type": descriptor_type, "symantics_type": semantics_type, "label": label_given},
                {"latent_symantics": 1, "image_id": 1}):
            destination_latent_semantics = y.get("latent_symantics")
            # print(destination_latent_semantics)
            similarity_vector_id = findsimilarity(source_latent_semantics, destination_latent_semantics)
            similarity_vector_given[y.get("image_id")] = similarity_vector_id

        given_similarity_vector = sorted(similarity_vector_given.items(), key=lambda x: x[1])[:1]

        for y in collection.find(
                {"descriptor_type": descriptor_type, "symantics_type": semantics_type, "label": label_vs},
                {"latent_symantics": 1, "image_id": 1}):
            destination_latent_semantics = y.get("latent_symantics")
            # print(destination_latent_semantics)
            similarity_vector_id = findsimilarity(source_latent_semantics, destination_latent_semantics)
            similarity_vector_vs[y.get("image_id")] = similarity_vector_id

        vs_similarity_vector = sorted(similarity_vector_given.items(), key=lambda x: x[1])[:1]

        if given_similarity_vector < vs_similarity_vector:
            print label_given
        else:
            print label_vs

    except Exception as e:
        traceback.print_exc()
        print("Error finding m related images")


def helper(feature_model, dimension_reduction, k, label, collection, config_object):
    write_to = config_object.write_path()
    descriptor_type = DescriptorType(feature_model).descriptor_type
    semantics_type = LatentSymanticsType(dimension_reduction).symantics_type
    label_given, label_vs = findLabels(label)

    ids1, ids2, feature_vector1, feature_vector2 = [], [], [], []
    if label < 5:
        for subject in collection.find({"aspectOfHand": {"$regex": label_given}}, {"imageName": 1}):
            image_id = subject['imageName']
            img_path = config_object.read_path() + image_id
            image = cv2.imread(img_path)
            ids1.append(image_id.replace(".jpg", ""));
            feature_descriptor = Descriptor(image, feature_model).feature_descriptor
            feature_vector1.append(feature_descriptor);

        for subject in collection.find({"aspectOfHand": {"$regex": label_vs}}, {"imageName": 1}):
            image_id = subject['imageName']
            img_path = config_object.read_path() + image_id
            image = cv2.imread(img_path)
            ids2.append(image_id.replace(".jpg", ""));

            feature_descriptor = Descriptor(image, feature_model).feature_descriptor
            feature_vector2.append(feature_descriptor);

        latent_symantics1 = LatentSymantics(np.array(feature_vector1), k, dimension_reduction).latent_symantics

        records1 = [
            {"imageid": ids1[i],
             "descriptor_type": descriptor_type,
             "symantics_type": semantics_type,
             "k": k,
             "label": label_given,
             "latent_symantic": latent_symantics1[i].tolist()
             }
            for i in range(len(ids1))
        ]

        latent_symantics2 = LatentSymantics(np.array(feature_vector2), k, dimension_reduction).latent_symantics

        records2 = [
            {"imageid": ids2[i],
             "descriptor_type": descriptor_type,
             "symantics_type": semantics_type,
             "k": k,
             "label": label_vs,
             "latent_symantic": latent_symantics2[i].tolist()
             }
            for i in range(len(ids2))
        ]

    return records1, records2


def starter(feature_model, dimension_reduction, k, label, imageID):
    config_object = Config()
    mongo_url = config_object.mongo_url()
    database_name = config_object.database_name()
    collection_name = config_object.collection_name()
    meta_collection_name = config_object.k_metadata_collection_name()

    try:
        connection = MongoClient(mongo_url)
        database = connection[database_name]
        collection = database[collection_name]
        meta_collection = database[meta_collection_name]

        # Function to parse csv to dictionary
        df = pd.read_csv(config_object.metadata_csv())
        records = df.to_dict(orient='records')

        collection.remove()
        collection.insert_many(records)

        meta_collection.remove()
        records1, records2 = helper(feature_model, dimension_reduction, k, label, collection, config_object)
        meta_collection.insert_many(records1)
        meta_collection.insert_many(records2)
        findlabel(feature_model, dimension_reduction, k, label, collection, config_object, imageID)

    except Exception as e:
        traceback.print_exc()
        print("Connection refused... ")


if __name__ == '__main__':
    starter(1, 1, 10, 4, 'Hand_0002079.jpg')
