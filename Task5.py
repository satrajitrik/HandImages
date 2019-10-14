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
        feature_descriptor = []
        similarity_vector_given = {}
        similarity_vector_vs = {}
        descriptor_type = DescriptorType(feature_model).descriptor_type
        semantics_type = LatentSymanticsType(dimension_reduction).symantics_type
        label_given, label_vs = findLabels(label)

        img_path = config_object.read_path() + image_id
        image = cv2.imread(img_path)
        feature_descriptor.append(Descriptor(image, feature_model).feature_descriptor)
        source_latent_semantics = (collection.find_one(
            {"descriptor_type": descriptor_type, "symantics_type": semantics_type, "label": "unknown"},
            {"latent_symantic": 1})).get("latent_symantic")
        for y in collection.find(
                {"descriptor_type": descriptor_type, "symantics_type": semantics_type, "label": label_given},
                {"latent_symantic": 1, "imageid": 1}):
            destination_latent_semantics = y.get("latent_symantic")
            similarity_vector_id = findsimilarity(source_latent_semantics, destination_latent_semantics)
            similarity_vector_given[y.get("imageid")] = similarity_vector_id

        given_similarity_vector = sorted(similarity_vector_given.items(), key=lambda x: x[1])[:1]

        for y in collection.find(
                {"descriptor_type": descriptor_type, "symantics_type": semantics_type, "label": label_vs},
                {"latent_symantic": 1, "imageid": 1}):
            destination_latent_semantics = y.get("latent_symantic")
            # print(destination_latent_semantics)
            similarity_vector_id = findsimilarity(source_latent_semantics, destination_latent_semantics)
            similarity_vector_vs[y.get("imageid")] = similarity_vector_id

        vs_similarity_vector = sorted(similarity_vector_given.items(), key=lambda x: x[1])[:1]

        if given_similarity_vector[0] < vs_similarity_vector[0]:
            print label_given
        else:
            print label_vs

    except Exception as e:
        traceback.print_exc()
        print("Error finding m related images")


def helper(feature_model, dimension_reduction, k, label, collection, config_object, imageID):
    write_to = config_object.write_path()
    descriptor_type = DescriptorType(feature_model).descriptor_type
    semantics_type = LatentSymanticsType(dimension_reduction).symantics_type
    label_given, label_vs = findLabels(label)

    ids1, ids2, feature_vector = [], [], []
    if label < 5:
        for subject in collection.find({"aspectOfHand": {"$regex": label_given}}, {"imageName": 1}):
            image_id = subject['imageName']
            img_path = config_object.read_path() + image_id
            image = cv2.imread(img_path)
            ids1.append(image_id.replace(".jpg", ""));
            feature_descriptor = Descriptor(image, feature_model).feature_descriptor
            feature_vector.append(feature_descriptor);

        feature_vector.append(
            Descriptor(cv2.imread(config_object.read_path() + imageID), feature_model).feature_descriptor)

        for subject in collection.find({"aspectOfHand": {"$regex": label_vs}}, {"imageName": 1}):
            image_id = subject['imageName']
            img_path = config_object.read_path() + image_id
            image = cv2.imread(img_path)
            ids2.append(image_id.replace(".jpg", ""));
            feature_descriptor = Descriptor(image, feature_model).feature_descriptor
            feature_vector.append(feature_descriptor);

        latent_symantics = LatentSymantics(np.array(feature_vector), k, dimension_reduction).latent_symantics

        records = [
            {"imageid": ids1[i],
             "descriptor_type": descriptor_type,
             "symantics_type": semantics_type,
             "k": k,
             "label": label_given,
             "latent_symantic": latent_symantics[i].tolist()
             }
            for i in range(len(ids1) - 1)
        ]
        records.append(
            {"imageid": imageID.replace(".jpg", ""),
             "descriptor_type": descriptor_type,
             "symantics_type": semantics_type,
             "k": k,
             "label": "unknown",
             "latent_symantic": latent_symantics[len(ids1) - 1].tolist()
             })
        for i in range(len(ids2)):
            records.append(
                {"imageid": ids2[i],
                 "descriptor_type": descriptor_type,
                 "symantics_type": semantics_type,
                 "k": k,
                 "label": label_vs,
                 "latent_symantic": latent_symantics[len(ids1) + i].tolist()
                 })

    return records


def starter(feature_model, dimension_reduction, k, label, imageID):
    config_object = Config()
    mongo_url = config_object.mongo_url()
    database_name = config_object.database_name()
    collection_name = config_object.collection_name()
    meta_collection_name = config_object.metadata_collection_name()

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
        records = helper(feature_model, dimension_reduction, k, label, collection, config_object, imageID)
        meta_collection.insert_many(records)
        findlabel(feature_model, dimension_reduction, k, label, meta_collection, config_object, imageID)

    except Exception as e:
        traceback.print_exc()
        print("Connection refused... ")


