import cv2
import numpy as np
import os
import traceback

from config import Config
from descriptor import Descriptor, DescriptorType
from latentsymantics import LatentSymantics, LatentSymanticsType
from pymongo import MongoClient


def process_files(path, feature_model):
    files = os.listdir(path)

    ids, x = [], []
    for file in files:
        print("Reading file: {}".format(file))
        image = cv2.imread(path + file)

        feature_descriptor = Descriptor(image, feature_model).feature_descriptor
        ids.append(file.replace(".jpg", ""))
        x.append(feature_descriptor)

    return x, ids


def starter(feature_model, dimension_reduction, k):
    config_object = Config()
    path = config_object.read_path()
    mongo_url = config_object.mongo_url()
    database_name = config_object.database_name()
    collection_name = config_object.collection_name()

    x, ids = process_files(path, feature_model)
    descriptor_type = DescriptorType(feature_model).descriptor_type
    symantics_type = LatentSymanticsType(dimension_reduction).symantics_type
    latent_symantics = LatentSymantics(
        np.array(x), k, dimension_reduction
    ).latent_symantics

    records = [
        {
            "image_id": ids[i],
            "descriptor_type": descriptor_type,
            "symantics_type": symantics_type,
            "k": k,
            "latent_symantics": latent_symantics[i].tolist(),
        }
        for i in range(len(ids))
    ]

    try:
        connection = MongoClient(mongo_url)
        database = connection[database_name]
        collection = database[collection_name]

        collection.insert_many(records)
        print("Successfully inserted into DB... ")

    except Exception as e:
        traceback.print_exc()
        print("Connection refused... ")

    print("Done... ")
