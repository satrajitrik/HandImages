import cv2
import numpy as np
import os
import traceback

from config import Config
from database import Database
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


def get_latentsymantics_and_insert(feature_model, dimension_reduction, k):
    path = Config().read_path()

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

    Database().insert_many(records)
    return latent_symantics


def starter(feature_model, dimension_reduction, k):
    print(get_latentsymantics_and_insert(feature_model, dimension_reduction, k))
    print("Done... ")
