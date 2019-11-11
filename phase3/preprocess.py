from database import Database
from imageprocessor import ImageProcessor
from config import Config
from pymongo import MongoClient
from descriptor import Descriptor, DescriptorType
from latentsymantics import LatentSymantics, LatentSymanticsType
from database import Database


import os
import numpy as np
import pandas
import cv2


def preprocess_images():
    ids, input_vector = ImageProcessor().id_vector_pair
    records = [
        {"id": ids[i], "vector": input_vector[i].tolist()} for i in range(len(ids))
    ]

    Database().insert_many(records)

def insert_image_in_database(path,feature_model,dimension_reduction,k,training,metadata=None):
    descriptor_type = DescriptorType(feature_model).descriptor_type
    symantics_type = LatentSymanticsType(dimension_reduction).symantics_type

    #Read images and feature extraction
    files = os.listdir(path)
    ids, x = [], []
    for file in files:
        print("Reading file: {}".format(file))
        image = cv2.imread("{}{}".format(path, file))

        feature_descriptor = Descriptor(
            image, feature_model, dimension_reduction
        ).feature_descriptor
        ids.append(file.replace(".jpg", ""))
        x.append(feature_descriptor)
    x = np.array(x)


    #Find Latent_symantics
    latent_symantics_model, latent_symantics = LatentSymantics(
        x, k, dimension_reduction
    ).latent_symantics


    #inserting data into Database
    records = []
    if training:
        '''
        dorsal = 1
        palmar = 0
        '''

        for i in range(len(ids)):
            imageName = ids[i] + ".jpg"
            y = metadata[metadata.imageName == imageName]['aspectOfHand'].values
            if(y== 'dorsal right' or y == "dorsal left"):
                y = 1
            else:
                y = 0
            record = {
                "image_id": ids[i],
                "latent_symantics" : latent_symantics[i].tolist(),
                "symantics_type": symantics_type,
                "descriptor_type": descriptor_type,
                "label" : y
            }
            records.append(record)
        Database().insert_many(records,collection_type = "training")
    else:
        for i in range(len(ids)):
            record = {
                "image_id": ids[i],
                "latent_symantics" : latent_symantics[i].tolist(),
                "symantics_type": symantics_type,
                "descriptor_type": descriptor_type,
            }
            records.append(record)
        Database().insert_many(records,collection_type = "testing")

    return

if __name__ == "__main__":
    preprocess_images()

    connection = MongoClient(Config().mongo_url())
    database = connection[Config().database_name()]
    metadata = pandas.read_csv(Config().metadata_file())

    # store training folder
    path = Config().read_training_data_path()
    insert_image_in_database(path,1,1,50,True,metadata)

    # store testing folder
    path = Config().read_testing_data_path()
    insert_image_in_database(path,1,1,50,False)
