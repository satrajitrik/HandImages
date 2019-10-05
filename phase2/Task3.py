import pandas as pd
import numpy as np
import json
import traceback
import csv

from Config import Config
from descriptor import DescriptorType
from latentsymantics import LatentSymanticsType
from pymongo import MongoClient


def findsimilarity(source_vector, dest_vector):
    dot_product = np.dot(source_vector, dest_vector)
    norm_a = np.linalg.norm(source_vector)
    norm_b = np.linalg.norm(dest_vector)
    return dot_product / (norm_a * norm_b)



def starter(feature_model, dimension_reduction, k, label):
    config_object = Config()
    path = config_object.read_path()
    writeto = config_object.write_path()
    mongo_url = config_object.mongo_url()
    database_name = config_object.database_name()
    collection_name = config_object.metadata_collection_name()
    source_latent_semantics = {}
    destination_latent_semantics = {}
    try:
        connection = MongoClient(mongo_url)
        database = connection[database_name]
        collection = database[collection_name]
        print(collection)

        # Function to parse csv to dictionary
        df = pd.read_csv(config_object.metadata_csv())  # csv file which you want to import
        records_ = df.to_dict(orient='records')
        # Final insert statement
        collection.insert_many(records_)




    except Exception as e:
        traceback.print_exc()
        print("Connection refused... ")

    print("Done... ")