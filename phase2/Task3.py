import pandas as pd
import numpy as np
import cv2
import traceback
import csv
from descriptor import Descriptor, DescriptorType
from latentsymantics import LatentSymantics, LatentSymanticsType
from Config import Config
from pymongo import MongoClient

def findlabel(label):

    if label==1:
        return "left"
    elif label==2:
        return "right"
    elif label == 3:
        return "dorsal"
    elif label == 4:
        return  "palmer"
    elif label == 5:
        return 1
    elif label ==6:
        return 0
    elif label ==7:
        return "male"
    elif label ==8:
        return "female"


def helper(feature_model, dimension_reduction, k, label, mycol,config_object):
    writeto = config_object.write_path()
    descriptor_type = DescriptorType(feature_model).descriptor_type
    symantics_type = LatentSymanticsType(dimension_reduction).symantics_type
    exactlabel = findlabel(label)
    print(exactlabel)

    ids, feature_vector = [], []
    if label < 5:
        for x in mycol.find({"aspectOfHand": {"$regex" :exactlabel}}, {"imageName": 1}):
            imageid = x.get("imageName");
            print(imageid)

            imgpath = config_object.read_path() + imageid;
            image = cv2.imread(imgpath)

            feature_descriptor = Descriptor(image, feature_model).feature_descriptor
            ids.append(imageid.replace(".jpg", ""))
            feature_vector.append(feature_descriptor)

    elif label ==5 or label==6:
        for x in mycol.find({"accessories": exactlabel}, {"imageName": 1}):
            imageid = x.get("imageName");
            print(imageid)

            imgpath = config_object.read_path() + imageid;
            image = cv2.imread(imgpath)

            feature_descriptor = Descriptor(image, feature_model).feature_descriptor
            ids.append(imageid.replace(".jpg", ""))
            feature_vector.append(feature_descriptor)


    elif label>6:
        for x in mycol.find({"gender": exactlabel}, {"imageName": 1}):
            imageid = x.get("imageName");
            print(imageid)

            imgpath = config_object.read_path() + imageid;
            image = cv2.imread(imgpath)

            feature_descriptor = Descriptor(image, feature_model).feature_descriptor
            ids.append(imageid.replace(".jpg", ""))
            feature_vector.append(feature_descriptor)
            print(feature_vector)

    latent_symantics = LatentSymantics(
        np.array(feature_vector), k, dimension_reduction
    ).latent_symantics
    records = [
        {
            "image_id": ids[i],
            "descriptor_type": descriptor_type,
            "symantics_type": symantics_type,
            "k": k,
            "label":label,
            "latent_symantics": latent_symantics[i].tolist(),
        }
        for i in range(len(ids))
    ]

    return records

def starter(feature_model, dimension_reduction, k, label):
    config_object = Config()
    path = config_object.read_path()
    writeto = config_object.write_path()
    mongo_url = config_object.mongo_url()
    database_name = config_object.database_name()
    collection_name = config_object.metadata_collection_name()
    meta_collection_name= config_object.k_metadata_collection_name()
    source_latent_semantics = {}
    destination_latent_semantics = {}
    try:
        connection = MongoClient(mongo_url)
        database = connection[database_name]
        collection = database[collection_name]
        meta_collection=database[meta_collection_name]
        print(collection)

        # Function to parse csv to dictionary
        df = pd.read_csv(config_object.metadata_csv())  # csv file which you want to import
        records_ = df.to_dict(orient='records')
        # Final insert statement
        collection.remove()
        collection.insert_many(records_)
        # inserting into new table having data for a particular label and its k latent features
        print(meta_collection)
        meta_collection.insert_many(helper(feature_model,dimension_reduction,k,label,collection,config_object))


    except Exception as e:
        traceback.print_exc()
        print("Connection refused")

    print("Task 3 completed")