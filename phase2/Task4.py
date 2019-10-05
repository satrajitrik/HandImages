import pandas as pd
import numpy as np
import cv2
import traceback
import json
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
        return  "palmar"
    elif label == 5:
        return 1
    elif label ==6:
        return 0
    elif label ==7:
        return "male"
    elif label ==8:
        return "female"

def findmrelatedimages(feature_model,dimension_reduction,k,label,collection,config_object,image_id):
    config_object = Config()
    path = config_object.read_path()
    writeto = config_object.write_path()
    mongo_url = config_object.mongo_url()
    print(collection)
    source_latent_semantics = {}
    destination_latent_semantics = {}
    try:
        similarity_vector = {}
        descriptor_type = DescriptorType(feature_model).descriptor_type
        symantics_type = LatentSymanticsType(dimension_reduction).symantics_type

        for x in collection.find(
                {"image_id": image_id, "descriptor_type": descriptor_type, "symantics_type": symantics_type, "label":label},
                {"latent_symantics": 1, "image_id": 1}):
            source_latent_semantics = x.get("latent_symantics")
            print(source_latent_semantics)

        for y in collection.find({"descriptor_type": descriptor_type, "symantics_type": symantics_type, "label":label},
                                 {"latent_symantics": 1, "image_id": 1}):
            destination_latent_semantics = y.get("latent_symantics")
            print(destination_latent_semantics)
            similarity_vector_id = findsimilarity(source_latent_semantics, destination_latent_semantics)
            similarity_vector[y.get("image_id")] = similarity_vector_id

        decresing_similarity_vector = sorted(similarity_vector.items(), key=lambda kv: kv[1], reverse=True)[:7]
        # Writing similarity as a Json dump to a file
        with open(writeto + image_id + "_k-dimesnion-labelwise.json", "w") as fp:
            json.dump(decresing_similarity_vector, fp, indent=4, sort_keys=True)

        print("Successfully inserted into Output file... ")

    except Exception as e:
        traceback.print_exc()
        print("Error finding m related images")

    print("Completed finding m related images... ")


def findsimilarity(source_vector, dest_vector):
    dot_product = np.dot(source_vector, dest_vector)
    norm_a = np.linalg.norm(source_vector)
    norm_b = np.linalg.norm(dest_vector)
    return dot_product / (norm_a * norm_b)

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

def starter(feature_model, dimension_reduction, k, label,image_id,):
    config_object = Config()
    path = config_object.read_path()
    writeto = config_object.write_path()
    mongo_url = config_object.mongo_url()
    database_name = config_object.database_name()
    collection_name = config_object.metadata_collection_name()
    meta_collection_name= config_object.k_metadata_collection_name()

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
        meta_collection.remove()
        meta_collection.insert_many(helper(feature_model,dimension_reduction,k,label,collection,config_object))
        findmrelatedimages(feature_model,dimension_reduction,k,label,meta_collection,config_object,image_id)


    except Exception as e:
        traceback.print_exc()
        print("Connection refused... ")

    print("Task 4 Completed")