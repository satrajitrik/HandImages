"""
    Renamed scripts.py to preprocess.py
"""
from gridfs import GridFS

import functions
import numpy as np
import pandas
import json

from config import Config
from pymongo import MongoClient


"""
    Required prior to running Tasks 3 & 4.
"""


def insert_metadata_to_db(database, metadata):
    collection = database[Config().metadata_collection_name()]

    collection.drop()
    for index, row in metadata.iterrows():
        image_name = row["imageName"].replace(".jpg", "")
        male = 1 if row["gender"] == "male" else 0
        dorsal = 1 if "dorsal" in row["aspectOfHand"] else 0
        left_hand = 1 if "left" in row["aspectOfHand"] else 0
        accessories = row["accessories"]

        output = collection.insert_one(
            {
                "image_id": image_name,
                "male": male,
                "dorsal": dorsal,
                "left_hand": left_hand,
                "accessories": accessories,
            }
        )
        if not output.acknowledged:
            print("ERROR: Could not save ", row)
            exit(1)
    print("Inserted Metadata for labels: ", len(metadata), "rows")


"""
    Required prior to running Task 6. Storing the gender, list of dorsal image ids and
    list of palmar image ids for a given subject.

    Parameters:
    1. metadata: pandas dataframe representing HandInfo.csv
    2. feature_model: {
        1: Color moments,
        2: LBP,
        3: HOG,
        4: SIFT (Not to be used because of uneven length complexity)
    }

    NOTE: Tried using Color moments which gave decent results. 
    Set feature_model to 2 or 3 and check results.
"""


def insert_subjects_metadata_to_db(database, metadata, feature_model=2):
    """
    
    :param database:
    :param metadata:
    :param feature_model:
    :return:
    """
    collection = database[Config().subjects_metadata_collection_name()]

    collection.drop()
    id_gender_pairs = set(tuple(x) for x in metadata[["id", "gender"]].values.tolist())
    for subject_id, gender in id_gender_pairs:
        dorsal = (
            metadata[
                (metadata.id == subject_id)
                & (metadata.aspectOfHand.str.contains("dorsal"))
            ]["imageName"]
            .str.replace(".jpg", "")
            .tolist()
        )
        palmar = (
            metadata[
                (metadata.id == subject_id)
                & (metadata.aspectOfHand.str.contains("palmar"))
            ]["imageName"]
            .str.replace(".jpg", "")
            .tolist()
        )

        dorsal_image_vectors, _ = functions.process_files(
            Config().read_all_path(), feature_model, dorsal
        )
        print("For subject {}: Completed reading dorsal images... ".format(subject_id))
        palmar_image_vectors, _ = functions.process_files(
            Config().read_all_path(), feature_model, palmar
        )
        print("For subject {}: Completed reading palmar images... ".format(subject_id))
        
        grid_fs = GridFS(database=database, collection=Config().subjects_metadata_collection_name())
        with grid_fs.new_file() as dorsal_file:
            dorsal_file.write(json.dumps(dorsal_image_vectors.tolist()).encode('utf-8'))
        with grid_fs.new_file() as palmar_file:
            palmar_file.write(json.dumps(palmar_image_vectors.tolist()).encode('utf-8'))
        output = collection.insert_one(
            {
                "subject_id": subject_id,
                "gender": gender,
                "dorsal": dorsal_file._id,
                "palmar": palmar_file._id,
            }
        )
        if not output.acknowledged:
            print("ERROR: Could not save ", subject_id)
            exit(1)
    print("Inserted Metadata for subjects: ", len(metadata), "rows")


if __name__ == "__main__":
    connection = MongoClient(Config().mongo_url())
    database = connection[Config().database_name()]
    metadata = pandas.read_csv(Config().metadata_file())

    insert_metadata_to_db(database, metadata)
    insert_subjects_metadata_to_db(database, metadata)
