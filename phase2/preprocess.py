"""
    Run this script first before anything else
"""

import functions
import numpy as np
import pandas
import Task6

from config import Config
from database import Database
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


def insert_subjects_metadata_to_db(database, metadata, feature_model=1):
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

        output = collection.insert_one(
            {
                "subject_id": subject_id,
                "gender": gender,
                "dorsal": dorsal_image_vectors.tolist(),
                "palmar": palmar_image_vectors.tolist(),
            }
        )
        if not output.acknowledged:
            print("ERROR: Could not save ", subject_id)
            exit(1)
    print("Inserted Metadata for subjects: ", len(metadata), "rows")


def insert_subject_similarities_to_db(database):
    collection = database[Config().subjects_similarity_collection_name()]
    subject_ids = Database().retrieve_all_subject_ids()

    collection.drop()
    for subject_id in subject_ids:
        print("Processing subject: {}".format(subject_id))
        source_subject, other_subjects = Database().retrieve_subjects(subject_id)
        similarity_values = [
            similarity
            for _, similarity in Task6.compare(source_subject, other_subjects, 7)
        ]
        output = collection.insert_one(
            {"subject_id": subject_id, "similarity_values": similarity_values}
        )
        if not output.acknowledged:
            print("ERROR: Could not save ", subject_id)
            exit(1)
        print("Completed processing subject: {}".format(subject_id))
    print("Inserted similarity values for subjects: ", len(subject_ids), "rows")


if __name__ == "__main__":
    connection = MongoClient(Config().mongo_url())
    database = connection[Config().database_name()]
    metadata = pandas.read_csv(Config().metadata_file())

    """
        NOTE: Order must be maintained for the following function calls
    """
    insert_metadata_to_db(database, metadata)
    insert_subjects_metadata_to_db(database, metadata)
    insert_subject_similarities_to_db(database)
