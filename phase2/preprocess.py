import functions
import numpy as np
import pandas

from config import Config
from latentsymantics import LatentSymantics
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
"""


def insert_subjects_metadata_to_db(database, metadata):
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

        """
            1: Color moments,
            2: LBP,
            3: HOG,
            4: SIFT 
            in process_files(_, 1/2/3/4, _)
        """
        dorsal_image_vectors, _ = functions.process_files(
            Config().read_all_path(), 1, dorsal
        )
        print("For subject {}: Completed reading dorsal images... ".format(subject_id))
        palmar_image_vectors, _ = functions.process_files(
            Config().read_all_path(), 1, palmar
        )
        print("For subject {}: Completed reading palmar images... ".format(subject_id))

        """
            1: PCA,
            2: SVD,
            3: NMF,
            4: LDA
            in LatentSymantics(_, k, 1/2/3/4)
        """
        dorsal_latent_symantics = LatentSymantics(
            np.transpose(dorsal_image_vectors), 1, 1
        ).latent_symantics
        palmar_latent_symantics = LatentSymantics(
            np.transpose(palmar_image_vectors), 1, 1
        ).latent_symantics

        print(
            "Dorsal image vector shape: {}, Palmar image vector shape: {}".format(
                dorsal_latent_symantics.shape, palmar_latent_symantics.shape
            )
        )

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


if __name__ == "__main__":
    connection = MongoClient(Config().mongo_url())
    database = connection[Config().database_name()]
    metadata = pandas.read_csv(Config().metadata_file())

    # insert_metadata_to_db(database, metadata)
    insert_subjects_metadata_to_db(database, metadata)
