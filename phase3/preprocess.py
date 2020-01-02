import cv2
import functions
import numpy as np
import os

from config import Config
from database import Database
from descriptor import Descriptor
from latentsymantics import LatentSymantics


def insert_images_in_database(
    feature_model, dimension_reduction, k, identifier, set1_dir=True, set2_dir=True
):
    """
    :param feature_model: 1 - CM, 2 - LBP, 3 - HOG, 4 - SIFT
    :param dimension_reduction: 1 - PCA, 2 - SVD, 3 - NMF, 4 - LDA
    :param k: reduced dimension value
    :param identifier: 0 - Read all, 1 - Read from Labelled, 2 - Read from Unlabelled
    :param set1_dir (Optional): True - Read from Set1 folder of Labelled/Unlabelled, False otherwise
    :param set2_dir (Optional): True - Read from Set2 folder of Labelled/Unlabelled, False otherwise
    :return None

    Default case: Read from both Set1 and Set2 folders
    """

    # Read images and feature extraction
    if identifier == 0:
        read_all_path = Config().read_all_path()
        files = os.listdir(read_all_path)
        connection = Database().open_connection()
        db = connection[Config().database_name()]
        collection = db[Config().collection_name()]

        for i, file in enumerate(files):
            print(
                "Reading file: {} | {} % Done".format(
                    file, ((i + 1) * 100.0) / len(files)
                )
            )
            image = cv2.imread("{}{}".format(read_all_path, file))

            feature_descriptor = Descriptor(
                image, feature_model, dimension_reduction
            ).feature_descriptor
            image_id = file.replace(".jpg", "")
            collection.insert_one(
                {"image_id": image_id, "vector": feature_descriptor.tolist()}
            )

        connection.close()
        query_results = Database().retrieve_many()
        ids = [item["image_id"] for item in query_results]
        x = np.array([item["vector"] for item in query_results])

    elif identifier == 1:
        if set1_dir and set2_dir:
            ids1, x1 = functions.process_files(
                Config().read_training_set1_path(), feature_model, dimension_reduction
            )
            ids2, x2 = functions.process_files(
                Config().read_training_set2_path(), feature_model, dimension_reduction
            )
            ids = ids1 + ids2
            x = np.concatenate((x1, x2))
        elif set1_dir:
            ids, x = functions.process_files(
                Config().read_training_set1_path(), feature_model, dimension_reduction
            )
        elif set2_dir:
            ids, x = functions.process_files(
                Config().read_training_set2_path(), feature_model, dimension_reduction
            )
    else:
        if set1_dir and set2_dir:
            ids1, x1 = functions.process_files(
                Config().read_testing_set1_path(), feature_model, dimension_reduction
            )
            ids2, x2 = functions.process_files(
                Config().read_testing_set2_path(), feature_model, dimension_reduction
            )
            ids = ids1 + ids2
            x = np.concatenate((x1, x2))
        elif set1_dir:
            ids, x = functions.process_files(
                Config().read_testing_set1_path(), feature_model, dimension_reduction
            )
        elif set2_dir:
            ids, x = functions.process_files(
                Config().read_testing_set2_path(), feature_model, dimension_reduction
            )

    # Find Latent_symantics
    _, latent_symantics = LatentSymantics(x, k, dimension_reduction).latent_symantics

    # inserting data into Database
    if identifier == 0:
        records = functions.set_records(ids, latent_symantics)
        Database().insert_many(records)
    elif identifier == 1:
        records = functions.set_records(ids, latent_symantics, training=True)
        Database().insert_many(records, collection_type="training")
    else:
        records = functions.set_records(ids, latent_symantics)
        Database().insert_many(records, collection_type="testing")
    print("Done... ")


if __name__ == "__main__":
    # insert_images_in_database(
    #     feature_model=1, dimension_reduction=1, k=256, identifier=0
    # )

    # store training folder
    insert_images_in_database(
        feature_model=1, dimension_reduction=1, k=100, identifier=1,set2_dir=False
    )

    # store testing folder
    insert_images_in_database(
        feature_model=1, dimension_reduction=1, k=100, identifier=2,set2_dir=False
    )
