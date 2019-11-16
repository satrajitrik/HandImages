import functions
import numpy as np
import os

from config import Config
from database import Database
from latentsymantics import LatentSymantics
from imageprocessor import ImageProcessor


def preprocess_images():
    ids, input_vector = ImageProcessor().id_vector_pair
    records = [
        {"id": ids[i], "vector": input_vector[i].tolist()} for i in range(len(ids))
    ]

    Database().insert_many(records)


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
        ids, x = functions.process_files(
            Config().read_all_path(), feature_model, dimension_reduction
        )
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


if __name__ == "__main__":
    # preprocess_images()

    # store training folder
    insert_images_in_database(
        feature_model=1, dimension_reduction=1, k=50, identifier=1
    )

    # store testing folder
    insert_images_in_database(
        feature_model=1, dimension_reduction=1, k=50, identifier=2
    )
