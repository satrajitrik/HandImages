import cv2
import numpy as np
import os
import pandas as pd

from config import Config
from descriptor import Descriptor
from scipy.spatial import distance


def calculate_similarity(source_vector, target_vector):
    return 1 / (1 + distance.euclidean(source_vector, target_vector) ** (0.25))


def find_similarity(source_vector, target_vectors, m):
    similarities = [
        (id, calculate_similarity(source_vector, vector))
        for id, vector in target_vectors
    ]

    return sorted(similarities, key=lambda x: x[1], reverse=True)[:m]


def process_files(path, feature_model, dimension_reduction):
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

    return ids, np.array(x)


def set_records(ids, vectors, training=False):
    records = []

    for i in range(len(ids)):
        record = {"image_id": ids[i], "vector": vectors[i].tolist()}
        if training:
            image_id = ids[i] + ".jpg"
            metadata = pd.read_csv(Config().metadata_file())
            y = "".join(metadata[metadata.imageName == image_id]["aspectOfHand"].values)
            record["label"] = 1 if "dorsal" in y else 0
        records.append(record)

    return records
