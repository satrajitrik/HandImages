"""
    Utility functions
"""
import json
import math
import numpy as np
import os
from queue import PriorityQueue

import cv2
from gridfs import GridFS
from pymongo import MongoClient
from scipy.spatial import distance

from config import Config
from descriptor import Descriptor, DescriptorType
from latentsymantics import LatentSymantics


def process_files(path, feature_model, filtered_image_ids=None):
    files = os.listdir(path)

    ids, x = [], []
    for file in files:
        if not filtered_image_ids or (
            filtered_image_ids and file.replace(".jpg", "") in filtered_image_ids
        ):
            print("Reading file: {}".format(file))
            image = cv2.imread("{}{}".format(path, file))

            feature_descriptor = Descriptor(image, feature_model).feature_descriptor
            ids.append(file.replace(".jpg", ""))
            x.append(feature_descriptor)

    if DescriptorType(feature_model).check_sift():
        """
    	    For SIFT, we flatten the image descriptor array into an array of keypoints.
    	    We return an extra list (pos) representing the number of keypoints for each image. 
    	    This is done to extract the feature descriptors (after dimensionality reduction) of 
    	    each image correctly while inserting into the DB.
    	"""
        sift_x, pos = x[0], [x[0].shape[0]]
        for i in range(1, len(x)):
            pos.append(x[i].shape[0])
            sift_x = np.vstack((sift_x, x[i]))
        return sift_x, ids, pos

    """
        For all other feature descriptors, return only the ids and descriptor array.
    """
    return np.array(x), ids


def set_records(ids, descriptor_type, symantics_type, k, latent_symantics, pos):
    records = []
    f, prev_start = 1 if pos else 0, 0

    for i in range(len(ids)):
        record = {
            "image_id": ids[i],
            "descriptor_type": descriptor_type,
            "symantics_type": symantics_type,
            "k": k,
            "male": -1,
            "dorsal": -1,
            "left_hand": -1,
            "accessories": -1,
        }
        if f == 1:
            """
                For SIFT.
            """
            start, end = prev_start, prev_start + pos[i]
            record["latent_symantics"] = latent_symantics[start:end].tolist()
            prev_start = end
        else:
            """
                For all other feature descriptors.
            """
            record["latent_symantics"] = latent_symantics[i].tolist()
        records.append(record)

    return records


"""
    Similarity method for SIFT. To be updated. Gives okayish results.
"""


def sift_distance(source_vector, target_vector):
    matches = 0
    for vector1 in source_vector:
        vector_distances = PriorityQueue()
        for vector2 in target_vector:
            vector_distances.put(distance.euclidean(vector1, vector2))
        min_distance = vector_distances.get()
        second_min_distance = vector_distances.get()
        if min_distance / second_min_distance > 0.9:
            matches += 1
    return float(matches) / len(source_vector)


def cm_distance(source_vector, target_vector):
    weight = [2, 1, 1, 2, 1, 1, 2, 1, 1]
    weighted_source_vector = []
    weighted_dest_vector = []
    for i in range(0, len(source_vector), 9):
        till_index = len(source_vector) - i
        lower_range = min(9, till_index)
        weighted_source_vector.extend(
            [weight[j] * source_vector[i + j] for j in range(lower_range)]
        )

    for i in range(0, len(target_vector), 9):
        till_index = len(target_vector) - i
        lower_range = min(9, till_index)
        weighted_dest_vector.extend(
            [weight[j] * target_vector[i + j] for j in range(lower_range)]
        )

    return distance.euclidean(weighted_source_vector, weighted_dest_vector)


"""
    Might cause an issue
"""


def distance_to_similarity(distances):
    return [[id, 1 / (1 + distance ** (0.25))] for id, distance in distances]


def compare(source, targets, m, descriptor_type):
    targets = [
        {"image_id": item["image_id"], "latent_symantics": item["latent_symantics"]}
        for item in targets
        if item["image_id"] != source["image_id"]
    ]

    distances = []
    for target in targets:
        image_distance_info = [target["image_id"]]

        if descriptor_type == "sift":
            image_distance_info.append(
                sift_distance(source["latent_symantics"], target["latent_symantics"])
            )
        elif descriptor_type == "cm":
            image_distance_info.append(
                cm_distance(source["latent_symantics"], target["latent_symantics"])
            )
        else:
            image_distance_info.append(
                distance.euclidean(
                    source["latent_symantics"], target["latent_symantics"]
                )
            )
        distances.append(image_distance_info)

    return sorted(distance_to_similarity(distances), key=lambda x: x[1], reverse=True)[
        :m
    ]


"""
    NOTE: If using LBP, use GridFS over here to extract vectors
"""


def concatenate_latent_symantics(subject, k, choice):
    connection = MongoClient(Config().mongo_url())
    database = connection[Config().database_name()]

    grid_fs = GridFS(
        database=database, collection=Config().subjects_metadata_collection_name()
    )
    with grid_fs.get(subject["dorsal"]) as dorsal_file:
        dorsal_image_vectors = json.loads(dorsal_file.read().decode("utf-8"))
    with grid_fs.get(subject["palmar"]) as palmar_file:
        palmar_image_vectors = json.loads(palmar_file.read().decode("utf-8"))

    _, dorsal_latent_symantics = LatentSymantics(
        np.transpose(dorsal_image_vectors), k, choice
    ).latent_symantics
    _, palmar_latent_symantics = LatentSymantics(
        np.transpose(palmar_image_vectors), k, choice
    ).latent_symantics

    dorsal_latent_symantics = [
        x for item in dorsal_latent_symantics.tolist() for x in item
    ]
    palmar_latent_symantics = [
        x for item in palmar_latent_symantics.tolist() for x in item
    ]
    return np.concatenate(
        (np.array(dorsal_latent_symantics), np.array(palmar_latent_symantics))
    )
