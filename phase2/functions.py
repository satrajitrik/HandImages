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
from database import Database
from descriptor import Descriptor, DescriptorType
from latentsymantics import LatentSymantics, LatentSymanticsType


def process_files(path, feature_model, dimension_reduction, filtered_image_ids=None):
    files = os.listdir(path)

    ids, x = [], []
    for file in files:
        if not filtered_image_ids or (
            filtered_image_ids and file.replace(".jpg", "") in filtered_image_ids
        ):
            print("Reading file: {}".format(file))
            image = cv2.imread("{}{}".format(path, file))

            feature_descriptor = Descriptor(
                image, feature_model, dimension_reduction
            ).feature_descriptor
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


def set_records(
    ids,
    descriptor_type,
    symantics_type,
    k,
    latent_symantics,
    pos,
    task,
    label=None,
    value=None,
):
    records = []
    f, prev_start = 1 if pos else 0, 0

    for i in range(len(ids)):
        record = {
            "image_id": ids[i],
            "descriptor_type": descriptor_type,
            "symantics_type": symantics_type,
            "task": task,
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

    if label:
        for record in records:
            record[label] = value

    return records


def store_in_db(
    feature_model,
    dimension_reduction,
    k,
    task,
    filtered_image_ids=None,
    label=None,
    value=None,
):
    path, pos = Config().read_path(), None
    descriptor_type = DescriptorType(feature_model).descriptor_type
    symantics_type = LatentSymanticsType(dimension_reduction).symantics_type

    if DescriptorType(feature_model).check_sift():
        x, ids, pos = process_files(
            path, feature_model, dimension_reduction, filtered_image_ids
        )
    else:
        x, ids = process_files(
            path, feature_model, dimension_reduction, filtered_image_ids
        )

    latent_symantics_model, latent_symantics = LatentSymantics(
        x, k, dimension_reduction
    ).latent_symantics

    records = set_records(
        ids,
        descriptor_type,
        symantics_type,
        k,
        latent_symantics,
        pos,
        task,
        label,
        value,
    )

    Database().insert_many(records)

    return latent_symantics_model, latent_symantics


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


def subject_similarity(source_subject, other_subjects, k=1, choice=1):
    """
        Idea is to extract all dorsal and palmar feature descriptors for a given subject.
    
        Dorsal/Palmar feature vectors are represented as numpy arrays of the form 
        ((# of dorsal/palmar images for a subject) X (feature descriptor length)) 
        where the # of rows is variable but the # of columns is constant for a given feature descriptor.
        We take the transpose of these feature vector matrices to represent all dorsal/palmar images 
        as features instead of objects for a given subject and apply dimensionality reduction 
        to reduce the number of features to 1 feature/image which best represents the subject. 
        After applying dimensionality reduction, we get 1 dorsal latent symantics matrix and 1 
        palmar latent symantics matrix. The shape of these matrices are same. ie. (1 X (feature descriptor length))
        We now concatenate the dorsal and palmar latent symantics and apply cosime similarity to
        compare two subjects to get the result.
        Parameters:
        k = 1 (Reduced dimension)
        choice: {
	        1: PCA,
	        2: SVD,
	        3: NMF,
	        4: LDA
        }
    """
    distances = []
    source_latent_symantics = concatenate_latent_symantics(source_subject, k, choice)

    for subject in other_subjects:
        other_latent_symantics = concatenate_latent_symantics(subject, k, choice)

        dist = distance.cosine(source_latent_symantics, other_latent_symantics)
        distances.append([subject["subject_id"], dist])

    distances.append([source_subject["subject_id"], 0])
    return sorted(distance_to_similarity(distances), key=lambda x: x[0])
