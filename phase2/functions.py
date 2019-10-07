"""
	Utility functions
"""
import os
from queue import PriorityQueue

import cv2
from scipy.spatial import distance

from descriptor import Descriptor


def process_files(path, feature_model, filtered_image_ids=None):
    files = os.listdir(path)
    
    ids, x = [], []
    for file in files:
        print("Reading file: {}".format(file))
        if not filtered_image_ids or (
                    filtered_image_ids and file.replace(".jpg", "") in filtered_image_ids
        ):
            image = cv2.imread("{}{}".format(path, file))
            
            feature_descriptor = Descriptor(image, feature_model).feature_descriptor
            ids.append(file.replace(".jpg", ""))
            x.append(feature_descriptor)
    
    return x, ids


def set_records(ids, descriptor_type, symantics_type, k, latent_symantics):
    return [
        {
            "image_id": ids[i],
            "descriptor_type": descriptor_type,
            "symantics_type": symantics_type,
            "k": k,
            "latent_symantics": latent_symantics[i].tolist(),
            "male": -1,
            "dorsal": -1,
            "left_hand": -1,
            "accessories": -1,
        }
        for i in range(len(ids))
        ]


def sift_distance(source_vector, target_vector):
    if len(source_vector) != len(target_vector):
        raise ValueError("The dimensions of the arguments mismatch")
    matches = 0
    for vector1 in source_vector:
        vector_distances = PriorityQueue()
        for vector2 in target_vector:
            vector_distances.put(distance.euclidean_distance(vector1, vector2))
        min_distance = distance.get()
        second_min_distance = distance.get()
        if min_distance / second_min_distance < 0.8:
            matches += 1
    return float(matches)


def compare(target, others, m):
    others = [
        {"image_id": item["image_id"], "latent_symantics": item["latent_symantics"]}
        for item in others
        if item["image_id"] != target["image_id"]
        ]
    distances = [
        (
            other["image_id"],
            distance.euclidean(target["latent_symantics"], other["latent_symantics"]),
        )
        for other in others
        ]
    distances = sorted(distances, key=lambda x: x[1])
    
    return distances[:m]
