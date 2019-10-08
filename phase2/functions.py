"""
	Utility functions
"""
import numpy as np
import os
from queue import PriorityQueue

import cv2
from scipy.spatial import distance

from descriptor import Descriptor, DescriptorType


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
        else:
            image_distance_info.append(
                distance.euclidean(
                    source["latent_symantics"], target["latent_symantics"]
                )
            )
        distances.append(image_distance_info)

    return sorted(distances, key=lambda x: x[1])[:m]
