"""
	Utility functions
"""
import cv2
import os
from descriptor import Descriptor


def process_files(path, feature_model, filtered_image_ids=None):
    files = os.listdir(path)

    ids, x = [], []
    for file in files:
        print("Reading file: {}".format(file))
        if not filtered_image_ids or (
            filtered_image_ids and file.replace(".jpg", "") in filtered_image_ids
        ):
            image = cv2.imread(path + file)

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
