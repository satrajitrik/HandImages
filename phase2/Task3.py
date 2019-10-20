import functions
import numpy as np

from config import Config
from database import Database
from descriptor import DescriptorType
from latentsymantics import LatentSymantics, LatentSymanticsType
from labels import Labels


def starter(feature_model, dimension_reduction, k, label_choice):
    path, pos = Config().read_path(), None
    label, value, _ = Labels(label_choice).label
    descriptor_type = DescriptorType(feature_model).descriptor_type
    symantics_type = LatentSymanticsType(dimension_reduction).symantics_type

    filtered_image_ids = [
        item["image_id"]
        for item in Database().retrieve_metadata_with_labels(label, value)
    ]

    if DescriptorType(feature_model).check_sift():
        x, ids, pos = functions.process_files(path, feature_model, filtered_image_ids)
    else:
        x, ids = functions.process_files(path, feature_model, filtered_image_ids)

    _, latent_symantics = LatentSymantics(x, k, dimension_reduction).latent_symantics

    records = functions.set_records(
        ids, descriptor_type, symantics_type, k, latent_symantics, pos
    )
    for record in records:
        record[label] = value

    Database().insert_many(records)

    print(latent_symantics)
    print("Done... ")

