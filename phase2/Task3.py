import functions
import numpy as np

from database import Database
from labels import Labels


def starter(feature_model, dimension_reduction, k, label_choice):
    label, value, _ = Labels(label_choice).label
    filtered_image_ids = [
        item["image_id"]
        for item in Database().retrieve_metadata_with_labels(label, value)
    ]

    _, latent_symantics = functions.store_in_db(
        feature_model, dimension_reduction, k, 3, filtered_image_ids, label, value
    )

    print(latent_symantics)
