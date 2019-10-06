import functions
import numpy as np

from config import Config
from database import Database
from descriptor import DescriptorType
from latentsymantics import LatentSymantics, LatentSymanticsType


def get_latentsymantics_and_insert(feature_model, dimension_reduction, k):
    path = Config().read_path()

    x, ids = functions.process_files(path, feature_model)
    descriptor_type = DescriptorType(feature_model).descriptor_type
    symantics_type = LatentSymanticsType(dimension_reduction).symantics_type

    latent_symantics = LatentSymantics(
        np.array(x), k, dimension_reduction
    ).latent_symantics

    records = functions.set_records(
        ids, descriptor_type, symantics_type, k, latent_symantics
    )

    Database().insert_many(records)
    return latent_symantics


def starter(feature_model, dimension_reduction, k):
    print(get_latentsymantics_and_insert(feature_model, dimension_reduction, k))
    print("Done... ")
