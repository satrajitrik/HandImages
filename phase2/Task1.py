import functions
import numpy as np

from config import Config
from database import Database
from descriptor import DescriptorType
from latentsymantics import LatentSymantics, LatentSymanticsType


def starter(feature_model, dimension_reduction, k):
    path, pos = Config().read_path(), None
    descriptor_type = DescriptorType(feature_model).descriptor_type
    symantics_type = LatentSymanticsType(dimension_reduction).symantics_type

    if DescriptorType(feature_model).check_sift():
        x, ids, pos = functions.process_files(path, feature_model)
    else:
        x, ids = functions.process_files(path, feature_model)

    latent_symantics_model,latent_symantics = LatentSymantics(x, k, dimension_reduction).latent_symantics

    records = functions.set_records(
        ids, descriptor_type, symantics_type, k, latent_symantics, pos
    )

    Database().insert_many(records)


    # term-weight pair
    if dimension_reduction == 3 : #NMF
        for index,latent_feature in enumerate(latent_symantics_model.components_):
            print("top 50 features for latent_topic #",index)
            print([i for i in latent_feature.argsort()[-50:]])
            print("\n")
    else:
        term_weight_pairs = []
        latent_symantics_transpose = latent_symantics.transpose()
        weights = latent_symantics_model.explained_variance_

        for i in range(len(latent_symantics_transpose)):
            term_weight_pairs.append([latent_symantics_transpose[i] , weights[i]])
        print( term_weight_pairs )
