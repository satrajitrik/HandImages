import functions
import numpy as np
from tabulate import tabulate

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

    latent_symantics_model, latent_symantics = LatentSymantics(
        x, k, dimension_reduction
    ).latent_symantics

    records = functions.set_records(
        ids, descriptor_type, symantics_type, k, latent_symantics, pos
    )

    Database().insert_many(records)

    """
        Helisha's change
    """
    if dimension_reduction == 3 or dimension_reduction == 4:  # NMF, LDA
        term_weight_pairs = []
        latent_symantics_transpose = latent_symantics.transpose()
        weights = latent_symantics_model.components_
        for i in range(len(latent_symantics_transpose)):
            term_weight_pairs.append([latent_symantics_transpose[i], weights[i]])

        print(tabulate(term_weight_pairs, headers=['Term', 'Weight']))

        print("Latent topics are described in terms of top 50 features.")
        for index, latent_feature in enumerate(latent_symantics_model.components_):
            print("top 50 features for latent_topic #", index)
            print([i for i in latent_feature.argsort()[-50:]])

    else: # PCA,SVD
        term_weight_pairs = []
        latent_symantics_transpose = latent_symantics.transpose()
        weights = latent_symantics_model.explained_variance_ratio_

        for i in range(len(latent_symantics_transpose)):
            term_weight_pairs.append([latent_symantics_transpose[i], weights[i]])
        term_weight_pairs = sorted(term_weight_pairs,key=lambda l:l[1], reverse=True)
        print(tabulate(term_weight_pairs, headers=['Term', 'Weight']))
