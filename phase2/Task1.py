import functions
import numpy as np
from tabulate import tabulate


def starter(feature_model, dimension_reduction, k):
    latent_symantics_model, latent_symantics = functions.store_in_db(
        feature_model, dimension_reduction, k, 1
    )
    if dimension_reduction == 3 or dimension_reduction == 4:  # NMF, LDA
        term_weight_pairs = []
        latent_symantics_transpose = latent_symantics.transpose()
        weights = latent_symantics_model.components_
        for i in range(len(latent_symantics_transpose)):
            term_weight_pairs.append([latent_symantics_transpose[i], weights[i]])

        print(tabulate(term_weight_pairs, headers=["Term", "Weight"]))

        print("Latent topics are described in terms of top 50 features.")
        for index, latent_feature in enumerate(latent_symantics_model.components_):
            print("top 50 features for latent_topic #", index)
            print([i for i in latent_feature.argsort()[-50:]])

    else:  # PCA,SVD
        term_weight_pairs = []
        latent_symantics_transpose = latent_symantics.transpose()
        weights = latent_symantics_model.explained_variance_ratio_

        for i in range(len(latent_symantics_transpose)):
            term_weight_pairs.append([latent_symantics_transpose[i], weights[i]])
        term_weight_pairs = sorted(term_weight_pairs, key=lambda l: l[1], reverse=True)
        print(tabulate(term_weight_pairs, headers=["Term", "Weight"]))
