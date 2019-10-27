import numpy as np

from database import Database
from latentsymantics import LatentSymantics
from tabulate import tabulate


def starter(k):
    subject_similarities = []
    subject_ids = Database().retrieve_all_subject_ids()

    for subject_id in subject_ids:
        subject_similarities.append(
            np.array(Database().retrieve_subject_similarities(subject_id))
        )

    print(np.array(subject_similarities))

    latent_symantics_model, latent_symantics = LatentSymantics(
        np.array(subject_similarities), k, 3
    ).latent_symantics

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
        print("\n")
