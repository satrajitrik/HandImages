import numpy as np

from database import Database
from latentsymantics import LatentSymantics


def starter(k):
    subject_similarities = []
    subject_ids = Database().retrieve_all_subject_ids()

    for subject_id in subject_ids:
        subject_similarities.append(
            np.array(Database().retrieve_subject_similarities(subject_id))
        )

    print(np.array(subject_similarities))

    latent_symantics_model, _ = LatentSymantics(
        np.array(subject_similarities), k, 3
    ).latent_symantics

    """
        According to Helisha's term weight code change
    """
    for index, latent_feature in enumerate(latent_symantics_model.components_):
        print("top 50 features for latent_topic #", index)
        print([i for i in latent_feature.argsort()[-50:]])
        print("\n")
