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

    print(LatentSymantics(np.array(subject_similarities), k, 3).latent_symantics)
