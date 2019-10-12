import numpy as np

from config import Config
from database import Database
from latentsymantics import LatentSymantics
from scipy.spatial import distance


def compare(source_subject, other_subjects):
    """
	    1: PCA,
	    2: SVD,
	    3: NMF,
	    4: LDA
	    in LatentSymantics(_, k, 1/2/3/4)
	"""
    source_dorsal_latent_symantics = LatentSymantics(
        np.transpose(source_subject["dorsal"]), 1, 1
    ).latent_symantics
    source_palmar_latent_symantics = LatentSymantics(
        np.transpose(source_subject["palmar"]), 1, 1
    ).latent_symantics

    distances = []
    for subject in other_subjects:
        if subject["gender"] == source_subject["gender"]:
            other_dorsal_latent_symantics = LatentSymantics(
                np.transpose(subject["dorsal"]), 1, 1
            ).latent_symantics
            other_palmar_latent_symantics = LatentSymantics(
                np.transpose(subject["palmar"]), 1, 1
            ).latent_symantics

            dorsal_distance = distance.euclidean(
                source_dorsal_latent_symantics, other_dorsal_latent_symantics
            )
            palmar_distance = distance.euclidean(
                source_palmar_latent_symantics, other_palmar_latent_symantics
            )

            average_distance = float(dorsal_distance + palmar_distance) / 2

            distances.append([subject["subject_id"], average_distance])

    distances = sorted(distances, key=lambda x: x[1])
    return distances[:3]


def starter(subject_id):
    source_subject, other_subjects = Database().retrieve_subjects(subject_id)

    print(compare(source_subject, other_subjects))
