import numpy as np

from config import Config
from database import Database
from latentsymantics import LatentSymantics
from scipy.spatial import distance


"""
    Idea is to extract all dorsal and palmar feature descriptors for a given subject.
    
    dorsal/palmar feature vectors are represented as numpy arrays of the form 
    ((# of dorsal/palmar images for a subject) X (feature descriptor length)) 
    where the # of rows is variable but the # of columns is constant for a given feature descriptor.

    We take the transpose of these feature vector matrices to represent all dorsal/palmar images 
    as features instead of objects for a given subject and apply dimensionality reduction 
    to reduce the number of features to 1 feature/image which best represents the subject. 

    After applying dimensionality reduction, we get 1 dorsal latent symantics matrix and 1 
    palmar latent symantics matrix. The shape of these matrices are same. ie. (1 X (feature descriptor length))

    We now concatenate the dorsal and palmar latent symantics and apply cosime similarity to
    compare two subjects to get the result.

    Parameters:
    k = 1 (Reduced dimension)
    choice: {
	    1: PCA,
	    2: SVD,
	    3: NMF,
	    4: LDA
    }

    NOTE: Checked using Color moments with PCA. You guys can check for others.
"""


def compare(source_subject, other_subjects, k=1, choice=1):
    source_dorsal_latent_symantics = LatentSymantics(
        np.transpose(source_subject["dorsal"]), k, choice
    ).latent_symantics
    source_palmar_latent_symantics = LatentSymantics(
        np.transpose(source_subject["palmar"]), k, choice
    ).latent_symantics

    source_dorsal_latent_symantics = [
        x for item in source_dorsal_latent_symantics.tolist() for x in item
    ]
    source_palmar_latent_symantics = [
        x for item in source_palmar_latent_symantics.tolist() for x in item
    ]
    source_latent_symantics = np.concatenate(
        (
            np.array(source_dorsal_latent_symantics),
            np.array(source_palmar_latent_symantics),
        )
    )

    distances = []
    for subject in other_subjects:
        if subject["gender"] == source_subject["gender"]:
            other_dorsal_latent_symantics = LatentSymantics(
                np.transpose(subject["dorsal"]), k, choice
            ).latent_symantics
            other_palmar_latent_symantics = LatentSymantics(
                np.transpose(subject["palmar"]), k, choice
            ).latent_symantics

            other_dorsal_latent_symantics = [
                x for item in other_dorsal_latent_symantics.tolist() for x in item
            ]
            other_palmar_latent_symantics = [
                x for item in other_palmar_latent_symantics.tolist() for x in item
            ]
            other_latent_symantics = np.concatenate(
                (
                    np.array(other_dorsal_latent_symantics),
                    np.array(other_palmar_latent_symantics),
                )
            )

            dist = distance.cosine(source_latent_symantics, other_latent_symantics)

            distances.append([subject["subject_id"], dist])

    distances = sorted(distances, key=lambda x: x[1])
    return distances[:3]


def starter(subject_id):
    source_subject, other_subjects = Database().retrieve_subjects(subject_id)

    print(compare(source_subject, other_subjects))
