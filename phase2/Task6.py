import numpy as np

from database import Database


def starter(subject_id):
    source_subject, _ = Database().retrieve_subjects(subject_id)

    subject_ids = Database().retrieve_all_subject_ids()
    subject_similarities = Database().retrieve_subject_similarities(subject_id)
    subject_similarity_info = [
        [id, subject_similarities[i]]
        for i, id in enumerate(subject_ids)
        if subject_id != id
        and source_subject["gender"] == Database().get_subject_gender(id)
    ]

    print(sorted(subject_similarity_info, key=lambda x: x[1], reverse=True)[:3])
