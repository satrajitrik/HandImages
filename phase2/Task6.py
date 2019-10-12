from config import Config
from database import Database


def starter(subject_id):
    source_subject, other_subjects = Database().retrieve_subjects(subject_id)

    print([other_subject for other_subject in other_subjects])
