import json


class Config(object):
    def __init__(self):
        self.__json_file_path = "constants.json"
        with open(self.__json_file_path) as f:
            self.constants = json.load(f)

    def database_name(self):
        return self.constants.get("DATABASE_NAME")

    def collection_name(self):
        return self.constants.get("COLLECTION_NAME")

    def feedback_collection_name(self):
        return self.constants.get("FEEDBACK_COLLECTION_NAME")

    def training_collection_name(self):
        return self.constants.get("TRAINING_COLLECTION_NAME")

    def testing_collection_name(self):
        return self.constants.get("TESTING_COLLECTION_NAME")

    def page_rank_collection_name(self):
        return self.constants.get("PAGE_RANK_COLLECTION_NAME")

    def mongo_url(self):
        return self.constants.get("MONGO_URL")

    def read_all_path(self):
        """
        Reads all 11k images.
        """
        return self.constants.get("READ_ALL_PATH")

    def read_path(self):
        """
        
        """
        return self.constants.get("READ_PATH")

    def read_training_set1_path(self):
        return self.constants.get("READ_TRAINING_SET1_PATH")

    def read_training_set2_path(self):
        return self.constants.get("READ_TRAINING_SET2_PATH")

    def read_testing_set1_path(self):
        return self.constants.get("READ_TESTING_SET1_PATH")

    def read_testing_set2_path(self):
        return self.constants.get("READ_TESTING_SET2_PATH")

    def write_path(self):
        return self.constants.get("WRITE_PATH")

    def metadata_collection_name(self):
        return self.constants.get("METADATA_COLLECTION_NAME")

    def subjects_metadata_collection_name(self):
        return self.constants.get("SUBJECTS_METADATA_COLLECTION_NAME")

    def subjects_similarity_collection_name(self):
        return self.constants.get("SUBJECTS_SIMILARITY_COLLECTION_NAME")

    def metadata_file(self):
        return self.constants.get("METADATA_FILE")
