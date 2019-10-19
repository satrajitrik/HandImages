import json


class Config(object):
    def __init__(self):
        self.__json_file_path = "/Users/satrajitmaitra/HandImages/phase2/constants.json"
        with open(self.__json_file_path) as f:
            self.constants = json.load(f)

    def database_name(self):
        return self.constants.get("DATABASE_NAME")

    def collection_name(self):
        return self.constants.get("COLLECTION_NAME")

    def mongo_url(self):
        return self.constants.get("MONGO_URL")

    """
        Reads all 11k images.
    """

    def read_all_path(self):
        return self.constants.get("READ_ALL_PATH")

    """
        Reads 63 images.
    """

    def read_path(self):
        return self.constants.get("READ_PATH")

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
