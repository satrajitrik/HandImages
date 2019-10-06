import json


class Config(object):
    def __init__(self):
        self.__json_file_path = "/Users/satrajitmaitra/HandImages/phase2/constants.json"
        self.__metadata_collection_name = "metadata"
        with open(self.__json_file_path) as f:
            self.constants = json.load(f)

    def database_name(self):
        return self.constants.get("DATABASE_NAME")

    def collection_name(self):
        return self.constants.get("COLLECTION_NAME")

    def mongo_url(self):
        return self.constants.get("MONGO_URL")

    def read_path(self):
        return self.constants.get("READ_PATH")

    def write_path(self):
        return self.constants.get("WRITE_PATH")

    def metadata_collection_name(self):
        return self.__metadata_collection_name

    def metadata_file(self):
        return self.constants.get("METADATA_FILE")
