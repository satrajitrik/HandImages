import traceback

from config import Config
from pymongo import MongoClient


class Database(object):
    def __init__(self):
        self.database_name = Config().database_name()
        self.collection_name = Config().collection_name()
        self.mongo_url = Config().mongo_url()

    def open_connection(self):
        try:
            connection = MongoClient(self.mongo_url)
            return connection
        except Exception as e:
            traceback.print_exc()
            print("Connection refused... ")
        return None

    def insert_many(self, records):
        connection = self.open_connection()
        database = connection[self.database_name]
        collection = database[self.collection_name]

        collection.drop()
        collection.insert_many(records)
        connection.close()

        print("Successfully inserted into DB... ")

    def retrieve_many(self, descriptor_type, symantics_type, k, label=None, value=-1):
        connection = self.open_connection()
        database = connection[self.database_name]
        collection = database[self.collection_name]

        if label:
            query_results = collection.find(
                {
                    "$and": [
                        {"descriptor_type": descriptor_type},
                        {"symantics_type": symantics_type},
                        {"k": k},
                        {label: value},
                    ]
                }
            )
        else:
            query_results = collection.find(
                {
                    "$and": [
                        {"descriptor_type": descriptor_type},
                        {"symantics_type": symantics_type},
                        {"k": k},
                        {"male": -1},
                        {"dorsal": -1},
                        {"left_hand": -1},
                        {"accessories": -1},
                    ]
                }
            )
        connection.close()

        return [item for item in query_results]

    def retrieve_one(
        self, image_id, descriptor_type, symantics_type, k, label=None, value=-1
    ):
        connection = self.open_connection()
        database = connection[self.database_name]
        collection = database[self.collection_name]

        if label:
            query_results = collection.find_one(
                {
                    "$and": [
                        {"image_id": image_id},
                        {"descriptor_type": descriptor_type},
                        {"symantics_type": symantics_type},
                        {"k": k},
                        {label: value},
                    ]
                }
            )
        else:
            query_results = collection.find_one(
                {
                    "$and": [
                        {"image_id": image_id},
                        {"descriptor_type": descriptor_type},
                        {"symantics_type": symantics_type},
                        {"k": k},
                        {"male": -1},
                        {"dorsal": -1},
                        {"left_hand": -1},
                        {"accessories": -1},
                    ]
                }
            )
        connection.close()

        return query_results

    def retrieve_metadata_with_labels(self, label=None, value=None):
        connection = self.open_connection()
        database = connection[self.database_name]
        collection = database[Config().metadata_collection_name()]

        if label:
            query_results = collection.find({label: value})
        else:
            query_results = collection.find()
        connection.close()

        return [item["image_id"] for item in query_results]

    def retrieve_subjects(self, subject_id):
        """
        
        :param subject_id:
        :return:
        """
        connection = self.open_connection()
        database = connection[self.database_name]
        collection = database[Config().subjects_metadata_collection_name()]

        source_subject_info = collection.find_one({"subject_id": subject_id})
        other_subjects_info = collection.find({"subject_id": {"$ne": subject_id}})
        connection.close()

        return source_subject_info, other_subjects_info
