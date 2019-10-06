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

        collection.insert_many(records)
        connection.close()

        print("Successfully inserted into DB... ")

    def retrieve_many(self, descriptor_type, symantics_type, k):
        connection = self.open_connection()
        database = connection[self.database_name]
        collection = database[self.collection_name]

        query_results = collection.find(
            {
                "$and": [
                    {"descriptor_type": descriptor_type},
                    {"symantics_type": symantics_type},
                    {"k": k},
                ]
            }
        )
        connection.close()

        return [item for item in query_results]

    def retrieve_one(self, image_id, descriptor_type, symantics_type, k):
        connection = self.open_connection()
        database = connection[self.database_name]
        collection = database[self.collection_name]

        query_results = collection.find_one(
            {
                "$and": [
                    {"image_id": image_id},
                    {"descriptor_type": descriptor_type},
                    {"symantics_type": symantics_type},
                    {"k": k},
                ]
            }
        )
        connection.close()

        return query_results

    def retrieve_with_labels(self, label, value):
        connection = self.open_connection()
        database = connection[self.database_name]
        collection = database[Config().metadata_collection_name()]

        query_results = collection.find({label: value})

        return [item["image_id"] for item in query_results]
