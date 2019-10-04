from config import Config
from pymongo import MongoClient


class Database(object):
    def __init__(self):
        self.database_name = Config().database_name()
        self.collection_name = Config().collection_name()
        self.mongo_url = Config().mongo_url()

    def insert_many(self, records):
        try:
            connection = MongoClient(self.mongo_url)
            database = connection[self.database_name]
            collection = database[self.collection_name]

            collection.insert_many(records)
            print("Successfully inserted into DB... ")
            connection.close()
        except Exception as e:
            traceback.print_exc()
            print("Connection refused... ")

    def retrieve_many(self, descriptor_type, symantics_type, k):
        try:
            connection = MongoClient(self.mongo_url)
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
        except Exception as e:
            traceback.print_exc()
            print("Connection refused... ")

            return None

    def retrieve_one(self, image_id, descriptor_type, symantics_type, k):
        try:
            connection = MongoClient(self.mongo_url)
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
        except Exception as e:
            traceback.print_exc()
            print("Connection refused... ")

            return None
