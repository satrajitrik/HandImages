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

    def retrieve_many(self, task, label=None, value=None):
        connection = self.open_connection()
        database = connection[self.database_name]
        collection = database[self.collection_name]

        if label:
            query_results = collection.find({"$and": [{"task": task}, {label: value}]})
        else:
            query_results = collection.find({"task": task})
        connection.close()

        return [item for item in query_results]

    def retrieve_one(self, image_id, task):
        connection = self.open_connection()
        database = connection[self.database_name]
        collection = database[self.collection_name]

        query_results = collection.find_one(
            {"$and": [{"image_id": image_id}, {"task": task}]}
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

        return query_results

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

    def retrieve_all_subject_ids(self):
        connection = self.open_connection()
        database = connection[self.database_name]
        collection = database[Config().subjects_metadata_collection_name()]

        query_results = collection.find()
        connection.close()

        return sorted([item["subject_id"] for item in query_results])

    def get_subject_gender(self, subject_id):
        connection = self.open_connection()
        database = connection[self.database_name]
        collection = database[Config().subjects_metadata_collection_name()]

        subject_gender = collection.find_one({"subject_id": subject_id})["gender"]
        connection.close()

        return subject_gender

    def retrieve_subject_similarities(self, subject_id):
        connection = self.open_connection()
        database = connection[self.database_name]
        collection = database[Config().subjects_similarity_collection_name()]

        query_results = collection.find_one({"subject_id": subject_id})
        connection.close()

        return query_results["similarity_values"]
