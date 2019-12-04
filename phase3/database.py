import traceback
import pandas as pd
import pymongo
from bson.binary import Binary
import pickle
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

    def find_label_from_metadata(self, image_id):
        connection = self.open_connection()
        database = connection[self.database_name]
        collection = database[Config().metadata_collection_name()]
        query_Results = collection.find_one({"image_id": image_id})
        if query_Results['dorsal'] == 0:
            return 0
        else:
            return 1

    def insert_many(self, records, collection_type='sample_test'):
        """
        :param records: List of records to insert
        :param collection_type (Optional): training - insert into hands_train, testing - insert into hands_test, null - insert into hands
        :return None
        """
        connection = self.open_connection()
        database = connection[self.database_name]

        if collection_type == "training":
            collection = database[Config().training_collection_name()]
        elif collection_type == "testing":
            collection = database[Config().testing_collection_name()]
        elif collection_type == "sample_test":
            collection = database[Config().sample_test_name()]
        else:
            collection = database[self.collection_name]

        collection.drop()
        collection.insert_many(records)
        connection.close()

        print("Successfully inserted into DB... ")

    def retrieve_many(self, image_ids= None, collection_type="sample_test"):
        """
        :param image_ids (Optional): List of image IDs as a filter while querying
        :param collection_type (Optional): training - query from hands_train, testing - query from hands_test, null - query from hands
        :return list(hash)
        """
        connection = self.open_connection()
        database = connection[self.database_name]
        all = 0

        if collection_type == "training":
            collection = database[Config().training_collection_name()]
        elif collection_type == "testing":
            collection = database[Config().testing_collection_name()]
        elif collection_type == "sample_test":
            collection = database[Config().sample_test_name()]
        else:
            all = 1
            collection = database[self.collection_name]

        if image_ids:
            if all == 0:
                query_results = collection.find({"image_id": {"$in": image_ids}}).sort(
                    "_id.getTimestamp()", pymongo.ASCENDING
                )
            else:
                query_results = collection.find({"image_id": {"$in": image_ids}})
        else:
            if all == 0:
                query_results = collection.find({}).sort(
                    "_id.getTimestamp()", pymongo.ASCENDING
                )
            else:
                query_results = collection.find({})

        connection.close()

        return list(query_results)

    def retrieve_many_t1(self, task, label=None, value=None):
        connection = self.open_connection()
        database = connection[self.database_name]
        collection = database[Config().sample_test_name()]

        if label:
            query_results = collection.find({"$and": [{"task": task}, {label: value}]})
        else:
            query_results = collection.find({"task": task})
        connection.close()

        return [item for item in query_results]

    def retrieve_one_t1(self,task, image_id,):
        connection = self.open_connection()
        database = connection[self.database_name]
        collection = database[Config().sample_test_name()]

        query_results = collection.find_one(
            {"$and": [{"image_id": image_id}, {"task": task}]}
        )
        connection.close()

        return query_results

    def retrieve_one(self, image_id):
        connection = self.open_connection()
        database = connection[self.database_name]
        collection = database[self.collection_name]

        query_result = collection.find_one({"image_id": image_id})
        connection.close()

        return query_result

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

    def insert_binary(self, key, value, collection_type=None):
        connection = self.open_connection()
        database = connection[self.database_name]
        if collection_type == "page_rank":
            collection = database[Config().page_rank_collection_name()]
            collection.update_one(
                {"graph_size": key},
                {"$set": {"graph": Binary(pickle.dumps(value))}},
                upsert=True,
            )

    def retrieve_binary(self, key, collection_type=None):
        connection = self.open_connection()
        database = connection[self.database_name]
        if collection_type == "page_rank":
            collection = database[Config().page_rank_collection_name()]
            graph = pickle.loads(collection.find_one({"graph_size": key})["graph"])
            return graph

    def store_feedback(self, image_id, feedback):
        connection = self.open_connection()
        database = connection[self.database_name]
        collection = database[Config().feedback_collection_name()]

        relevant_list = [id for id, response in feedback if response == "y"]
        irrelevant_list = [id for id, response in feedback if response == "n"]

        collection.update(
            {"image_id": image_id},
            {
                "$push": {
                    "relevant": {"$each": relevant_list},
                    "irrelevant": {"$each": irrelevant_list},
                }
            },
            upsert=True,
        )
        connection.close()

        print("Successfully inserted into DB... ")

    def retrieve_feedback(self, image_id):
        connection = self.open_connection()
        database = connection[self.database_name]
        collection = database[Config().feedback_collection_name()]

        query_result = collection.find_one({"image_id": image_id})
        connection.close()

        return query_result["relevant"], query_result["irrelevant"]
