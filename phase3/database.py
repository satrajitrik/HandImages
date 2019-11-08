import traceback

from config import Config
from pymongo import MongoClient


class Database(object):
	def __init__(self):
		self.database_name = Config().database_name()
		self.collection_name = Config().collection_name()
		self.mongo_url = Config().mongo_url()

	def __open_connection(self):
		try:
			connection = MongoClient(self.mongo_url)
			return connection
		except Exception as e:
			traceback.print_exc()
			print("Connection refused... ")
		return None

	def insert_many(self, records):
		connection = self.__open_connection()
		database = connection[self.database_name]
		collection = database[self.collection_name]

		collection.drop()
		collection.insert_many(records)
		connection.close()

		print("Successfully inserted into DB... ")

	def retrieve_many(self):
		connection = self.__open_connection()
		database = connection[self.database_name]
		collection = database[self.collection_name]

		query_results = collection.find({})
		connection.close()

		return [(item["id"], item["vector"]) for item in query_results]