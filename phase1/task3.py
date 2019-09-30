import math
import pickle
import traceback

import constants

from pymongo import MongoClient
from scipy.spatial import distance


def k_similar_features(feature_vector, query_results, model, k):
	distances = []

	for item in query_results:
		item_vector = pickle.loads(item[model.lower()])
		if model == "LBP":
			distances.append([item["name"], distance.euclidean(feature_vector, item_vector)])
		else:
			matches = []
			for vec in feature_vector:
				min_dist = min([distance.euclidean(vec, item_vec) for item_vec in item_vector])
				matches.append(min_dist)
			distances.append([item["name"], sum(matches) / len(matches)])

	return sorted(distances, key=lambda x: x[1])[:k]

def starter():
	image_id, model, k = input("Enter image ID, model name (LBP or SIFT) and k: ").split()

	constants_dict = constants.read_json()
	db_name = constants_dict["DATABASE_NAME"]
	collection_name = constants_dict["COLLECTION_NAME"]
	mongo_url = constants_dict["MONGO_URL"]

	try:
		conn = MongoClient(mongo_url)

		database = conn[db_name]
		collection = database[collection_name]

		feature_vector = None
		image_dict = collection.find_one({"name": image_id})
		if image_dict and model.lower() in image_dict:
			feature_vector = pickle.loads(image_dict[model.lower()])

		if feature_vector is not None:
			query_results = collection.find({"name": {"$ne": image_id}})
			print(k_similar_features(feature_vector, query_results, model, int(k)))
		else:
			print("Please enter correct values..... ")
		
		conn.close()

	except Exception as detail:
		traceback.print_exc()

	print("Done... ")
