import pickle
import traceback

from bson.binary import Binary
from pymongo import MongoClient
from scipy.spatial import distance


DATABASE_NAME = "mwdb"
COLLECTION_NAME = "hands"

def k_similar_features(feature_vector, query_results, model, k):
	distances = []

	for item in query_results:
		if model == "LBP":
			item_vector = pickle.loads(item[model.lower()])
			distances.append([item["name"], distance.euclidean(feature_vector, item_vector)])
		else:
			item_vector = pickle.loads(item[model.lower()])
			matches = []
			for vec in feature_vector:
			    min_dist = distance.euclidean(vec, item_vector[0])
			    for i in range(1, len(item_vector)):
			        dist = distance.euclidean(vec, item_vector[i])
			        if min_dist > dist:
			            min_dist = dist
			    matches.append(min_dist)
			distances.append([item["name"], sum(matches)])

	return sorted(distances, key=lambda x: x[1])[:k]

def main():
	image_id, model, k = input("Enter image ID, model name (LBP or SIFT) and k: ").split()

	try:
		conn = MongoClient('mongodb://localhost:27017/')

		database = conn[DATABASE_NAME]
		collection = database[COLLECTION_NAME]

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

if __name__ == '__main__':
	main()
