import numpy as np
import pandas
import visualizer

from config import Config
from database import Database
from decisiontreeclassifier import DecisionTreeClassifier as dt


def starter():

    # retrive Training data from Database
    query_results = Database().retrieve_many(image_ids=None, collection_type="training")
    training_data = np.array([item["vector"] for item in query_results])
    labels = np.array([item["label"] for item in query_results])

    classifer = dt(3)
    classifer.fit(training_data, labels)

    query_results = Database().retrieve_many(image_ids=None, collection_type="testing")
    testing_data = np.array([item["vector"] for item in query_results])

    y = classifer.predict(testing_data)
    print(y)

    m = len(y)
    images = [item["image_id"] for item in query_results]

    metadata = pandas.read_csv(Config().metadata_file())
    oy = []
    correct = 0
    for i in range(len(images)):
        image_id = images[i] + ".jpg"
        temp = "".join(metadata[metadata.imageName == image_id]["aspectOfHand"].values)
        oy.append(1 if "dorsal" in temp else 0)
        if oy[i] == y[i]:
            correct += 1
    print(images)
    print(oy)
    print("correctly classified : ", correct)
    print("Incorrectly classified : ", (m - correct))
    print("accuracy : ", (correct * 1.0) / (m * 1.0))

    visualizer.visualize_task4(images, y)
