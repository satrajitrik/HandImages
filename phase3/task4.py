import numpy as np
import pandas
import visualizer

from config import Config
from database import Database
from decisiontreeclassifier import DecisionTreeClassifier as dt
from basicsvm import SVMClassfier as SVM
import svm
from kernel import Kernel


def starter(classifiermodel):

    # retrive Training data from Database
    query_results = Database().retrieve_many(image_ids=None, collection_type="training")
    training_data = np.array([item["vector"] for item in query_results])
    labels = np.array([item["label"] for item in query_results])
    print(labels)

    if classifiermodel == 1:
        classifer = dt(3)
    elif classifiermodel == 2:
        #classifer = svm()
        classifer = svm.binary_classification_qp(kernel=Kernel.linear())
        labels = np.where(labels <= 0, -1, 1)
    elif classifiermodel ==3:
        classifer = dt(3)
    else:
        print("Enter valid classifier")

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
        if classifiermodel == 2:
            oy.append(1 if "dorsal" in temp else -1)
        else:
            oy.append(1 if "dorsal" in temp else 0)
        if oy[i] == y[i]:
            correct += 1
    print(images)
    print(oy)
    print("correctly classified : ", correct)
    print("Incorrectly classified : ", (m - correct))
    print("accuracy : ", (correct * 1.0) / (m * 1.0))

    visualizer.visualize_task4(images, y)
