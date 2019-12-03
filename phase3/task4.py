import numpy as np
import pandas as pd
import visualizer
from config import Config
from database import Database
from decisiontreeclassifier import DecisionTreeClassifier as dt
from basicsvm import SVMClassfier as SVM
import TkinterCheckbox
import svm
from kernel import Kernel


def starter(classifiermodel):

    # retrive Training data from Database
    query_results = Database().retrieve_many(image_ids=None, collection_type="training")
    training_data = np.array([item["vector"] for item in query_results])
    labels = np.array([item["label"] for item in query_results])
    #print(labels)

    if classifiermodel == 1:
        classifer = dt(3)
    elif classifiermodel == 2:
        #classifer = svm()
        #classifer = svm.binary_classification_qp(kernel=Kernel.quadratic())
        classifer = svm.binary_classification_qp(kernel=Kernel._polykernel(5)) # try 5 and 10 for dimensions in polykernel
        labels = np.where(labels <= 0, -1, 1)
    elif classifiermodel ==3:
        classifer = dt(3)
    else:
        print("Enter valid classifier")

    classifer.fit(training_data, labels)

    query_results = Database().retrieve_many(image_ids=None, collection_type="testing")
    testing_data = np.array([item["vector"] for item in query_results])

    y = classifer.predict(testing_data)
    #print(y)

    m = len(y)
    images = [item["image_id"] for item in query_results]

    metadata = pd.read_csv(Config().metadata_file())
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

    k_th_eigenvector_all = []
    for i in range(len(images)):
        arr=[]
        if(y[i]==1):
            val='dorsal'
        else:
            val='palmer'
        arr.append((images[i] + ".jpg",val))
        k_th_eigenvector_all.append(arr)
    print(k_th_eigenvector_all)
    k_th_eigenvector_all = pd.DataFrame(k_th_eigenvector_all)
    visualizer.visualize_svm_classifier(k_th_eigenvector_all,'SVM')


