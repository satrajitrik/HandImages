import numpy as np
import pandas
import visualizer
from config import Config
from database import Database
from decisiontreeclassifier import DecisionTreeClassifier as dt
import svm
from kernel import Kernel
from ppr import PageRank


def starter(classifier):
    if classifier == 1:

        query_results = Database().retrieve_many(
        image_ids=None, collection_type="training"
        )
        training_data = np.array([item["vector"] for item in query_results])
        labels = np.array([item["label"] for item in query_results])

        classifer = svm.binary_classification_qp(kernel=Kernel._polykernel(5)) # try 5 and 10 for dimensions in polykernel
        labels = np.where(labels <= 0, -1, 1)

        classifer.fit(training_data, labels)

        query_results = Database().retrieve_many(
        image_ids=None, collection_type="testing"
        )
        testing_data = np.array([item["vector"] for item in query_results])
        y = classifer.predict(testing_data)
        m = len(y)
        images = [item["image_id"] for item in query_results]
        metadata = pandas.read_csv(Config().metadata_file())
        oy = []
        correct = 0
        for i in range(len(images)):
            image_id = images[i] + ".jpg"
            temp = "".join(metadata[metadata.imageName == image_id]["aspectOfHand"].values)
            oy.append(1 if "dorsal" in temp else -1)
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
        k_th_eigenvector_all = pandas.DataFrame(k_th_eigenvector_all)
        visualizer.visualize_svm_classifier(k_th_eigenvector_all , 'SVM')

    elif classifier == 2:
        # retrive Training data from Database
        query_results = Database().retrieve_many(
            image_ids=None, collection_type="training"
        )
        training_data = np.array([item["vector"] for item in query_results])
        labels = np.array([item["label"] for item in query_results])

        classifer = dt(3)
        classifer.fit(training_data, labels)

        query_results = Database().retrieve_many(
            image_ids=None, collection_type="testing"
        )
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
            temp = "".join(
                metadata[metadata.imageName == image_id]["aspectOfHand"].values
            )
            oy.append(1 if "dorsal" in temp else 0)
            if oy[i] == y[i]:
                correct += 1
        print(images)
        print(oy)
        print("correctly classified : ", correct)
        print("Incorrectly classified : ", (m - correct))
        print("accuracy : ", (correct * 1.0) / (m * 1.0))

        visualizer.visualize_task4(images, y, "Decision Tree")

    elif classifier == 3:
        # PPR
        training_image_vectors = Database().retrieve_many(
            image_ids=None, collection_type="training"
        )
        testing_image_vectors = Database().retrieve_many(
            image_ids=None, collection_type="testing"
        )

        page_rank = PageRank(
            k=300,
            labelled_images=training_image_vectors,
            unlabelled_images=testing_image_vectors,
        )
        result = page_rank.label_images()
        print(result)
        metadata = pandas.read_csv(Config().metadata_file())
        # print(metadata['imageName'][0], metadata['aspectOfHand'][0])
        # print(metadata['imageName'].index('Hand_0000942'))
        # print("dorsal" in str(metadata[metadata['imageName'] == 'Hand_0000942.jpg']['aspectOfHand']))
        correct, incorrect = 0, 0
        imagename_list, aspect_list = [], []
        for imagename, aspect in result.items():
            imagename_list.append(imagename)
            aspect_list.append(aspect)
            if aspect in str(
                metadata[metadata["imageName"] == imagename + ".jpg"]["aspectOfHand"]
            ):
                correct += 1
            else:
                incorrect += 1
        print("Correct prediction = ", correct)
        print("Incorrect prediction = ", incorrect)
        print("Prediction accuracy = ", float(correct) / (incorrect + correct))
        # visualizer.visualize_task4(imagename_list, aspect_list, "PPR")


if __name__ == "__main__":
    starter(classifier=1)

