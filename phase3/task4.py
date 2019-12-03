import numpy as np
import pandas
import visualizer

from config import Config
from database import Database
from decisiontreeclassifier import DecisionTreeClassifier as dt
from ppr import PageRank


def starter(classifier):
    if classifier == 1:
        # SVM
        pass
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
        # accuracy_list = []
        # k_list = []
        # for i in range(3, 180):
        page_rank = PageRank(
        k=8,
        labelled_images=training_image_vectors,
        unlabelled_images=testing_image_vectors,
        )
        result = page_rank.label_images()
        metadata = pandas.read_csv(Config().metadata_file())
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
            # k_list.append(i)
            # accuracy_list.append(float(correct) / (incorrect + correct))
        # print(k_list)
        # print(accuracy_list)
        # visualizer.visualize_task4(imagename_list, aspect_list, "PPR")


if __name__ == "__main__":
    starter(classifier=3)
